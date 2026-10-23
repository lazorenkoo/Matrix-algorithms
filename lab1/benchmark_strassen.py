import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from matrix_cases import case_directory, load_matrix_pair
from strassen import strassen


def benchmark_strassen(
    sizes: list[int], output_path: Path, resume: bool = False,
    time_budget: float = 600,
    algorithm=strassen, algorithm_name: str = "strassen",
    small_repeats: int | None = None,
) -> None:
    completed = set()
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as file:
            completed = {int(row["size"]) for row in csv.DictReader(file)}
    else:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "size", "time_seconds", "multiplications",
                "additions" if algorithm_name.startswith("binet") else "additions_subtractions",
                "flops", "repeats", "max_absolute_error", "matches_octave",
            ])

    batch_start = perf_counter()
    for n in sorted(set(sizes)):
        if n in completed:
            continue
        if perf_counter() - batch_start >= time_budget:
            print("Time budget reached; completed results are saved.", flush=True)
            break

        repeats = 5 if n <= 32 else 3 if n <= 128 else 1
        if small_repeats is not None:
            repeats = small_repeats if n <= 128 else 1
        print(f"Starting {algorithm_name} n={n}, repeats={repeats}", flush=True)
        A, B = load_matrix_pair(n)
        times = []
        for _ in range(repeats):
            start = perf_counter()
            C, multiplications, additions = algorithm(A, B)
            times.append(perf_counter() - start)

        elapsed = median(times)
        flops = multiplications + additions
        folder = case_directory(n)
        reference = np.loadtxt(folder / "C_octave_matlab.csv", delimiter=",", ndmin=2)
        max_error = np.max(np.abs(C - reference))
        matching = np.allclose(C, reference, atol=1e-12, rtol=1e-12)
        np.savetxt(folder / f"C_{algorithm_name}.csv", C, delimiter=",", fmt="%.17g")

        with output_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                n, elapsed, multiplications, additions, flops,
                repeats, max_error, int(matching),
            ])
        print(
            f"n={n}, time={elapsed:.6f} s, FLOPs={flops}, "
            f"Octave match={matching}, error={max_error:.6e}", flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Strassen using saved comparison matrices.")
    parser.add_argument("--compiled", action="store_true", help="Use the native strassen_compiled module and separate results.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128, 256, 512],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    args = parser.parse_args()
    algorithm = strassen
    algorithm_name = "strassen"
    if args.compiled:
        from strassen_compiled import strassen as algorithm
        import strassen_compiled

        print(f"Compiled module: {strassen_compiled.__file__}", flush=True)
        algorithm_name = "strassen_compiled"
    output_path = Path(__file__).with_name(f"{algorithm_name}_results.csv")
    benchmark_strassen(
        args.sizes, output_path, args.resume, args.time_budget,
        algorithm=algorithm, algorithm_name=algorithm_name,
    )
    print(f"Results saved to {output_path}")
