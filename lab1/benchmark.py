import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

from binet import binet, generate_matrix
from matrix_cases import load_matrix_pair


def benchmark(
    max_size: int = 128,
    repeats: int = 5,
    output_path: Path | None = None,
    sizes: list[int] | None = None,
    resume: bool = False,
    adaptive: bool = False,
    time_budget: float | None = None,
    algorithm=binet,
    saved_inputs: bool = False,
) -> list[tuple[int, float, int, int, int]]:
    results = []

    if resume and output_path is not None and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                results.append((
                    int(row["size"]), float(row["time_seconds"]),
                    int(row["multiplications"]), int(row["additions"]),
                    int(row["flops"]),
                ))
    elif output_path is not None:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["size", "time_seconds", "multiplications", "additions", "flops"])

    completed_sizes = {row[0] for row in results}
    requested_sizes = sorted(set(sizes)) if sizes is not None else range(1, max_size + 1)
    batch_start = perf_counter()

    for n in requested_sizes:
        if n in completed_sizes:
            continue
        if time_budget is not None and perf_counter() - batch_start >= time_budget:
            print("Time budget reached; completed results are saved.", flush=True)
            break

        size_repeats = repeats
        if adaptive:
            size_repeats = 7 if n <= 100 else 3 if n <= 200 else 1

        print(f"Starting n={n}, repeats={size_repeats}", flush=True)
        if saved_inputs:
            A, B = load_matrix_pair(n)
        else:
            A = generate_matrix(n)
            B = generate_matrix(n)

        times = []

        for _ in range(size_repeats):
            start = perf_counter()
            C, multiplications, additions = algorithm(A, B)
            times.append(perf_counter() - start)

        elapsed = median(times)

        flops = multiplications + additions
        row = (n, elapsed, multiplications, additions, flops)
        results.append(row)

        if output_path is not None:
            with output_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(row)

        print(f"n={n}, median time={elapsed:.6f} s, FLOPs={flops}", flush=True)

    return sorted(results, key=lambda row: row[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Binet multiplication time and FLOPs.")
    parser.add_argument("--compiled", action="store_true", help="Use the native binet_compiled module and separate results.")
    parser.add_argument(
        "--max-size", type=int, default=128,
        help="Measure every matrix size from 1 to this value (default: 128).",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of measurements per size (default: 5).",
    )
    parser.add_argument("--sizes", type=int, nargs="+", help="Measure selected sizes instead of a range.")
    parser.add_argument(
        "--practical", action="store_true",
        help="Measure 1..128 plus 160, 200, 256 with adaptive repeats and a 600 s budget.",
    )
    parser.add_argument("--resume", action="store_true", help="Keep existing CSV rows and skip completed sizes.")
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use 7 repeats up to size 100, 3 up to 200, and 1 above 200.",
    )
    parser.add_argument(
        "--time-budget", type=float,
        help="Time budget in seconds, checked between sizes.",
    )
    args = parser.parse_args()

    if args.practical:
        args.sizes = list(range(1, 129)) + [160, 200, 256]
        args.adaptive = True
        if args.time_budget is None:
            args.time_budget = 600

    algorithm = binet
    result_name = "binet"
    if args.compiled:
        from binet_compiled import binet as algorithm
        import binet_compiled

        print(f"Compiled module: {binet_compiled.__file__}", flush=True)
        result_name = "binet_compiled"
    output_path = Path(__file__).with_name(f"{result_name}_results.csv")
    results = benchmark(
        max_size=args.max_size, repeats=args.repeats, output_path=output_path,
        sizes=args.sizes, resume=args.resume, adaptive=args.adaptive,
        time_budget=args.time_budget,
        algorithm=algorithm, saved_inputs=args.compiled,
    )

    print(f"Results saved to {output_path}")
