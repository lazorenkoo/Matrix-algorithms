import argparse
import csv
import gc
from pathlib import Path
from time import perf_counter
import tracemalloc

from binet import binet, generate_matrix


def benchmark_memory(
    sizes: list[int], output_path: Path, resume: bool = False,
    time_budget: float = 600,
) -> None:
    completed = set()
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as file:
            completed = {int(row["size"]) for row in csv.DictReader(file)}
    else:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "size", "peak_traced_memory_mib", "input_buffers_mib",
                "multiplications", "additions", "flops",
            ])

    batch_start = perf_counter()
    for n in sorted(set(sizes)):
        if n in completed:
            continue
        if perf_counter() - batch_start >= time_budget:
            print("Time budget reached; completed results are saved.", flush=True)
            break

        print(f"Starting memory measurement n={n}", flush=True)
        A = generate_matrix(n)
        B = generate_matrix(n)
        input_buffers_mib = (A.nbytes + B.nbytes) / (1024 ** 2)
        gc.collect()

        tracemalloc.start()
        try:
            C, multiplications, additions = binet(A, B)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mib = peak_bytes / (1024 ** 2)
        flops = multiplications + additions
        with output_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                n, peak_mib, input_buffers_mib,
                multiplications, additions, flops,
            ])
        print(f"n={n}, peak traced memory={peak_mib:.6f} MiB, FLOPs={flops}", flush=True)
        del A, B, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Binet memory separately from timing.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 200, 256],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    args = parser.parse_args()
    output_path = Path(__file__).with_name("binet_memory_results.csv")
    benchmark_memory(args.sizes, output_path, args.resume, args.time_budget)
    print(f"Memory results saved to {output_path}")
