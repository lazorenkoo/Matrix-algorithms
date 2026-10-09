import csv
from pathlib import Path
from time import perf_counter

from binet import binet, generate_matrix


def benchmark(max_size: int = 20) -> list[tuple[int, float, int, int, int]]:
    results = []

    for n in range(1, max_size + 1):
        A = generate_matrix(n)
        B = generate_matrix(n)

        start = perf_counter()
        C, multiplications, additions = binet(A, B)
        elapsed = perf_counter() - start

        flops = multiplications + additions
        results.append((n, elapsed, multiplications, additions, flops))
        print(f"n={n}, time={elapsed:.6f} s, FLOPs={flops}")

    return results


if __name__ == "__main__":
    results = benchmark()
    output_path = Path(__file__).with_name("binet_results.csv")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["size", "time_seconds", "multiplications", "additions", "flops"])
        writer.writerows(results)

    print(f"Results saved to {output_path}")
