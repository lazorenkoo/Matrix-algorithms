import argparse
from pathlib import Path

from alphatensor import alphatensor, coefficients
from benchmark_strassen import benchmark_strassen as benchmark_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark AlphaTensor using saved matrices.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128, 256, 512],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    args = parser.parse_args()
    coefficients()
    output_path = Path(__file__).with_name("alphatensor_results.csv")
    benchmark_saved(
        args.sizes, output_path, args.resume, args.time_budget,
        algorithm=alphatensor, algorithm_name="alphatensor",
    )
    print(f"Results saved to {output_path}")
