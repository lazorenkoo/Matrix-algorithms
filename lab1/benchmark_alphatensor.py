import argparse
from pathlib import Path

from alphatensor import alphatensor, coefficients
from benchmark_strassen import benchmark_strassen as benchmark_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark AlphaTensor using saved matrices.")
    parser.add_argument("--compiled", action="store_true", help="Use the native alphatensor_compiled module and separate results.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128, 256, 512],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    args = parser.parse_args()
    algorithm = alphatensor
    algorithm_name = "alphatensor"
    load_coefficients = coefficients
    if args.compiled:
        from alphatensor_compiled import alphatensor as algorithm
        from alphatensor_compiled import coefficients as load_coefficients
        import alphatensor_compiled

        print(f"Compiled module: {alphatensor_compiled.__file__}", flush=True)
        algorithm_name = "alphatensor_compiled"
    load_coefficients()
    output_path = Path(__file__).with_name(f"{algorithm_name}_results.csv")
    benchmark_saved(
        args.sizes, output_path, args.resume, args.time_budget,
        algorithm=algorithm, algorithm_name=algorithm_name,
    )
    print(f"Results saved to {output_path}")
