import argparse
import csv
import importlib
import json
import platform
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from lab2.matrix_cases import load_matrix


FIELDS = [
    "size", "method", "time_seconds", "repeats", "multiplications",
    "additions_subtractions", "divisions", "flops", "max_absolute_error",
    "relative_reconstruction_error", "normalized_residual", "lower_triangular",
    "upper_triangular", "unit_lower_diagonal", "matches_reference", "A_sha256",
]
ALGORITHM = "recursive_triangular_solves_tiled_v2"


def benchmark(sizes, methods, implementation="compiled", repeats=2, resume=False,
              time_budget=600, output_dir=None, data_dir=None):
    root = Path(__file__).parent
    output_dir = Path(output_dir) if output_dir is not None else root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("lab2.lu_compiled" if implementation == "compiled" else "lab2.lu")
    output_path = output_dir / f"lu_{implementation}.csv"
    completed = {}
    if resume and output_path.exists():
        implementation_path = output_path.with_suffix(".implementation.json")
        if not implementation_path.exists() or json.loads(implementation_path.read_text())["algorithm"] != ALGORITHM:
            raise ValueError("Existing LU measurements use a different algorithm; archive them before restarting.")
        with output_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing CSV uses a different format.")
            completed = {(int(row["size"]), row["method"]): row for row in reader}
    else:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDS).writeheader()
        output_path.with_suffix(".implementation.json").write_text(
            json.dumps({"algorithm": ALGORITHM}, indent=2) + "\n", encoding="utf-8")

    started = datetime.now().astimezone()
    metadata = {
        "started_at": started.isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "implementation": implementation, "module": module.__file__,
        "sizes": sorted(set(sizes)), "methods": methods, "small_repeats": repeats,
        "large_repeats": 1, "large_threshold": 128, "row_permutations": False,
        "relative_error_tolerance": 1e-8, "normalized_residual_tolerance": 1e-10,
        "unary_negations_counted": False, "complete": False,
        "algorithm": ALGORITHM,
    }
    if implementation == "compiled":
        import Cython
        metadata["cython"] = Cython.__version__
    metadata_path = output_dir / f"lu_{implementation}_environment_{started:%Y%m%d_%H%M%S_%f}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Using {module.__file__}", flush=True)
    batch_start = perf_counter()
    interrupted = False
    try:
        for size in sorted(set(sizes)):
            A, checksum = load_matrix(size, data_dir)
            norm_A = np.linalg.norm(A, ord=np.inf)
            for method in methods:
                key = size, method
                if key in completed:
                    if completed[key]["A_sha256"] != checksum or completed[key]["matches_reference"] != "1":
                        raise ValueError(f"Cannot reuse invalid or mismatched result: {key}")
                    continue
                if perf_counter() - batch_start >= time_budget:
                    interrupted = True
                    break
                count = repeats if size <= 128 else 1
                print(f"Starting LU {method}, n={size}, repeats={count}", flush=True)
                times = []
                for _ in range(count):
                    start = perf_counter()
                    L, U, operations = module.lu(A, method)
                    times.append(perf_counter() - start)
                difference = L @ U - A
                residual = np.linalg.norm(difference, ord=np.inf)
                relative = residual / norm_A
                normalized = residual / (np.linalg.norm(L, ord=np.inf) * np.linalg.norm(U, ord=np.inf) + norm_A)
                lower = bool(np.all(np.triu(L, 1) == 0))
                upper = bool(np.all(np.tril(U, -1) == 0))
                unit = bool(np.all(np.diag(L) == 1))
                matching = bool(relative <= 1e-8 and normalized <= 1e-10 and lower and upper and unit)
                row = dict(zip(FIELDS, [
                    size, method, median(times), count, operations.multiplications,
                    operations.additions_subtractions, operations.divisions, operations.total,
                    np.max(np.abs(difference)), relative, normalized,
                    int(lower), int(upper), int(unit), int(matching), checksum,
                ]))
                with output_path.open("a", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=FIELDS).writerow(row)
                print(f"n={size}, LU {method}: {row['time_seconds']:.6f} s, FLOPs={operations.total}, match={matching}", flush=True)
                if not matching:
                    raise ArithmeticError(f"LU accuracy check failed for {method}, n={size}; diagnostics saved in {output_path}")
            if interrupted:
                break
        metadata["complete"] = not interrupted
    except Exception as error:
        metadata["error"] = str(error)
        raise
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {output_path}; complete={not interrupted}", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128])
    parser.add_argument("--methods", choices=["binet", "strassen"], nargs="+", default=["binet", "strassen"])
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.time_budget <= 0:
        parser.error("Sizes, repeats and time budget must be positive.")
    benchmark(args.sizes, args.methods, args.implementation, args.repeats, args.resume, args.time_budget)
