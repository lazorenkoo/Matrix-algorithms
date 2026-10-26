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
    "size", "method", "time_seconds", "repeats", "refinement_steps", "multiplications",
    "additions_subtractions", "divisions", "flops", "max_absolute_error",
    "relative_inverse_error", "left_residual_inf", "right_residual_inf",
    "normalized_residual", "condition_inf", "matches_reference", "A_sha256",
]


def benchmark(sizes, methods, implementation="compiled", repeats=2, resume=False,
              time_budget=600, output_dir=None, data_dir=None, refinement_steps=1):
    root = Path(__file__).parent
    output_dir = Path(output_dir) if output_dir is not None else root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("lab2.inverse_compiled" if implementation == "compiled" else "lab2.inverse")
    output_path = output_dir / f"inverse_{implementation}.csv"
    completed = {}
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as file:
            if (reader := csv.DictReader(file)).fieldnames != FIELDS:
                raise ValueError("Existing CSV uses a different format.")
            completed = {(int(row["size"]), row["method"]): row for row in reader}
    else:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDS).writeheader()

    started = datetime.now().astimezone()
    metadata = {
        "started_at": started.isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "implementation": implementation, "module": module.__file__,
        "sizes": sorted(set(sizes)), "methods": methods, "small_repeats": repeats,
        "large_repeats": 1, "large_threshold": 128,
        "relative_error_tolerance": 1e-8, "normalized_residual_tolerance": 1e-10,
        "unary_negations_counted": False,
        "refinement_steps": refinement_steps,
    }
    if implementation == "compiled":
        import Cython
        metadata["cython"] = Cython.__version__
    metadata_path = output_dir / f"inverse_{implementation}_environment_{started:%Y%m%d_%H%M%S_%f}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Using {module.__file__}", flush=True)
    batch_start = perf_counter()
    interrupted = False
    for size in sorted(set(sizes)):
        A, checksum = load_matrix(size, data_dir)
        reference = None
        for method in methods:
            key = size, method
            if key in completed:
                if (completed[key]["A_sha256"] != checksum or completed[key]["matches_reference"] != "1"
                        or int(completed[key]["refinement_steps"]) != refinement_steps):
                    raise ValueError(f"Cannot reuse invalid or mismatched result: {key}")
                continue
            if perf_counter() - batch_start >= time_budget:
                interrupted = True
                break
            if reference is None:
                reference = np.linalg.inv(A)
                identity = np.eye(size)
                norm_A = np.linalg.norm(A, ord=np.inf)
                norm_reference = np.linalg.norm(reference, ord=np.inf)
            count = repeats if size <= 128 else 1
            print(f"Starting {method}, n={size}, repeats={count}", flush=True)
            times = []
            for _ in range(count):
                start = perf_counter()
                result, operations = module.inverse(A, method, refinement_steps=refinement_steps)
                times.append(perf_counter() - start)
            relative_error = np.linalg.norm(result - reference, ord=np.inf) / norm_reference
            left = np.linalg.norm(A @ result - identity, ord=np.inf)
            right = np.linalg.norm(result @ A - identity, ord=np.inf)
            normalized = max(left, right) / (norm_A * np.linalg.norm(result, ord=np.inf) + 1)
            matching = bool(relative_error <= 1e-8 and normalized <= 1e-10)
            row = dict(zip(FIELDS, [
                size, method, median(times), count, refinement_steps, operations.multiplications,
                operations.additions_subtractions, operations.divisions, operations.total,
                np.max(np.abs(result - reference)), relative_error, left, right,
                normalized, norm_A * norm_reference, int(matching), checksum,
            ]))
            with output_path.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=FIELDS).writerow(row)
            print(f"n={size}, {method}: {row['time_seconds']:.6f} s, FLOPs={operations.total}, match={matching}", flush=True)
            if not matching:
                raise ArithmeticError(f"Accuracy check failed for {method}, n={size}; diagnostics saved in {output_path}")
        if interrupted:
            break
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata["complete"] = not interrupted
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {output_path}; complete={not interrupted}", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128])
    parser.add_argument("--methods", choices=["binet", "strassen", "laderman"], nargs="+", default=["binet", "strassen"])
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=600)
    parser.add_argument("--refinement-steps", type=int, default=1)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.time_budget <= 0 or args.refinement_steps < 0:
        parser.error("Sizes, repeats and time budget must be positive.")
    benchmark(args.sizes, args.methods, args.implementation, args.repeats, args.resume, args.time_budget,
              refinement_steps=args.refinement_steps)
