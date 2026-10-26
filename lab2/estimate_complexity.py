import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


FIT_FIELDS = [
    "task", "method", "refinement_steps", "metric", "status", "min_size",
    "max_size", "points", "exponent_p", "constant_C", "r_squared_log",
    "slope_standard_error", "min_adjacent_p", "max_adjacent_p",
]
LOCAL_FIELDS = ["task", "method", "refinement_steps", "metric", "size_from", "size_to", "local_p"]


def estimate(implementation="compiled", min_size=32, max_size=None, min_time=0.001,
             results_dir=None, tasks=("inverse", "lu")):
    results_dir = Path(results_dir) if results_dir is not None else Path(__file__).parent / "results"
    fits = []
    local_slopes = []
    for task in tasks:
        path = results_dir / f"{task}_{implementation}.csv"
        if not path.exists():
            print(f"Skipping missing results: {path}")
            continue
        groups = defaultdict(dict)
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                size = int(row["size"])
                if row["matches_reference"] != "1" or size < min_size or (max_size is not None and size > max_size):
                    continue
                key = row["method"], int(row.get("refinement_steps", 0))
                if size in groups[key]:
                    raise ValueError(f"Duplicate size {size} for {key} in {path}")
                groups[key][size] = row
        for (method, refinement), rows in sorted(groups.items()):
            for metric in ("time_seconds", "flops"):
                samples = [(n, float(row[metric])) for n, row in sorted(rows.items())
                           if np.isfinite(float(row[metric])) and float(row[metric]) > 0
                           and (metric != "time_seconds" or float(row[metric]) >= min_time)]
                base = dict(task=task, method=method, refinement_steps=refinement, metric=metric,
                            min_size=samples[0][0] if samples else "",
                            max_size=samples[-1][0] if samples else "", points=len(samples))
                if len(samples) < 3:
                    fits.append(dict(base, status="insufficient_points"))
                    print(f"{task}/{method}/{metric}: need at least 3 points in the selected range")
                    continue
                sizes, values = np.array(samples, dtype=float).T
                x, y = np.log(sizes), np.log(values)
                design = np.column_stack((x, np.ones(len(x))))
                exponent, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
                residual = y - (exponent * x + intercept)
                error_sum = float(residual @ residual)
                total_sum = float(np.sum((y - y.mean()) ** 2))
                r_squared = 1 - error_sum / total_sum if total_sum > 0 else float("nan")
                standard_error = np.sqrt(error_sum / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
                adjacent = np.diff(y) / np.diff(x)
                fits.append(dict(base, status="ok", exponent_p=exponent, constant_C=np.exp(intercept),
                                 r_squared_log=r_squared, slope_standard_error=standard_error,
                                 min_adjacent_p=float(adjacent.min()), max_adjacent_p=float(adjacent.max())))
                for i, value in enumerate(adjacent):
                    local_slopes.append(dict(task=task, method=method, refinement_steps=refinement,
                                             metric=metric, size_from=int(sizes[i]),
                                             size_to=int(sizes[i + 1]), local_p=value))
                print(f"{task}/{method}/{metric}: empirical p={exponent:.3f}, R2(log)={r_squared:.4f}, "
                      f"n={int(sizes[0])}..{int(sizes[-1])}, points={len(sizes)}")
    if not fits:
        print("No eligible measurements; no estimates were produced.")
        return None
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in [
        (f"complexity_{implementation}.csv", FIT_FIELDS, fits),
        (f"complexity_local_{implementation}.csv", LOCAL_FIELDS, local_slopes),
    ]:
        with (results_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print("These are finite-range experimental fits, not proofs of asymptotic complexity.")
    return results_dir / f"complexity_{implementation}.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--min-time", type=float, default=0.001)
    args = parser.parse_args()
    if args.min_size < 1 or args.min_time < 0 or (args.max_size is not None and args.max_size < args.min_size):
        parser.error("Invalid fitting range.")
    estimate(args.implementation, args.min_size, args.max_size, args.min_time)
