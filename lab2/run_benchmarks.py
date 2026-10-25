import argparse
import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lab2.benchmark_inverse import benchmark as benchmark_inverse
from lab2.benchmark_lu import benchmark as benchmark_lu
from lab2.estimate_complexity import estimate
from lab2.matrix_cases import load_matrix


def run(sizes, implementation="compiled", repeats=2, time_budget=3600,
        refinement_steps=1, min_fit_size=32, tasks=("inverse", "lu")):
    root = Path(__file__).parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(sizes))
    checksums = {n: load_matrix(n)[1] for n in sizes}
    started = datetime.now().astimezone()
    metadata = {
        "started_at": started.isoformat(), "implementation": implementation,
        "python": platform.python_version(), "platform": platform.platform(),
        "sizes": sizes, "tasks": list(tasks), "repeats_up_to_128": repeats,
        "repeats_above_128": 1, "refinement_steps": refinement_steps,
        "matrix_checksums": checksums, "time_budget_seconds": time_budget,
        "resume": True, "complete": False,
    }
    metadata_path = results_dir / f"run_{implementation}_{started:%Y%m%d_%H%M%S_%f}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    batch_start = perf_counter()
    try:
        completed_tasks = []
        for task in tasks:
            remaining = time_budget - (perf_counter() - batch_start)
            if remaining <= 0:
                print("Time budget reached; saved results can be resumed.", flush=True)
                break
            methods = ["binet", "strassen", "laderman"] if task == "inverse" else ["binet", "strassen"]
            kwargs = dict(implementation=implementation, repeats=repeats, resume=True, time_budget=remaining)
            if task == "inverse":
                path = benchmark_inverse(sizes, methods, refinement_steps=refinement_steps, **kwargs)
            else:
                path = benchmark_lu(sizes, methods, **kwargs)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            valid = {(int(row["size"]), row["method"]) for row in rows if row["matches_reference"] == "1"}
            required = {(n, method) for n in sizes for method in methods}
            if required <= valid:
                completed_tasks.append(task)
        from lab2.plot_inverse import plot_results

        for task in tasks:
            path = results_dir / f"{task}_{implementation}.csv"
            if path.exists():
                with path.open(newline="", encoding="utf-8") as file:
                    has_rows = any(row["matches_reference"] == "1" for row in csv.DictReader(file))
                if has_rows:
                    plot_results(implementation=implementation, task=task)
        estimate(implementation=implementation, min_size=min_fit_size, tasks=tasks)
        metadata["completed_tasks"] = completed_tasks
        metadata["complete"] = len(completed_tasks) == len(tasks)
    except Exception as error:
        metadata["error"] = str(error)
        raise
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Run complete={metadata['complete']}; results: {results_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 17, 27, 32, 64, 96, 128, 192, 256, 384, 512])
    parser.add_argument("--tasks", choices=["inverse", "lu"], nargs="+", default=["inverse", "lu"])
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=3600)
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument("--min-fit-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.time_budget <= 0 or args.refinement_steps < 0 or args.min_fit_size < 1:
        parser.error("Invalid benchmark parameters.")
    run(args.sizes, args.implementation, args.repeats, args.time_budget,
        args.refinement_steps, args.min_fit_size, tuple(dict.fromkeys(args.tasks)))
