import json
import platform
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import Cython
import numpy as np

import alphatensor_compiled
import binet_compiled
import strassen_compiled
from benchmark_strassen import benchmark_strassen
from memory_benchmark import benchmark_memory


if __name__ == "__main__":
    root = Path(__file__).parent
    sizes = [1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128, 256, 512]
    memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 200, 256]
    methods = [
        (binet_compiled, binet_compiled.binet),
        (strassen_compiled, strassen_compiled.strassen),
        (alphatensor_compiled, alphatensor_compiled.alphatensor),
    ]
    backup = Path(tempfile.mkdtemp(prefix="matrix_compiled_previous_"))
    for module, _ in methods:
        for suffix in ["results", "memory_results"]:
            path = root / f"{module.__name__}_{suffix}.csv"
            if path.exists():
                shutil.copy2(path, backup / path.name)
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cython": Cython.__version__,
        "platform": platform.platform(),
        "timing_sizes": sizes,
        "repeats_up_to_128": 2,
        "repeats_above_128": 1,
        "memory_sizes": memory_sizes,
        "saved_inputs": True,
        "modules": {module.__name__: module.__file__ for module, _ in methods},
    }
    metadata_path = root / "compiled_benchmark_environment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    alphatensor_compiled.coefficients()
    print(f"Previous results backed up to {backup}", flush=True)
    for module, algorithm in methods:
        print(f"TIMING {module.__name__}: {module.__file__}", flush=True)
        benchmark_strassen(
            sizes, root / f"{module.__name__}_results.csv",
            time_budget=float("inf"), algorithm=algorithm,
            algorithm_name=module.__name__, small_repeats=2,
        )
    for module, algorithm in methods:
        print(f"MEMORY {module.__name__}", flush=True)
        benchmark_memory(
            memory_sizes, root / f"{module.__name__}_memory_results.csv",
            time_budget=float("inf"), algorithm=algorithm, saved_inputs=True,
        )
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("All compiled benchmarks completed.", flush=True)
