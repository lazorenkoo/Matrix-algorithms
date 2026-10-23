import hashlib
import json
from pathlib import Path

import numpy as np


def case_directory(n: int) -> Path:
    return Path(__file__).parent / "comparison" / f"size_{n}"


def load_matrix_pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    folder = case_directory(n)
    folder.mkdir(parents=True, exist_ok=True)
    a_path = folder / "A.csv"
    b_path = folder / "B.csv"

    if a_path.exists() and b_path.exists():
        A = np.loadtxt(a_path, delimiter=",", ndmin=2)
        B = np.loadtxt(b_path, delimiter=",", ndmin=2)
    else:
        rng = np.random.Generator(np.random.PCG64(42))
        low = np.nextafter(1e-8, 1.0)
        high = np.nextafter(1.0, 0.0)
        A = rng.uniform(low, high, size=(n, n))
        B = rng.uniform(low, high, size=(n, n))
        np.savetxt(a_path, A, delimiter=",", fmt="%.17g")
        np.savetxt(b_path, B, delimiter=",", fmt="%.17g")

    metadata = {
        "size": n,
        "dtype": "float64",
        "seed": 42,
        "generator": "PCG64",
        "generation_order": "A, then B; generator restarted for each size",
        "interval": "(1e-8, 1.0)",
        "A_sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
        "B_sha256": hashlib.sha256(b_path.read_bytes()).hexdigest(),
        "reuse": "Binet and Strassen must load the same saved A.csv and B.csv",
    }
    (folder / "matrices.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return A, B
