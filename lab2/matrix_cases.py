import hashlib
import json
from pathlib import Path

import numpy as np


def load_matrix(size: int, directory: Path | None = None) -> tuple[np.ndarray, str]:
    directory = Path(directory) if directory is not None else Path(__file__).parent / "data"
    folder = directory / f"size_{size}"
    folder.mkdir(parents=True, exist_ok=True)
    matrix_path = folder / "A.csv"
    metadata_path = folder / "matrix.json"
    if not matrix_path.exists():
        rng = np.random.Generator(np.random.PCG64(42))
        A = rng.uniform(np.nextafter(1e-8, 1.0), np.nextafter(1.0, 0.0), (size, size))
        np.savetxt(matrix_path, A, delimiter=",", fmt="%.17g")
        checksum = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps({
            "size": size,
            "dtype": "float64",
            "generator": "PCG64",
            "seed": 42,
            "interval": "(1e-8, 1)",
            "A_sha256": checksum,
        }, indent=2) + "\n", encoding="utf-8")
    A = np.loadtxt(matrix_path, delimiter=",", ndmin=2)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if A.shape != (size, size) or not np.isfinite(A).all() or not ((A > 1e-8) & (A < 1)).all():
        raise ValueError(f"Invalid saved input: {matrix_path}")
    if metadata["size"] != size or metadata["A_sha256"] != checksum:
        raise ValueError(f"Saved input does not match its metadata: {matrix_path}")
    return A, checksum
