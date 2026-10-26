import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lab2.benchmark_lu import benchmark
from lab2.estimate_complexity import estimate
from lab2.operations import Operations
from lab2.run_benchmarks import run


class MeasurementPipelineTests(unittest.TestCase):
    def test_lu_measurements_and_resume(self):
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            kwargs = dict(implementation="python", repeats=1, output_dir=root / "results", data_dir=root / "data")
            path = benchmark([1, 2, 3, 5], ["binet", "strassen"], **kwargs)
            before = path.read_bytes()
            benchmark([1, 2, 3, 5], ["binet", "strassen"], resume=True, **kwargs)
            self.assertEqual(before, path.read_bytes())
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 8)
            for first, second in zip(rows[::2], rows[1::2]):
                self.assertEqual(first["A_sha256"], second["A_sha256"])
                self.assertEqual(first["matches_reference"], "1")
                self.assertEqual(second["matches_reference"], "1")
            for row in rows:
                self.assertEqual(int(row["flops"]), sum(int(row[key]) for key in
                                 ("multiplications", "additions_subtractions", "divisions")))
            version_path = path.with_suffix(".implementation.json")
            version_path.write_text(json.dumps({"algorithm": "legacy_explicit_inverse"}))
            with self.assertRaisesRegex(ValueError, "different algorithm"):
                benchmark([1, 2, 3, 5], ["binet", "strassen"], resume=True, **kwargs)
            self.assertEqual(before, path.read_bytes())

    def test_lu_rejects_wrong_factorization(self):
        def wrong(A, method):
            return np.eye(len(A)), np.zeros_like(A), Operations()

        module = SimpleNamespace(__file__="stub", lu=wrong)
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            with patch("lab2.benchmark_lu.importlib.import_module", return_value=module):
                with self.assertRaises(ArithmeticError):
                    benchmark([2], ["binet"], implementation="python", repeats=1,
                              output_dir=root / "results", data_dir=root / "data")
            with (root / "results" / "lu_python.csv").open(newline="") as file:
                self.assertEqual(next(csv.DictReader(file))["matches_reference"], "0")
            metadata_path = next((root / "results").glob("lu_python_environment_*.json"))
            metadata = json.loads(metadata_path.read_text())
            self.assertFalse(metadata["complete"])
            self.assertIn("error", metadata)

    def test_known_experimental_exponent(self):
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            fields = ["size", "method", "refinement_steps", "time_seconds", "flops", "matches_reference"]
            with (root / "inverse_compiled.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for size in (8, 16, 32, 64):
                    writer.writerow(dict(size=size, method="binet", refinement_steps=1,
                                         time_seconds=0.01 * size ** 3, flops=2 * size ** 3, matches_reference=1))
                writer.writerow(dict(size=128, method="binet", refinement_steps=1,
                                     time_seconds=1e9, flops=1e9, matches_reference=0))
            output = estimate(min_size=8, results_dir=root, tasks=("inverse",))
            with output.open(newline="") as file:
                fits = list(csv.DictReader(file))
            self.assertEqual(len(fits), 2)
            for fit in fits:
                self.assertEqual(fit["points"], "4")
                self.assertAlmostEqual(float(fit["exponent_p"]), 3)
                self.assertAlmostEqual(float(fit["r_squared_log"]), 1)
            estimate(min_size=32, results_dir=root, tasks=("inverse",))
            with output.open(newline="") as file:
                self.assertTrue(all(row["status"] == "insufficient_points" for row in csv.DictReader(file)))

    def test_combined_run_and_generated_files(self):
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            kwargs = dict(implementation="python", repeats=1, min_fit_size=1,
                          output_dir=root / "results", data_dir=root / "data")
            metadata_path = run([1, 2, 3, 5], **kwargs)
            self.assertTrue(json.loads(metadata_path.read_text())["complete"])
            paths = [root / "results" / f"{task}_python.csv" for task in ("inverse", "lu")]
            snapshots = [path.read_bytes() for path in paths]
            run([1, 2, 3, 5], **kwargs)
            self.assertEqual(snapshots, [path.read_bytes() for path in paths])
            with paths[0].open(newline="") as file:
                inverse_rows = list(csv.DictReader(file))
            with paths[1].open(newline="") as file:
                lu_rows = list(csv.DictReader(file))
            self.assertEqual(len(inverse_rows), 12)
            self.assertEqual(len(lu_rows), 8)
            for size in (1, 2, 3, 5):
                checksums = {r["A_sha256"] for r in inverse_rows + lu_rows if int(r["size"]) == size}
                self.assertEqual(len(checksums), 1)
            for task in ("inverse", "lu"):
                for field in ("time", "flops"):
                    path = root / "plots" / f"{task}_python_{field}.png"
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((root / "results" / "complexity_python.csv").exists())


if __name__ == "__main__":
    unittest.main()
