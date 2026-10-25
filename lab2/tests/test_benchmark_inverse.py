import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lab2.benchmark_inverse import benchmark
from lab2.matrix_cases import load_matrix


class BenchmarkInverseTests(unittest.TestCase):
    def test_saved_input_integrity(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            A, checksum = load_matrix(3, root)
            B, saved_checksum = load_matrix(3, root)
            np.testing.assert_array_equal(A, B)
            self.assertEqual(checksum, saved_checksum)
            path = root / "size_3" / "A.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                load_matrix(3, root)

    def test_shared_inputs_and_resume(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = dict(implementation="python", repeats=1, output_dir=root / "results", data_dir=root / "data")
            path = benchmark([1, 2, 3], ["binet", "strassen"], **kwargs)
            original = path.read_bytes()
            benchmark([1, 2, 3], ["binet", "strassen"], resume=True, **kwargs)
            self.assertEqual(path.read_bytes(), original)
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            for first, second in zip(rows[::2], rows[1::2]):
                self.assertEqual(first["A_sha256"], second["A_sha256"])
                self.assertEqual(first["matches_reference"], "1")
                self.assertEqual(second["matches_reference"], "1")
            for row in rows:
                self.assertEqual(int(row["flops"]), sum(int(row[key]) for key in
                                 ["multiplications", "additions_subtractions", "divisions"]))


if __name__ == "__main__":
    unittest.main()
