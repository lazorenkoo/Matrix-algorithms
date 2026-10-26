import unittest

import numpy as np

from lab1.strassen import strassen
from lab2.inverse import inverse
from lab2.multiplication import strassen_rectangular


class InverseTests(unittest.TestCase):
    def check_inverse(self, A):
        original = A.copy()
        for method in ("binet", "strassen", "laderman"):
            with self.subTest(size=A.shape, method=method):
                result, operations = inverse(A, method)
                np.testing.assert_allclose(result, np.linalg.inv(A), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(A @ result, np.eye(len(A)), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(result @ A, np.eye(len(A)), rtol=1e-9, atol=1e-9)
                np.testing.assert_array_equal(A, original)
                self.assertGreater(operations.divisions, 0)

    def test_scalar(self):
        for method in ("binet", "strassen", "laderman"):
            result, operations = inverse([[0.5]], method)
            np.testing.assert_array_equal(result, [[2.0]])
            self.assertEqual((operations.multiplications, operations.additions_subtractions,
                              operations.divisions, operations.total), (0, 0, 1, 1))

    def test_hand_calculation(self):
        for method in ("binet", "strassen", "laderman"):
            result, operations = inverse([[4, 1], [2, 3]], method)
            np.testing.assert_allclose(result, [[0.3, -0.1], [-0.2, 0.4]])
            self.assertEqual((operations.multiplications, operations.additions_subtractions,
                              operations.divisions, operations.total), (6, 2, 2, 10))

    def test_random_positive_matrices(self):
        rng = np.random.default_rng(42)
        for size in range(1, 20):
            self.check_inverse(rng.uniform(1e-8, 1, size=(size, size)))

    def test_refinement_counts(self):
        for method, expected in (("binet", (22, 18, 2, 42)), ("strassen", (20, 46, 2, 68)), ("laderman", (22, 18, 2, 42))):
            result, operations = inverse([[4, 1], [2, 3]], method, refinement_steps=1)
            np.testing.assert_allclose(result, [[0.3, -0.1], [-0.2, 0.4]])
            self.assertEqual((operations.multiplications, operations.additions_subtractions,
                              operations.divisions, operations.total), expected)

    def test_well_conditioned_matrices(self):
        rng = np.random.default_rng(14)
        for size in (3, 5, 8, 11, 16, 17):
            A = rng.uniform(0.01, 0.02, size=(size, size))
            np.fill_diagonal(A, rng.uniform(0.7, 0.9, size=size))
            self.check_inverse(A)

    def test_row_permutations(self):
        for size in (2, 3, 4, 5, 7):
            self.check_inverse(np.roll(np.eye(size), 1, axis=0))
        self.check_inverse(np.array([
            [1., 2., 1., 0.],
            [2., 4., 0., 1.],
            [1., 0., 3., 0.],
            [0., 1., 0., 3.],
        ]))

    def test_singular_and_invalid_inputs(self):
        for method in ("binet", "strassen", "laderman"):
            for A in ([[0]], [[1, 2], [2, 4]], np.zeros((3, 3))):
                with self.assertRaises(np.linalg.LinAlgError):
                    inverse(A, method)
            for A in ([], [1, 2], [[1, 2]], [[np.nan]], [[np.inf]]):
                with self.assertRaises(ValueError):
                    inverse(A, method)

    def test_rectangular_strassen(self):
        rng = np.random.default_rng(7)
        for rows, common, cols in ((1, 5, 3), (4, 1, 7), (3, 5, 1), (2, 3, 4), (5, 4, 7)):
            A = rng.uniform(-1, 1, size=(rows, common))
            B = rng.uniform(-1, 1, size=(common, cols))
            result, mul, add = strassen_rectangular(A, B)
            np.testing.assert_allclose(result, A @ B, rtol=1e-12, atol=1e-12)
            if min(rows, common, cols) == 1:
                self.assertEqual((mul, add), (rows * common * cols, rows * cols * (common - 1)))
            else:
                size = max(rows, common, cols)
                _, expected_mul, expected_add = strassen(np.zeros((size, size)), np.zeros((size, size)))
                self.assertEqual((mul, add), (expected_mul, expected_add))

    def test_thin_tiled_strassen(self):
        rng = np.random.default_rng(18)
        for rows, common, cols in ((2, 2, 32), (32, 2, 2), (2, 32, 2),
                                   (3, 3, 17), (3, 17, 5), (17, 5, 3), (5, 3, 7)):
            A = rng.uniform(-1, 1, (rows, common))
            B = rng.uniform(-1, 1, (common, cols))
            C, mul, add = strassen_rectangular(A, B)
            np.testing.assert_allclose(C, A @ B, rtol=1e-12, atol=1e-12)
            block = min(rows, common, cols)
            _, tile_mul, tile_add = strassen(np.zeros((block, block)), np.zeros((block, block)))
            tiles = ((rows + block - 1) // block) * ((common + block - 1) // block) * ((cols + block - 1) // block)
            expected_add = tiles * tile_add + rows * cols * ((common + block - 1) // block - 1)
            self.assertEqual((mul, add), (tiles * tile_mul, expected_add))


if __name__ == "__main__":
    unittest.main()
