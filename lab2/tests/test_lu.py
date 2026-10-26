import unittest

import numpy as np

from lab2.lu import lu


class LUTests(unittest.TestCase):
    def check_factorization(self, A):
        original = A.copy()
        factors = []
        for method in ("binet", "strassen"):
            with self.subTest(size=A.shape, method=method):
                L, U, operations = lu(A, method)
                np.testing.assert_allclose(L @ U, A, rtol=1e-9, atol=1e-9)
                np.testing.assert_array_equal(np.triu(L, 1), np.zeros_like(L))
                np.testing.assert_array_equal(np.tril(U, -1), np.zeros_like(U))
                np.testing.assert_array_equal(np.diag(L), np.ones(len(A)))
                np.testing.assert_array_equal(A, original)
                self.assertGreaterEqual(operations.total, 0)
                factors.append((L, U))
        np.testing.assert_allclose(factors[0][0], factors[1][0], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(factors[0][1], factors[1][1], rtol=1e-8, atol=1e-8)

    def test_scalar(self):
        for value in (0, 0.5, -3):
            for method in ("binet", "strassen"):
                L, U, operations = lu([[value]], method)
                np.testing.assert_array_equal(L, [[1.]])
                np.testing.assert_array_equal(U, [[value]])
                self.assertEqual(operations.total, 0)

    def test_hand_calculation_and_counts(self):
        for method in ("binet", "strassen"):
            L, U, operations = lu([[4, 1], [2, 3]], method)
            np.testing.assert_array_equal(L, [[1, 0], [0.5, 1]])
            np.testing.assert_array_equal(U, [[4, 1], [0, 2.5]])
            self.assertEqual((operations.multiplications, operations.additions_subtractions,
                              operations.divisions, operations.total), (1, 1, 1, 3))

    def test_known_factors(self):
        rng = np.random.default_rng(81)
        for size in (3, 5, 8, 11, 17):
            expected_L = np.tril(rng.uniform(-0.1, 0.1, (size, size)), -1) + np.eye(size)
            expected_U = np.triu(rng.uniform(-0.1, 0.1, (size, size)))
            np.fill_diagonal(expected_U, rng.uniform(0.7, 1.0, size))
            A = expected_L @ expected_U
            for method in ("binet", "strassen"):
                L, U, _ = lu(A, method)
                np.testing.assert_allclose(L, expected_L, rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(U, expected_U, rtol=1e-11, atol=1e-11)

    def test_random_positive_sizes(self):
        rng = np.random.default_rng(42)
        for size in list(range(1, 20)) + [32]:
            self.check_factorization(rng.uniform(1e-8, 1, (size, size)))

    def test_triangular_inputs(self):
        rng = np.random.default_rng(31)
        for size in (3, 5, 8):
            A = rng.uniform(0.1, 1.0, (size, size))
            self.check_factorization(np.tril(A))
            self.check_factorization(np.triu(A))

    def test_singular_with_valid_final_pivot_position(self):
        self.check_factorization(np.array([[1., 2.], [2., 4.]]))

    def test_zero_leading_pivot(self):
        for method in ("binet", "strassen"):
            for A in ([[0, 1], [1, 0]], [[1, 1, 0], [1, 1, 1], [0, 1, 1]]):
                with self.assertRaisesRegex(np.linalg.LinAlgError, "without row permutations"):
                    lu(A, method)

    def test_invalid_inputs(self):
        for A in ([], [1, 2], [[1, 2]], [[np.nan]], [[np.inf]]):
            with self.assertRaises(ValueError):
                lu(A)


if __name__ == "__main__":
    unittest.main()
