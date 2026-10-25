from itertools import product
import unittest

import numpy as np

from lab2.inverse import inverse
from lab2.lu import lu

try:
    from lab2.inverse_compiled import inverse as compiled_inverse
    from lab2.lu_compiled import lu as compiled_lu
except ModuleNotFoundError:
    compiled_inverse = None
    compiled_lu = None


class ExhaustiveSmallTests(unittest.TestCase):
    def test_all_invertible_ternary_2x2(self):
        algorithms = [inverse] if compiled_inverse is None else [inverse, compiled_inverse]
        checked = 0
        for entries in product((-1., 0., 1.), repeat=4):
            A = np.array(entries).reshape(2, 2)
            if abs(np.linalg.det(A)) < 0.5:
                continue
            checked += 1
            for algorithm in algorithms:
                for method in ("binet", "strassen", "laderman"):
                    result, _ = algorithm(A, method)
                    np.testing.assert_allclose(A @ result, np.eye(2), atol=1e-12)
                    np.testing.assert_allclose(result @ A, np.eye(2), atol=1e-12)
        self.assertEqual(checked, 48)

    def test_all_invertible_binary_3x3(self):
        algorithms = [inverse] if compiled_inverse is None else [inverse, compiled_inverse]
        checked = 0
        for entries in product((0., 1.), repeat=9):
            A = np.array(entries).reshape(3, 3)
            if abs(np.linalg.det(A)) < 0.5:
                continue
            checked += 1
            for algorithm in algorithms:
                for method in ("binet", "strassen", "laderman"):
                    result, _ = algorithm(A, method)
                    np.testing.assert_allclose(A @ result, np.eye(3), atol=1e-12)
                    np.testing.assert_allclose(result @ A, np.eye(3), atol=1e-12)
        self.assertEqual(checked, 174)

    def test_binary_3x3_lu_with_nonzero_leading_minors(self):
        algorithms = [lu] if compiled_lu is None else [lu, compiled_lu]
        checked = 0
        for entries in product((0., 1.), repeat=9):
            A = np.array(entries).reshape(3, 3)
            if any(abs(np.linalg.det(A[:k, :k])) < 0.5 for k in (1, 2, 3)):
                continue
            checked += 1
            for algorithm in algorithms:
                for method in ("binet", "strassen"):
                    L, U, _ = algorithm(A, method)
                    np.testing.assert_allclose(L @ U, A, atol=1e-12)
                    np.testing.assert_array_equal(np.triu(L, 1), np.zeros((3, 3)))
                    np.testing.assert_array_equal(np.tril(U, -1), np.zeros((3, 3)))
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
