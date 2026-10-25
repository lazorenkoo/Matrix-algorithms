import importlib.machinery
import unittest

import numpy as np

from lab2.laderman import laderman

try:
    import lab2.laderman_compiled as compiled
except ModuleNotFoundError:
    compiled = None


class LadermanTests(unittest.TestCase):
    def algorithms(self):
        return [laderman] if compiled is None else [laderman, compiled.laderman]

    def test_basis_products(self):
        for algorithm in self.algorithms():
            for left in range(9):
                for right in range(9):
                    A = np.zeros((3, 3))
                    B = np.zeros((3, 3))
                    A.flat[left] = 1
                    B.flat[right] = 1
                    C, mul, add = algorithm(A, B)
                    np.testing.assert_array_equal(C, A @ B)
                    self.assertEqual((mul, add), (23, 98))

    def test_recursive_counts(self):
        rng = np.random.default_rng(13)
        for level in (1, 2, 3):
            size = 3 ** level
            A = rng.uniform(-1, 1, (size, size))
            B = rng.uniform(-1, 1, (size, size))
            for algorithm in self.algorithms():
                C, mul, add = algorithm(A, B)
                np.testing.assert_allclose(C, A @ B, rtol=1e-10, atol=1e-10)
                self.assertEqual(mul, 23 ** level)
                self.assertEqual(add, 7 * (23 ** level - 9 ** level))

    def test_arbitrary_dimensions(self):
        rng = np.random.default_rng(62)
        shapes = [(n, n, n) for n in range(1, 18)]
        shapes += [(1, 7, 5), (5, 1, 7), (7, 5, 1), (2, 7, 4), (7, 2, 5),
                   (5, 7, 2), (3, 5, 7), (7, 11, 9), (13, 17, 6), (9, 27, 9)]
        for rows, common, cols in shapes:
            A = rng.uniform(-1, 1, (rows, common))
            B = rng.uniform(-1, 1, (common, cols))
            expected_A, expected_B = A.copy(), B.copy()
            outputs = []
            for algorithm in self.algorithms():
                C, mul, add = algorithm(A, B)
                np.testing.assert_allclose(C, A @ B, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(A, expected_A)
                np.testing.assert_array_equal(B, expected_B)
                if min(rows, common, cols) < 3:
                    self.assertEqual((mul, add), (rows * common * cols, rows * cols * (common - 1)))
                outputs.append((C, mul, add))
            if len(outputs) == 2:
                np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
                self.assertEqual(outputs[0][1:], outputs[1][1:])

    def test_padding_counts(self):
        for algorithm in self.algorithms():
            _, mul, add = algorithm(np.ones((4, 4)), np.ones((4, 4)))
            self.assertEqual(mul, 23 * 8)
            self.assertEqual(add, 98 * 4 + 23 * 4)

    @unittest.skipIf(compiled is None, "Build Cython extensions first")
    def test_native_direct_multiplier(self):
        import lab1.strassen_compiled as source

        self.assertIsInstance(compiled.__loader__, importlib.machinery.ExtensionFileLoader)
        self.assertIs(compiled.multiply_direct, source.multiply_direct)


if __name__ == "__main__":
    unittest.main()
