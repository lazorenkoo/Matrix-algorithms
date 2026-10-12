import unittest

import numpy as np

from alphatensor import alphatensor


class AlphaTensorTests(unittest.TestCase):
    def test_basis_products(self):
        for left in range(20):
            for right in range(25):
                A = np.zeros((4, 5))
                B = np.zeros((5, 5))
                A.flat[left] = 1
                B.flat[right] = 1
                C, mul, add = alphatensor(A, B)
                np.testing.assert_array_equal(C, A @ B)
                self.assertEqual((mul, add), (76, 549))

    def test_recursive_kernel(self):
        rng = np.random.default_rng(42)
        A = rng.uniform(-1, 1, size=(16, 25))
        B = rng.uniform(-1, 1, size=(25, 25))
        C, mul, _ = alphatensor(A, B)
        np.testing.assert_allclose(C, A @ B, rtol=1e-12, atol=1e-12)
        self.assertEqual(mul, 76 ** 2)

    def test_arbitrary_shapes(self):
        rng = np.random.default_rng(12)
        shapes = [(n, n, n) for n in range(1, 18)]
        shapes += [(1, 7, 4), (4, 1, 7), (7, 9, 1), (7, 11, 9), (13, 17, 6)]
        for rows, common, cols in shapes:
            with self.subTest(shape=(rows, common, cols)):
                A = rng.uniform(-1, 1, size=(rows, common))
                B = rng.uniform(-1, 1, size=(common, cols))
                C, mul, add = alphatensor(A, B)
                self.assertEqual(C.shape, (rows, cols))
                np.testing.assert_allclose(C, A @ B, rtol=1e-12, atol=1e-12)
                self.assertGreater(mul, 0)
                self.assertGreaterEqual(add, 0)


if __name__ == "__main__":
    unittest.main()
