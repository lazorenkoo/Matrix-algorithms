import unittest

import numpy as np

from lab3.hmatrix import compress, reconstruct


class CompressionTests(unittest.TestCase):
    def test_zero_leaf(self):
        matrix = np.zeros((3, 5))
        root = compress(matrix, 1, 0)
        self.assertFalse(root.children)
        self.assertIsNone(root.left)
        self.assertIsNone(root.dense)
        np.testing.assert_array_equal(reconstruct(root), matrix)

    def test_rank_one(self):
        matrix = np.outer(np.arange(1., 42), np.arange(1., 46))
        root = compress(matrix, 1, 1e-6)
        self.assertEqual(root.left.shape, (41, 1))
        np.testing.assert_allclose(reconstruct(root), matrix)

    def test_strict_threshold(self):
        matrix = np.diag([10., 2., 1.])
        self.assertEqual(len(compress(matrix, 1, 2).children), 4)
        root = compress(matrix, 1, 3)
        np.testing.assert_allclose(reconstruct(root), np.diag([10., 0., 0.]))

    def test_odd_rectangular_split(self):
        matrix = np.arange(35.).reshape(5, 7)
        root = compress(matrix, 1, 0)
        self.assertEqual([node.shape for node in root.children],
                         [(2, 3), (2, 4), (3, 3), (3, 4)])
        np.testing.assert_array_equal(reconstruct(root), matrix)

    def test_small_dense_leaf(self):
        matrix = np.array([[9., 2., 7.]])
        root = compress(matrix, 4, 0)
        self.assertIsNotNone(root.dense)
        np.testing.assert_array_equal(reconstruct(root), matrix)


if __name__ == "__main__":
    unittest.main()
