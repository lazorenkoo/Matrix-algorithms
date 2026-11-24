from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, svds


@dataclass(frozen=True)
class HMatrixNode:
    bounds: tuple[int, int, int, int]
    children: tuple["HMatrixNode", ...] = ()
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    dense: np.ndarray | None = None
    singular_values: np.ndarray | None = None

    @property
    def shape(self):
        r0, r1, c0, c1 = self.bounds
        return r1 - r0, c1 - c0


def compress(matrix, max_rank, threshold):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Expected a nonempty finite matrix.")
    if not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
        raise ValueError("Expected a positive integer rank.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Expected a nonnegative finite threshold.")
    return _build(matrix, (0, matrix.shape[0], 0, matrix.shape[1]), max_rank, threshold)


def _build(matrix, bounds, rank, threshold):
    r0, r1, c0, c1 = bounds
    block = matrix[r0:r1, c0:c1]
    if not np.any(block):
        return HMatrixNode(bounds)

    if min(block.shape) <= rank:
        return HMatrixNode(bounds, dense=block.copy())

    u, sigma, vt = leading_svd(block, rank + 1)
    if sigma[rank] < threshold:
        return HMatrixNode(
            bounds,
            left=u[:, :rank].copy(),
            right=sigma[:rank, None] * vt[:rank, :],
        )

    row_mid = (r0 + r1) // 2
    col_mid = (c0 + c1) // 2
    children = tuple(
        _build(matrix, (lo_r, hi_r, lo_c, hi_c), rank, threshold)
        for lo_r, hi_r in ((r0, row_mid), (row_mid, r1))
        for lo_c, hi_c in ((c0, col_mid), (col_mid, c1))
    )
    return HMatrixNode(bounds, children=children)


def leading_svd(block, count):
    if count < min(block.shape) and max(block.shape) > 32:
        try:
            u, sigma, vt = svds(block, k=count, rng=0)
            order = np.argsort(sigma)[::-1]
            return u[:, order], sigma[order], vt[order, :]
        except ArpackNoConvergence:
            pass
    u, sigma, vt = np.linalg.svd(block, full_matrices=False)
    return u[:, :count], sigma[:count], vt[:count, :]


def reconstruct(node):
    if node.children:
        top_left, top_right, bottom_left, bottom_right = map(reconstruct, node.children)
        return np.block([[top_left, top_right], [bottom_left, bottom_right]])
    if node.dense is not None:
        return node.dense.copy()
    if node.left is not None:
        return node.left @ node.right
    return np.zeros(node.shape)
