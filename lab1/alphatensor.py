from functools import lru_cache
from pathlib import Path

import numpy as np

if __name__ == "alphatensor_compiled":
    from strassen_compiled import multiply_direct
else:
    from strassen import multiply_direct


@lru_cache(maxsize=1)
def coefficients():
    with np.load(Path(__file__).parent / "data" / "alphatensor_455.npz") as archive:
        U, V, W = archive["U"], archive["V"], archive["W"]

    def terms(vector):
        return [(int(i), int(vector[i])) for i in np.flatnonzero(vector)]

    return (
        [terms(U[:, k]) for k in range(U.shape[1])],
        [terms(V[:, k]) for k in range(V.shape[1])],
        [terms(W[k, :]) for k in range(W.shape[0])],
    )


def combine(blocks, terms):
    index, sign = terms[0]
    result = blocks[index] if sign == 1 else -blocks[index]
    operations = 0
    for index, sign in terms[1:]:
        result = result + blocks[index] if sign == 1 else result - blocks[index]
        operations += result.size
    return result, operations


def alphatensor(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows, common = A.shape
    cols = B.shape[1]
    if rows < 4 or common < 5 or cols < 5:
        return multiply_direct(A, B)

    padded_rows = ((rows + 3) // 4) * 4
    padded_common = ((common + 4) // 5) * 5
    padded_cols = ((cols + 4) // 5) * 5

    if A.shape != (padded_rows, padded_common):
        padded_A = np.zeros((padded_rows, padded_common))
        padded_A[:rows, :common] = A
        A = padded_A
    if B.shape != (padded_common, padded_cols):
        padded_B = np.zeros((padded_common, padded_cols))
        padded_B[:common, :cols] = B
        B = padded_B

    block_rows = padded_rows // 4
    block_common = padded_common // 5
    block_cols = padded_cols // 5
    A_blocks = [
        A[i * block_rows:(i + 1) * block_rows,
          j * block_common:(j + 1) * block_common]
        for i in range(4) for j in range(5)
    ]
    B_blocks = [
        B[i * block_common:(i + 1) * block_common,
          j * block_cols:(j + 1) * block_cols]
        for i in range(5) for j in range(5)
    ]
    U, V, W = coefficients()
    products = []
    multiplications = 0
    additions = 0

    for left_terms, right_terms in zip(U, V):
        left, left_ops = combine(A_blocks, left_terms)
        right, right_ops = combine(B_blocks, right_terms)
        product, mul, add = alphatensor(left, right)
        products.append(product)
        multiplications += mul
        additions += left_ops + right_ops + add

    C = np.zeros((padded_rows, padded_cols))
    for index, output_terms in enumerate(W):
        block, operations = combine(products, output_terms)
        i, j = divmod(index, 5)
        C[i * block_rows:(i + 1) * block_rows,
          j * block_cols:(j + 1) * block_cols] = block
        additions += operations

    return C[:rows, :cols], multiplications, additions
