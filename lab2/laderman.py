import numpy as np

if __name__ == "lab2.laderman_compiled":
    from lab1.strassen_compiled import multiply_direct
else:
    from lab1.strassen import multiply_direct


LEFT_TERMS = (
    ((4, 1), (5, -1)),
    ((4, 1), (7, 1)),
    ((8, 1),),
    ((5, -1),),
    ((0, -1), (2, 1), (4, -1), (5, 1), (6, -1), (7, -1), (8, 1)),
    ((7, 1),),
    ((4, -1), (5, 1), (7, -1)),
    ((0, 1), (2, -1)),
    ((0, 1), (6, 1)),
    ((2, 1),),
    ((6, -1),),
    ((0, -1), (2, 1), (6, -1)),
    ((3, 1), (5, -1)),
    ((3, 1), (6, 1)),
    ((1, -1), (2, 1), (3, -1), (5, 1), (6, -1), (7, -1), (8, 1)),
    ((3, -1), (5, 1), (6, -1)),
    ((1, -1), (2, 1), (7, -1)),
    ((1, 1), (2, -1)),
    ((1, 1), (7, 1)),
    ((4, 1),),
    ((3, 1),),
    ((0, 1),),
    ((1, 1),),
)

RIGHT_TERMS = (
    ((4, 1), (5, -1)),
    ((4, 1), (7, 1)),
    ((8, 1),),
    ((0, 1), (2, -1), (4, 1), (5, -1), (6, 1), (7, 1), (8, -1)),
    ((7, 1),),
    ((5, 1),),
    ((4, 1), (5, -1), (7, 1)),
    ((1, 1), (2, -1)),
    ((1, 1), (7, 1)),
    ((1, -1), (2, 1), (3, -1), (5, 1), (6, -1), (7, -1), (8, 1)),
    ((2, 1),),
    ((1, 1), (2, -1), (7, 1)),
    ((0, 1), (2, -1)),
    ((0, 1), (6, 1)),
    ((6, 1),),
    ((0, 1), (2, -1), (6, 1)),
    ((3, 1), (5, -1), (6, 1)),
    ((3, 1), (5, -1)),
    ((3, 1), (6, 1)),
    ((3, 1),),
    ((1, 1),),
    ((0, 1),),
    ((4, 1),),
)

OUTPUT_TERMS = (
    ((5, -1), (16, 1), (17, 1), (18, 1), (21, 1)),
    ((7, 1), (8, 1), (10, 1), (11, 1), (22, 1)),
    ((5, -1), (8, 1), (9, 1), (10, 1), (11, 1), (16, 1), (18, 1)),
    ((10, 1), (12, 1), (13, 1), (15, 1), (19, 1)),
    ((0, 1), (1, 1), (5, -1), (6, 1), (20, 1)),
    ((1, 1), (3, 1), (5, -1), (6, 1), (10, 1), (13, 1), (15, 1)),
    ((5, 1), (10, -1), (12, -1), (14, 1), (15, -1), (16, -1), (17, -1)),
    ((0, -1), (4, 1), (5, 1), (6, -1), (7, -1), (10, -1), (11, -1)),
    ((2, 1), (5, 1), (10, -1)),
)


def _combine(blocks, terms):
    index, sign = terms[0]
    result = blocks[index] if sign == 1 else -blocks[index]
    additions = 0
    for index, sign in terms[1:]:
        result = result + blocks[index] if sign == 1 else result - blocks[index]
        additions += result.size
    return result, additions


def laderman(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows, common = A.shape
    cols = B.shape[1]
    if min(rows, common, cols) < 3:
        return multiply_direct(A, B)

    padded_rows = ((rows + 2) // 3) * 3
    padded_common = ((common + 2) // 3) * 3
    padded_cols = ((cols + 2) // 3) * 3
    if A.shape != (padded_rows, padded_common):
        padded_A = np.zeros((padded_rows, padded_common))
        padded_A[:rows, :common] = A
        A = padded_A
    if B.shape != (padded_common, padded_cols):
        padded_B = np.zeros((padded_common, padded_cols))
        padded_B[:common, :cols] = B
        B = padded_B

    block_rows = padded_rows // 3
    block_common = padded_common // 3
    block_cols = padded_cols // 3
    A_blocks = [
        A[i * block_rows:(i + 1) * block_rows, j * block_common:(j + 1) * block_common]
        for i in range(3) for j in range(3)
    ]
    B_blocks = [
        B[i * block_common:(i + 1) * block_common, j * block_cols:(j + 1) * block_cols]
        for i in range(3) for j in range(3)
    ]
    products = []
    multiplications = 0
    additions = 0
    for left_terms, right_terms in zip(LEFT_TERMS, RIGHT_TERMS):
        left, left_add = _combine(A_blocks, left_terms)
        right, right_add = _combine(B_blocks, right_terms)
        product, mul, add = laderman(left, right)
        products.append(product)
        multiplications += mul
        additions += left_add + right_add + add

    C = np.zeros((padded_rows, padded_cols))
    for index, terms in enumerate(OUTPUT_TERMS):
        block, add = _combine(products, terms)
        i, j = divmod(index, 3)
        C[i * block_rows:(i + 1) * block_rows, j * block_cols:(j + 1) * block_cols] = block
        additions += add
    return C[:rows, :cols], multiplications, additions
