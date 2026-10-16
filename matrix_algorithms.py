from functools import lru_cache

import numpy as np

def binet(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:

    rows, common = A.shape
    cols = B.shape[1]

    if rows == 1 or common == 1 or cols == 1:
        C = np.zeros((rows, cols))
        multiplications = 0
        additions = 0

        for i in range(rows):
            for j in range(cols):
                C[i, j] = A[i, 0] * B[0, j]
                multiplications += 1

                for k in range(1, common):
                    C[i, j] += A[i, k] * B[k, j]
                    multiplications += 1
                    additions += 1

        return C, multiplications, additions

    mid_rows = rows // 2
    mid_common = common // 2
    mid_cols = cols // 2

    A11 = A[:mid_rows, :mid_common]
    A12 = A[:mid_rows, mid_common:]
    A21 = A[mid_rows:, :mid_common]
    A22 = A[mid_rows:, mid_common:]

    B11 = B[:mid_common, :mid_cols]
    B12 = B[:mid_common, mid_cols:]
    B21 = B[mid_common:, :mid_cols]
    B22 = B[mid_common:, mid_cols:]

    P1, mul1, add1 = binet(A11, B11)
    P2, mul2, add2 = binet(A12, B21)
    C11 = P1 + P2

    P3, mul3, add3 = binet(A11, B12)
    P4, mul4, add4 = binet(A12, B22)
    C12 = P3 + P4

    P5, mul5, add5 = binet(A21, B11)
    P6, mul6, add6 = binet(A22, B21)
    C21 = P5 + P6

    P7, mul7, add7 = binet(A21, B12)
    P8, mul8, add8 = binet(A22, B22)
    C22 = P7 + P8

    C = np.block([[C11, C12], [C21, C22]])

    multiplications = (
        mul1 + mul2 + mul3 + mul4 + mul5 + mul6 + mul7 + mul8
    )
    additions = (
        add1 + add2 + add3 + add4 + add5 + add6 + add7 + add8
        + C11.size + C12.size + C21.size + C22.size
    )

    return C, multiplications, additions


def multiply_direct(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows, common = A.shape
    cols = B.shape[1]
    C = np.zeros((rows, cols))
    multiplications = 0
    additions = 0

    for i in range(rows):
        for j in range(cols):
            C[i, j] = A[i, 0] * B[0, j]
            multiplications += 1
            for k in range(1, common):
                C[i, j] += A[i, k] * B[k, j]
                multiplications += 1
                additions += 1

    return C, multiplications, additions


def strassen(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    n = A.shape[0]

    if n == 1:
        return np.array([[A[0, 0] * B[0, 0]]]), 1, 0

    if n % 2 == 1:
        return strassen_odd(A, B)

    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    P1, mul1, add1 = strassen(A11 + A22, B11 + B22)
    P2, mul2, add2 = strassen(A21 + A22, B11)
    P3, mul3, add3 = strassen(A11, B12 - B22)
    P4, mul4, add4 = strassen(A22, B21 - B11)
    P5, mul5, add5 = strassen(A11 + A12, B22)
    P6, mul6, add6 = strassen(A21 - A11, B11 + B12)
    P7, mul7, add7 = strassen(A12 - A22, B21 + B22)

    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    C = np.block([[C11, C12], [C21, C22]])

    multiplications = mul1 + mul2 + mul3 + mul4 + mul5 + mul6 + mul7
    additions = add1 + add2 + add3 + add4 + add5 + add6 + add7 + 18 * mid * mid
    return C, multiplications, additions


def strassen_odd(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    A0 = A[:-1, :-1]
    a = A[:-1, -1:]
    r = A[-1:, :-1]
    alpha = A[-1:, -1:]
    B0 = B[:-1, :-1]
    b = B[:-1, -1:]
    s = B[-1:, :-1]
    beta = B[-1:, -1:]

    P1, mul1, add1 = strassen(A0, B0)
    P2, mul2, add2 = multiply_direct(a, s)
    P3, mul3, add3 = multiply_direct(A0, b)
    P4, mul4, add4 = multiply_direct(a, beta)
    P5, mul5, add5 = multiply_direct(r, B0)
    P6, mul6, add6 = multiply_direct(alpha, s)
    P7, mul7, add7 = multiply_direct(r, b)
    P8, mul8, add8 = multiply_direct(alpha, beta)

    C11 = P1 + P2
    C12 = P3 + P4
    C21 = P5 + P6
    C22 = P7 + P8
    C = np.block([[C11, C12], [C21, C22]])

    multiplications = mul1 + mul2 + mul3 + mul4 + mul5 + mul6 + mul7 + mul8
    additions = (
        add1 + add2 + add3 + add4 + add5 + add6 + add7 + add8
        + C11.size + C12.size + C21.size + C22.size
    )
    return C, multiplications, additions


@lru_cache(maxsize=1)
def coefficients():
    return ([[(11, 1)], [(1, 1), (2, -1), (12, -1)], [(14, -1), (16, -1), (19, -1)],
      [(4, -1), (16, -1), (19, -1)], [(0, 1), (5, -1), (6, 1)], [(3, -1), (11, -1), (13, -1)],
      [(9, -1), (16, -1), (19, -1)], [(8, 1), (11, -1), (13, -1)], [(16, 1), (19, 1)],
      [(1, -1), (2, 1), (7, -1)], [(5, -1), (6, 1), (10, 1)], [(5, 1), (6, -1), (15, 1)],
      [(5, -1), (6, 1)], [(1, 1), (2, -1)], [(11, 1), (13, 1)], [(17, -1)], [(18, 1)],
      [(17, -1), (18, -1), (19, 1)], [(8, -1), (12, -1), (13, 1)], [(10, -1)],
      [(10, -1), (12, 1), (13, -1)], [(12, 1)], [(0, -1), (2, 1), (17, 1)], [(4, 1)], [(0, 1)],
      [(5, -1), (9, 1), (10, 1)], [(0, -1), (2, 1), (4, -1)], [(9, 1)], [(8, -1)],
      [(4, -1), (18, 1), (19, -1)], [(5, -1), (8, 1), (9, 1)], [(14, -1)], [(7, 1)],
      [(5, -1), (8, 1), (15, -1)], [(5, 1), (8, -1), (9, 1), (15, 1), (17, -1), (18, -1), (19, 1)],
      [(0, -1), (5, 1), (8, -1), (9, -1)], [(3, -1)], [(2, 1), (4, -1), (7, -1)],
      [(3, -1), (10, -1), (13, -1)], [(0, 1), (2, -1), (4, 1), (12, -1)],
      [(10, 1), (12, -1), (13, 1), (18, 1)], [(14, -1), (17, 1), (19, -1)], [(6, 1)], [(1, 1)],
      [(1, 1), (2, -1), (6, -1), (7, 1), (8, -1), (9, -1), (11, 1), (13, 1), (16, -1), (19, -1)],
      [(16, 1)],
      [(0, -1), (1, 1), (3, 1), (4, 1), (5, 1), (6, -1), (11, 1), (13, 1), (16, 1), (19, 1)],
      [(8, 1), (13, -1)], [(5, 1), (9, -1), (10, -1), (14, 1)], [(2, -1), (4, 1), (7, 1), (9, -1)],
      [(5, -1), (10, 1)], [(0, 1), (2, -1), (3, 1), (4, 1), (10, 1), (12, -1), (13, 1), (14, 1)],
      [(3, 1), (13, 1)], [(7, -1), (8, 1), (12, 1), (13, -1)], [(9, 1), (19, 1)],
      [(12, 1), (13, -1), (14, -1), (17, 1), (18, -1), (19, -1)], [(2, -1), (7, 1)],
      [(5, -1), (8, 1), (15, -1), (18, 1)], [(2, 1), (12, 1)],
      [(0, 1), (2, -1), (3, 1), (10, 1), (12, -1), (13, 1)], [(4, -1), (19, -1)],
      [(2, -1), (3, -1), (4, 1), (17, -1), (18, -1), (19, 1)], [(2, -1), (17, -1)],
      [(0, -1), (2, 1), (15, -1), (17, 1)],
      [(10, -1), (12, 1), (13, -1), (14, -1), (15, -1), (17, 1), (18, -1), (19, -1)], [(15, 1)],
      [(11, -1), (13, -1), (18, -1)],
      [(1, -1), (2, 1), (5, 1), (6, -1), (11, -1), (13, -1), (15, 1), (16, -1), (17, 1), (18, -1)],
      [(0, -1), (5, 1)], [(5, 1), (15, 1)],
      [(1, -1), (2, 1), (5, -1), (6, 1), (10, 1), (11, -1), (12, 1), (14, -1), (16, -1), (19, -1)],
      [(1, -1), (2, 1), (17, 1)], [(7, -1), (8, -1), (9, 1), (17, -1), (18, -1), (19, 1)],
      [(13, -1), (18, -1)], [(10, -1), (12, 1), (13, -1), (15, -1), (17, 1), (18, -1)],
      [(0, -1), (2, -1), (3, 1), (4, 1), (5, 1), (7, 1), (8, -1), (9, -1)]],
     [[(6, 1), (8, 1), (16, -1)], [(8, -1), (11, 1)], [(8, 1), (21, 1)], [(5, -1), (21, -1)],
      [(1, 1), (5, -1)], [(5, -1), (16, -1)], [(9, 1), (21, -1)], [(9, 1), (16, -1)], [(21, -1)],
      [(9, 1), (11, 1)], [(1, 1), (8, -1)], [(1, -1), (7, -1)], [(1, 1)], [(11, 1)], [(16, 1)],
      [(11, -1), (12, 1), (22, 1)], [(16, 1), (17, -1), (22, -1)], [(22, 1)],
      [(11, -1), (13, -1), (14, 1), (18, -1)], [(1, 1), (3, 1), (18, -1)], [(18, 1)],
      [(11, -1), (13, -1), (18, -1)], [(0, 1), (1, 1), (2, -1), (10, 1)],
      [(10, 1), (20, 1), (21, 1)], [(0, 1), (1, 1), (10, 1)], [(4, 1), (21, -1), (23, -1), (24, 1)],
      [(10, 1)], [(4, 1), (21, -1), (24, 1)], [(4, 1), (16, -1), (19, 1)],
      [(15, 1), (16, 1), (17, -1), (22, -1)], [(4, -1)], [(21, 1), (23, 1)], [(11, -1), (14, 1)],
      [(4, -1), (16, 1), (17, -1), (19, -1), (22, -1)], [(11, -1), (12, 1), (14, 1), (22, 1)],
      [(4, 1), (15, -1), (16, -1), (19, 1)], [(15, -1), (16, -1)],
      [(10, -1), (15, 1), (16, 1), (19, -1), (20, -1), (21, -1), (24, 1)],
      [(0, -1), (1, -1), (3, -1), (18, 1), (20, 1), (21, 1), (23, 1)],
      [(10, -1), (20, -1), (21, -1), (23, -1)], [(1, -1), (2, 1), (3, -1), (18, 1)],
      [(1, -1), (2, 1), (3, -1), (11, -1), (12, 1), (13, -1), (22, 1)], [(1, 1), (6, 1), (9, -1)],
      [(5, -1), (6, -1), (11, -1)], [(9, 1)], [(6, 1), (7, -1), (21, -1)], [(5, -1)],
      [(9, -1), (11, -1), (13, -1), (14, 1), (18, -1), (19, 1)], [(21, -1), (23, -1), (24, 1)],
      [(15, 1), (16, 1), (19, -1), (20, -1), (21, -1), (24, 1)],
      [(3, 1), (4, -1), (8, 1), (21, 1), (23, 1), (24, -1)], [(20, 1), (21, 1), (23, 1)],
      [(0, -1), (1, -1), (3, -1), (5, -1), (15, 1), (18, 1)], [(11, 1), (13, 1), (14, -1)],
      [(9, 1), (11, 1), (12, -1), (14, -1), (22, -1), (24, -1)],
      [(1, -1), (2, 1), (3, -1), (11, -1), (12, 1), (13, -1)],
      [(9, -1), (10, 1), (14, -1), (20, 1), (21, 1), (24, -1)],
      [(11, 1), (12, -1), (14, -1), (16, -1), (17, 1), (19, 1)],
      [(0, 1), (1, 1), (3, 1), (8, 1), (10, 1), (13, 1)],
      [(0, 1), (1, 1), (3, 1), (20, -1), (21, -1), (23, -1)],
      [(5, 1), (15, -1), (16, -1), (17, 1), (20, -1), (22, 1)], [(15, 1), (16, 1), (17, -1)],
      [(0, -1), (1, -1), (2, 1), (7, 1), (10, -1), (12, 1), (15, 1), (16, 1), (17, -1)],
      [(0, 1), (1, 1), (2, -1)],
      [(1, 1), (2, -1), (3, 1), (8, -1), (11, 1), (12, -1), (13, 1), (22, -1), (23, 1)],
      [(1, -1), (2, 1)], [(7, -1), (16, 1)], [(7, -1)],
      [(0, -1), (4, 1), (5, -1), (15, -1), (16, -1), (19, 1)],
      [(2, -1), (4, -1), (7, -1), (11, -1), (12, 1), (14, 1), (16, 1), (17, -1), (19, -1)],
      [(8, -1)], [(7, 1), (11, 1)], [(11, 1), (12, -1), (14, -1)],
      [(7, -1), (11, -1), (12, 1), (13, -1), (17, 1), (18, -1)],
      [(8, -1), (11, 1), (12, -1), (13, 1), (22, -1), (23, 1)], [(15, 1), (16, 1), (19, -1)]],
     [[(3, -1), (4, -1), (5, -1), (8, -1), (12, 1), (14, 1), (23, 1), (24, 1), (26, 1), (36, 1),
       (46, -1)],
      [(3, 1), (4, 1), (5, 1), (8, 1), (12, -1), (13, -1), (14, -1), (43, -1), (46, 1)],
      [(13, -1), (15, 1), (17, -1), (22, 1), (24, 1), (29, 1), (36, 1), (61, 1), (62, -1),
       (71, -1)],
      [(1, -1), (13, 1), (20, 1), (21, 1), (23, -1), (24, -1), (38, -1), (39, -1), (58, 1),
       (59, 1)],
      [(9, -1), (13, -1), (23, 1), (30, 1), (32, 1), (35, -1), (36, 1), (37, -1), (56, 1),
       (75, -1)],
      [(4, -1), (12, 1), (24, 1), (26, 1), (27, 1), (28, -1), (35, 1), (37, 1), (49, 1), (68, -1)],
      [(6, 1), (7, -1), (8, 1), (9, -1), (12, -1), (13, -1), (14, 1), (42, 1), (44, -1)],
      [(11, 1), (12, -1), (17, -1), (28, 1), (32, -1), (33, -1), (34, 1), (65, -1), (69, -1),
       (72, 1)],
      [(10, -1), (12, 1), (18, 1), (19, -1), (20, -1), (25, -1), (27, 1), (32, 1), (50, -1),
       (53, -1)],
      [(6, 1), (7, -1), (8, 1), (9, -1), (13, -1), (14, 1), (27, 1), (28, -1), (30, 1), (32, 1),
       (44, -1)],
      [(5, 1), (14, -1), (19, 1), (26, -1), (31, 1), (36, -1), (38, 1), (39, 1), (51, 1), (52, 1)],
      [(0, 1), (1, -1), (2, -1), (8, 1), (10, 1), (12, -1), (13, 1), (14, 1), (70, -1)],
      [(14, 1), (16, 1), (17, -1), (19, -1), (21, -1), (40, 1), (41, -1), (55, 1), (66, 1),
       (73, -1)],
      [(1, 1), (2, 1), (8, -1), (10, -1), (12, 1), (13, -1), (19, -1), (20, -1), (21, -1), (31, -1),
       (70, 1)],
      [(7, -1), (14, 1), (18, -1), (21, -1), (25, 1), (28, -1), (30, 1), (31, -1), (47, -1),
       (48, 1)],
      [(3, 1), (8, 1), (16, -1), (22, 1), (23, -1), (26, -1), (29, 1), (60, 1), (63, -1), (65, 1)],
      [(8, -1), (11, -1), (12, 1), (13, 1), (14, -1), (45, 1), (66, -1), (67, 1), (71, 1)],
      [(11, -1), (12, 1), (13, 1), (14, -1), (15, -1), (16, -1), (17, 1), (65, 1), (66, -1),
       (67, 1), (71, 1)],
      [(2, -1), (8, 1), (15, -1), (20, 1), (31, 1), (40, 1), (41, -1), (64, -1), (65, 1), (74, 1)],
      [(6, -1), (8, -1), (15, 1), (16, 1), (27, -1), (30, -1), (33, 1), (34, -1), (54, -1),
       (57, 1)]])


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
