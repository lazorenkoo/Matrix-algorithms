from time import perf_counter

import numpy as np


def generate_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    low = np.nextafter(0.00000001, 1.0)
    high = np.nextafter(1.0, 0.0)
    return rng.uniform(low, high, size=(n, n))


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


if __name__ == "__main__":
    n = 4
    A = generate_matrix(n)
    B = generate_matrix(n)

    start = perf_counter()
    C, multiplications, additions = binet(A, B)
    elapsed = perf_counter() - start

    print("A:\n", A)
    print("B:\n", B)
    print("C:\n", C)
    print("Multiplications:", multiplications)
    print("Additions:", additions)
    print("FLOPs:", multiplications + additions)
    print("Time (seconds):", elapsed)
