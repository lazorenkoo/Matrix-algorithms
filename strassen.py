import numpy as np


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
