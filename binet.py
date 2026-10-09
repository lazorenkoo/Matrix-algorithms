import numpy as np


def binet(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Multiply equal-sized square matrices; return C, multiplications, additions.

    Inputs must be nonempty square matrices of equal size, a power of two.
    """
    n = A.shape[0]

    if n == 1:
        C = np.array([[A[0, 0] * B[0, 0]]])
        return C, 1, 0

    mid = n // 2

    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

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
