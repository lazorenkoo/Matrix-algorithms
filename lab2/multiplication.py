import numpy as np

if __name__ == "lab2.multiplication_compiled":
    from lab1.binet_compiled import binet
    from lab1.strassen_compiled import multiply_direct, strassen
    from lab2.laderman_compiled import laderman
else:
    from lab1.binet import binet
    from lab1.strassen import multiply_direct, strassen
    from lab2.laderman import laderman
from lab2.operations import Operations


def strassen_rectangular(A: np.ndarray, B: np.ndarray):
    rows, common = A.shape
    cols = B.shape[1]
    if min(rows, common, cols) == 1:
        return multiply_direct(A, B)
    if rows == common == cols:
        return strassen(A, B)
    size = max(rows, common, cols)
    padded_A = np.zeros((size, size))
    padded_B = np.zeros((size, size))
    padded_A[:rows, :common] = A
    padded_B[:common, :cols] = B
    C, multiplications, additions = strassen(padded_A, padded_B)
    return C[:rows, :cols], multiplications, additions


def get_multiplication(method: str):
    return {"binet": binet, "strassen": strassen_rectangular, "laderman": laderman}[method]


def multiply(A: np.ndarray, B: np.ndarray, algorithm, operations: Operations):
    C, multiplications, additions = algorithm(A, B)
    operations.multiplications += multiplications
    operations.additions_subtractions += additions
    return C
