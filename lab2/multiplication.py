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
    block_size = min(rows, common, cols)
    if max(rows, common, cols) > 2 * block_size:
        C = np.zeros((rows, cols))
        multiplications = 0
        additions = 0
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                height = min(block_size, rows - i)
                width = min(block_size, cols - j)
                for k in range(0, common, block_size):
                    depth = min(block_size, common - k)
                    padded_A = np.zeros((block_size, block_size))
                    padded_B = np.zeros((block_size, block_size))
                    padded_A[:height, :depth] = A[i:i + height, k:k + depth]
                    padded_B[:depth, :width] = B[k:k + depth, j:j + width]
                    part, mul, add = strassen(padded_A, padded_B)
                    if k == 0:
                        C[i:i + height, j:j + width] = part[:height, :width]
                    else:
                        C[i:i + height, j:j + width] += part[:height, :width]
                        additions += height * width
                    multiplications += mul
                    additions += add
        return C, multiplications, additions
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
