import numpy as np

from lab2.multiplication import get_multiplication, multiply
from lab2.operations import Operations


class _ZeroPivotError(np.linalg.LinAlgError):
    pass


def _pivot_rows(A: np.ndarray, width: int, operations: Operations):
    work = A[:, :width].copy()
    order = np.arange(A.shape[0])
    for j in range(width):
        pivot = j + int(np.argmax(np.abs(work[j:, j])))
        if work[pivot, j] == 0:
            raise _ZeroPivotError("Matrix is singular: no nonzero pivot is available.")
        if pivot != j:
            work[[j, pivot]] = work[[pivot, j]]
            order[[j, pivot]] = order[[pivot, j]]
        if j + 1 == width:
            break
        for i in range(j + 1, A.shape[0]):
            if work[i, j] == 0:
                continue
            factor = work[i, j] / work[j, j]
            operations.divisions += 1
            work[i, j + 1:] -= factor * work[j, j + 1:]
            operations.multiplications += width - j - 1
            operations.additions_subtractions += width - j - 1
            work[i, j] = 0
    return order


def _inverse(A: np.ndarray, algorithm, operations: Operations):
    size = A.shape[0]
    if size == 1:
        if A[0, 0] == 0:
            raise _ZeroPivotError("Matrix is singular: zero scalar pivot.")
        operations.divisions += 1
        return np.array([[1.0 / A[0, 0]]])

    mid = size // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    try:
        inverse11 = _inverse(A11, algorithm, operations)
    except _ZeroPivotError:
        order = _pivot_rows(A, mid, operations)
        inverse_permuted = _inverse(A[order, :], algorithm, operations)
        return inverse_permuted[:, np.argsort(order)]

    left = multiply(inverse11, A12, algorithm, operations)
    right = multiply(A21, inverse11, algorithm, operations)
    schur = A22 - multiply(right, A12, algorithm, operations)
    operations.additions_subtractions += schur.size
    inverse_schur = _inverse(schur, algorithm, operations)

    upper_right = multiply(left, inverse_schur, algorithm, operations)
    lower_left = multiply(inverse_schur, right, algorithm, operations)
    upper_left = inverse11 + multiply(upper_right, right, algorithm, operations)
    operations.additions_subtractions += upper_left.size

    return np.block([
        [upper_left, -upper_right],
        [-lower_left, inverse_schur],
    ])


def inverse(A: np.ndarray, method: str = "binet") -> tuple[np.ndarray, Operations]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("Expected a nonempty square matrix.")
    if not np.isfinite(A).all():
        raise ValueError("Matrix entries must be finite.")
    algorithm = get_multiplication(method)
    operations = Operations()
    result = _inverse(A, algorithm, operations)
    if not np.isfinite(result).all():
        raise np.linalg.LinAlgError("The inverse cannot be represented with finite float64 values.")
    return result, operations
