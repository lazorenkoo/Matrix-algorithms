import numpy as np

if __name__ == "lab2.lu_compiled":
    from lab2.inverse_compiled import _inverse, _ZeroPivotError
    from lab2.multiplication_compiled import get_multiplication, multiply
else:
    from lab2.inverse import _inverse, _ZeroPivotError
    from lab2.multiplication import get_multiplication, multiply
from lab2.operations import Operations


def _lu(A: np.ndarray, algorithm, operations: Operations):
    size = A.shape[0]
    if size == 1:
        return np.ones((1, 1)), A.copy()

    mid = size // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    L11, U11 = _lu(A11, algorithm, operations)
    inverse_L11 = _inverse(L11, algorithm, operations)
    inverse_U11 = _inverse(U11, algorithm, operations)
    U12 = multiply(inverse_L11, A12, algorithm, operations)
    L21 = multiply(A21, inverse_U11, algorithm, operations)
    schur = A22 - multiply(L21, U12, algorithm, operations)
    operations.additions_subtractions += schur.size
    L22, U22 = _lu(schur, algorithm, operations)

    L = np.block([
        [L11, np.zeros((mid, size - mid))],
        [L21, L22],
    ])
    U = np.block([
        [U11, U12],
        [np.zeros((size - mid, mid)), U22],
    ])
    return L, U


def lu(A: np.ndarray, method: str = "binet") -> tuple[np.ndarray, np.ndarray, Operations]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("Expected a nonempty square matrix.")
    if not np.isfinite(A).all():
        raise ValueError("Matrix entries must be finite.")
    algorithm = get_multiplication(method)
    operations = Operations()
    try:
        L, U = _lu(A, algorithm, operations)
    except _ZeroPivotError as error:
        raise np.linalg.LinAlgError(
            "Block LU without row permutations failed: a leading pivot is zero."
        ) from error
    if not np.isfinite(L).all() or not np.isfinite(U).all():
        raise np.linalg.LinAlgError("LU factors cannot be represented with finite float64 values.")
    return L, U, operations
