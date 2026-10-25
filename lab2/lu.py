import numpy as np

if __name__ == "lab2.lu_compiled":
    from lab2.inverse_compiled import _ZeroPivotError
    from lab2.multiplication_compiled import get_multiplication, multiply
else:
    from lab2.inverse import _ZeroPivotError
    from lab2.multiplication import get_multiplication, multiply
from lab2.operations import Operations


def _solve_lower(L: np.ndarray, B: np.ndarray, algorithm, operations: Operations):
    if L.shape[0] == 1:
        return B.copy()
    mid = L.shape[0] // 2
    top = _solve_lower(L[:mid, :mid], B[:mid, :], algorithm, operations)
    remainder = B[mid:, :] - multiply(L[mid:, :mid], top, algorithm, operations)
    operations.additions_subtractions += remainder.size
    bottom = _solve_lower(L[mid:, mid:], remainder, algorithm, operations)
    return np.vstack((top, bottom))


def _solve_upper_right(B: np.ndarray, U: np.ndarray, algorithm, operations: Operations):
    if U.shape[0] == 1:
        if U[0, 0] == 0:
            raise _ZeroPivotError("Zero upper-triangular pivot.")
        operations.divisions += B.size
        return B / U[0, 0]
    mid = U.shape[0] // 2
    left = _solve_upper_right(B[:, :mid], U[:mid, :mid], algorithm, operations)
    remainder = B[:, mid:] - multiply(left, U[:mid, mid:], algorithm, operations)
    operations.additions_subtractions += remainder.size
    right = _solve_upper_right(remainder, U[mid:, mid:], algorithm, operations)
    return np.hstack((left, right))


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
    U12 = _solve_lower(L11, A12, algorithm, operations)
    L21 = _solve_upper_right(A21, U11, algorithm, operations)
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
