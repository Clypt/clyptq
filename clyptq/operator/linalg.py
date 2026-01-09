"""Linear Algebra operators for matrix operations.

This module provides Expr-based linear algebra operations.
All functions return Expr and map 1:1 to Rust backend.

Rust mapping:
    - matmul -> nalgebra: A * B
    - transpose -> nalgebra: A.transpose()
    - linalg_inv -> nalgebra: A.try_inverse()
    - linalg_solve -> nalgebra: A.lu().solve(&b)
    - linalg_lstsq -> nalgebra: A.svd().solve(&b, eps)
    - linalg_svd -> nalgebra: A.svd(true, true)
    - etc.

Usage:
    ```python
    from clyptq.operator.linalg import matmul, linalg_inv, linalg_lstsq

    # Mean-Variance Optimization
    cov_inv = linalg_inv(cov_matrix, regularize=1e-6)
    optimal_weights = matmul(cov_inv, expected_returns)

    # OLS Regression
    beta = linalg_lstsq(X, y)
    residuals = y - matmul(X, beta)
    ```
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clyptq.operator.base import Expr, OpCode, _ensure_expr


# --- Basic Matrix Operations ---

def matmul(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    B: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, pd.DataFrame, np.ndarray]:
    """Matrix multiplication: A @ B

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)
        lazy: If True, always return Expr

    Returns:
        Matrix product (m x p)

    Example:
        ```python
        # Compute optimal weights
        weights = matmul(cov_inv, mu)

        # Linear transformation
        transformed = matmul(rotation_matrix, data)
        ```
    """
    has_expr = isinstance(A, Expr) or isinstance(B, Expr)
    if has_expr or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        B_expr = _ensure_expr(B) if not isinstance(B, Expr) else B
        return Expr(OpCode.MATMUL, inputs=[A_expr, B_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    B_val = B.values if isinstance(B, pd.DataFrame) else B
    result = A_val @ B_val

    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        return pd.DataFrame(result, index=A.index, columns=B.columns)
    elif isinstance(A, pd.DataFrame):
        return pd.DataFrame(result, index=A.index)
    elif isinstance(B, pd.DataFrame):
        return pd.DataFrame(result, columns=B.columns)
    return result


def transpose(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, pd.DataFrame, np.ndarray]:
    """Transpose: A.T

    Args:
        A: Input matrix
        lazy: If True, always return Expr

    Returns:
        Transposed matrix
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.TRANSPOSE, inputs=[A_expr])

    # Eager execution
    if isinstance(A, pd.DataFrame):
        return A.T
    return A.T


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: type = float,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Create identity matrix

    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Data type
        lazy: If True, always return Expr

    Returns:
        n x m identity matrix
    """
    if lazy:
        return Expr(OpCode.EYE, args=(n,), kwargs={"m": m, "dtype": dtype})

    # Eager execution
    return np.eye(n, m, dtype=dtype)


def diag(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    k: int = 0,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Create diagonal matrix or extract diagonal

    Args:
        A: Vector (create diagonal matrix) or matrix (extract diagonal)
        k: Diagonal offset (0: main diagonal)
        lazy: If True, always return Expr

    Returns:
        Diagonal matrix or diagonal elements
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.DIAG, kwargs={"k": k}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.diag(A_val, k=k)


def trace(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, float]:
    """Trace (sum of diagonal)

    Args:
        A: Square matrix
        lazy: If True, always return Expr

    Returns:
        Sum of diagonal elements
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.TRACE, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.trace(A_val)


# --- Matrix Decompositions ---

def linalg_lu(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, Dict[str, np.ndarray]]:
    """LU decomposition: A = P @ L @ U

    Args:
        A: Matrix to decompose
        lazy: If True, always return Expr

    Returns:
        {"P": P, "L": L, "U": U} dictionary
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_LU, inputs=[A_expr])

    # Eager execution
    from scipy import linalg as sp_linalg
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    P, L, U = sp_linalg.lu(A_val)
    return {"P": P, "L": L, "U": U}


def linalg_qr(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    mode: str = "reduced",
    lazy: bool = False,
) -> Union[Expr, Dict[str, np.ndarray]]:
    """QR decomposition: A = Q @ R

    Args:
        A: Matrix to decompose
        mode: "reduced" or "complete"
        lazy: If True, always return Expr

    Returns:
        {"Q": Q, "R": R} dictionary
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_QR, kwargs={"mode": mode}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    Q, R = np.linalg.qr(A_val, mode=mode)
    return {"Q": Q, "R": R}


def linalg_svd(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    full_matrices: bool = True,
    compute_uv: bool = True,
    lazy: bool = False,
) -> Union[Expr, Dict[str, np.ndarray]]:
    """SVD decomposition: A = U @ diag(s) @ Vh

    Args:
        A: Matrix to decompose
        full_matrices: True for full size, False for reduced form
        compute_uv: Whether to compute U, Vh
        lazy: If True, always return Expr

    Returns:
        {"U": U, "s": s, "Vh": Vh} dictionary

    Example:
        ```python
        result = linalg_svd(A)
        U, s, Vh = result["U"], result["s"], result["Vh"]

        # Dimensionality reduction (k principal components)
        A_approx = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
        ```
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(
            OpCode.LINALG_SVD,
            kwargs={"full_matrices": full_matrices, "compute_uv": compute_uv},
            inputs=[A_expr]
        )

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    if compute_uv:
        U, s, Vh = np.linalg.svd(A_val, full_matrices=full_matrices)
        return {"U": U, "s": s, "Vh": Vh}
    else:
        s = np.linalg.svd(A_val, full_matrices=full_matrices, compute_uv=False)
        return {"s": s}


def linalg_cholesky(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    lower: bool = True,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Cholesky decomposition: A = L @ L.T (positive-definite symmetric matrix)

    Args:
        A: Positive-definite symmetric matrix
        lower: True for lower triangular matrix, False for upper triangular matrix
        lazy: If True, always return Expr

    Returns:
        Cholesky factor L

    Example:
        ```python
        # Covariance matrix decomposition (used for correlated sampling)
        L = linalg_cholesky(cov_matrix)
        correlated_samples = matmul(L, uncorrelated_samples)
        ```
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_CHOLESKY, kwargs={"lower": lower}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    L = np.linalg.cholesky(A_val)
    if not lower:
        L = L.T
    return L


def linalg_eigen(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    hermitian: bool = False,
    lazy: bool = False,
) -> Union[Expr, Dict[str, np.ndarray]]:
    """Eigenvalue decomposition: A @ v = lambda @ v

    Args:
        A: Matrix to decompose
        hermitian: True for Hermitian/symmetric matrix (more efficient)
        lazy: If True, always return Expr

    Returns:
        {"eigenvalues": lambda, "eigenvectors": V} dictionary

    Example:
        ```python
        # PCA
        result = linalg_eigen(cov_matrix, hermitian=True)
        principal_components = result["eigenvectors"]
        variance_explained = result["eigenvalues"]
        ```
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_EIGEN, kwargs={"hermitian": hermitian}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(A_val)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(A_val)
    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


# --- Matrix Properties ---

def linalg_det(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, float]:
    """Determinant

    Args:
        A: Square matrix
        lazy: If True, always return Expr

    Returns:
        Determinant value
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_DET, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.linalg.det(A_val)


def linalg_rank(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    tol: Optional[float] = None,
    lazy: bool = False,
) -> Union[Expr, int]:
    """Matrix rank

    Args:
        A: Input matrix
        tol: Singular value threshold (default: based on machine precision)
        lazy: If True, always return Expr

    Returns:
        Matrix rank
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_RANK, kwargs={"tol": tol}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.linalg.matrix_rank(A_val, tol=tol)


def linalg_norm(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    ord: Optional[Union[int, float, str]] = None,
    axis: Optional[int] = None,
    lazy: bool = False,
) -> Union[Expr, float, np.ndarray]:
    """Matrix/vector norm

    Args:
        A: Input matrix or vector
        ord: Norm order (2, 'fro', 'nuc', inf, -inf, 1, -1, ...)
        axis: Axis (None for entire matrix)
        lazy: If True, always return Expr

    Returns:
        Norm value
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_NORM, kwargs={"ord": ord, "axis": axis}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.linalg.norm(A_val, ord=ord, axis=axis)


def linalg_cond(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    p: Optional[Union[int, float, str]] = None,
    lazy: bool = False,
) -> Union[Expr, float]:
    """Condition number

    Args:
        A: Input matrix
        p: Norm order
        lazy: If True, always return Expr

    Returns:
        Condition number
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_COND, kwargs={"p": p}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.linalg.cond(A_val, p=p)


# --- Linear System Solvers ---

def linalg_inv(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    regularize: float = 0.0,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Inverse matrix

    Args:
        A: Square matrix
        regularize: Regularization value (computes inverse of A + lambda*I)
        lazy: If True, always return Expr

    Returns:
        Inverse matrix

    Example:
        ```python
        # Mean-Variance Optimization
        cov_inv = linalg_inv(cov_matrix, regularize=1e-6)
        optimal_weights = matmul(cov_inv, expected_returns) / risk_aversion
        ```
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_INV, kwargs={"regularize": regularize}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    if regularize > 0:
        A_val = A_val + regularize * np.eye(A_val.shape[0])
    return np.linalg.inv(A_val)


def linalg_pinv(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    rcond: Optional[float] = None,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Pseudo-inverse (Moore-Penrose pseudo-inverse)

    Args:
        A: Input matrix (does not need to be square)
        rcond: Small singular value threshold
        lazy: If True, always return Expr

    Returns:
        Pseudo-inverse

    Note:
        - Can be used for non-square matrices
        - Can be used for singular matrices (ignores small singular values)
    """
    if isinstance(A, Expr) or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        return Expr(OpCode.LINALG_PINV, kwargs={"rcond": rcond}, inputs=[A_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    return np.linalg.pinv(A_val, rcond=rcond)


def linalg_solve(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    b: Union[Expr, pd.DataFrame, np.ndarray],
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Linear system solution: Ax = b

    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n,) or matrix (n x k)
        lazy: If True, always return Expr

    Returns:
        Solution x

    Note:
        Use when A is square and invertible.
        For non-square/singular matrices, use linalg_lstsq.
    """
    has_expr = isinstance(A, Expr) or isinstance(b, Expr)
    if has_expr or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        b_expr = _ensure_expr(b) if not isinstance(b, Expr) else b
        return Expr(OpCode.LINALG_SOLVE, inputs=[A_expr, b_expr])

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    b_val = b.values if isinstance(b, pd.DataFrame) else b
    return np.linalg.solve(A_val, b_val)


def linalg_lstsq(
    A: Union[Expr, pd.DataFrame, np.ndarray],
    b: Union[Expr, pd.DataFrame, np.ndarray],
    rcond: Optional[float] = None,
    return_residuals: bool = False,
    lazy: bool = False,
) -> Union[Expr, np.ndarray, Dict[str, Any]]:
    """Least squares solution (OLS): min ||Ax - b||^2

    Args:
        A: Design matrix (n x p)
        b: Target vector/matrix (n,) or (n x k)
        rcond: Small singular value threshold
        return_residuals: If True, return additional info including residuals
        lazy: If True, always return Expr

    Returns:
        Coefficients x (return_residuals=False)
        or {"x", "residuals", "rank", "s"} (return_residuals=True)

    Example:
        ```python
        # OLS regression
        X = np.column_stack([np.ones(n), features])  # Include intercept
        beta = linalg_lstsq(X, y)
        predictions = matmul(X, beta)

        # Ridge regression (direct implementation)
        XtX = matmul(transpose(X), X)
        XtX_reg = XtX + eye(p) * lambda_
        Xty = matmul(transpose(X), y)
        beta_ridge = linalg_solve(XtX_reg, Xty)
        ```
    """
    has_expr = isinstance(A, Expr) or isinstance(b, Expr)
    if has_expr or lazy:
        A_expr = _ensure_expr(A) if not isinstance(A, Expr) else A
        b_expr = _ensure_expr(b) if not isinstance(b, Expr) else b
        return Expr(
            OpCode.LINALG_LSTSQ,
            kwargs={"rcond": rcond, "return_residuals": return_residuals},
            inputs=[A_expr, b_expr]
        )

    # Eager execution
    A_val = A.values if isinstance(A, pd.DataFrame) else A
    b_val = b.values if isinstance(b, pd.DataFrame) else b
    x, residuals, rank, s = np.linalg.lstsq(A_val, b_val, rcond=rcond)
    if return_residuals:
        return {"x": x, "residuals": residuals, "rank": rank, "s": s}
    return x


# --- Higher-level functions built on primitives ---

def ols(
    X: Union[Expr, pd.DataFrame, np.ndarray],
    y: Union[Expr, pd.DataFrame, np.ndarray],
    add_intercept: bool = True,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """OLS regression (convenience function)

    Args:
        X: Independent variables (n x p)
        y: Dependent variable (n,)
        add_intercept: Whether to add intercept
        lazy: If True, always return Expr

    Returns:
        Coefficients (intercept is first if add_intercept=True)
    """
    has_expr = isinstance(X, Expr) or isinstance(y, Expr)
    if has_expr or lazy:
        return linalg_lstsq(X, y, lazy=True)

    # Eager execution
    X_val = X.values if isinstance(X, pd.DataFrame) else X
    y_val = y.values if isinstance(y, pd.DataFrame) else y
    if add_intercept:
        X_val = np.column_stack([np.ones(X_val.shape[0]), X_val])
    return linalg_lstsq(X_val, y_val)


def ridge(
    X: Union[Expr, pd.DataFrame, np.ndarray],
    y: Union[Expr, pd.DataFrame, np.ndarray],
    lambda_: float = 1.0,
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Ridge regression: min ||X*beta - y||^2 + lambda*||beta||^2

    Args:
        X: Independent variables (n x p)
        y: Dependent variable (n,)
        lambda_: Regularization strength
        lazy: If True, always return Expr

    Returns:
        Coefficients beta

    Note:
        beta = (X'X + lambda*I)^(-1) X'y
    """
    has_expr = isinstance(X, Expr) or isinstance(y, Expr)
    if has_expr or lazy:
        X_expr = _ensure_expr(X) if not isinstance(X, Expr) else X
        y_expr = _ensure_expr(y) if not isinstance(y, Expr) else y
        return Expr(
            OpCode.CUSTOM,
            kwargs={"func": "ridge", "lambda_": lambda_},
            inputs=[X_expr, y_expr]
        )

    # Eager execution: beta = (X'X + lambda*I)^(-1) X'y
    X_val = X.values if isinstance(X, pd.DataFrame) else X
    y_val = y.values if isinstance(y, pd.DataFrame) else y
    p = X_val.shape[1]
    XtX = X_val.T @ X_val
    XtX_reg = XtX + lambda_ * np.eye(p)
    Xty = X_val.T @ y_val
    return np.linalg.solve(XtX_reg, Xty)


# --- Cross-Alpha Operations (for Combiner) ---

def ca_stack(
    *alphas: Union[Expr, pd.DataFrame],
    lazy: bool = False,
) -> Union[Expr, np.ndarray]:
    """Stack multiple alphas into a 3D tensor

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        (T, N, K) tensor where K=number of alphas
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_STACK, inputs=alpha_exprs)

    # Eager execution
    return np.stack([a.values for a in alphas], axis=-1)


def ca_weighted_sum(
    *alphas: Union[Expr, pd.DataFrame],
    weights: Union[Dict[str, float], List[float]],
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha weighted average

    Args:
        *alphas: Alpha DataFrames
        weights: Weights (dict or list)
        lazy: If True, always return Expr

    Returns:
        Weighted average alpha (T x N)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_WEIGHTED_SUM, kwargs={"weights": weights}, inputs=alpha_exprs)

    # Eager execution
    if isinstance(weights, dict):
        w = list(weights.values())
    else:
        w = list(weights)

    w = np.array(w, dtype=float)

    # Align all DataFrames to common index and columns
    common_index = alphas[0].index
    common_columns = alphas[0].columns
    for a in alphas[1:]:
        common_index = common_index.intersection(a.index)
        common_columns = common_columns.intersection(a.columns)

    aligned = [a.loc[common_index, common_columns].values for a in alphas]
    stacked = np.stack(aligned, axis=-1)
    result = np.nansum(stacked * w, axis=-1)
    return pd.DataFrame(result, index=common_index, columns=common_columns)


def ca_rank_average(
    *alphas: Union[Expr, pd.DataFrame],
    weights: Optional[Union[Dict[str, float], List[float]]] = None,
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha rank average (rank each alpha then average)

    Args:
        *alphas: Alpha DataFrames
        weights: Weights (None for equal weights)
        lazy: If True, always return Expr

    Returns:
        Rank average alpha (T x N)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_RANK_AVERAGE, kwargs={"weights": weights}, inputs=alpha_exprs)

    # Eager execution: rank each alpha then average
    from clyptq.operator.cross_section import rank

    ranked = [rank(a) for a in alphas]
    if weights is None:
        w = np.ones(len(ranked)) / len(ranked)
    elif isinstance(weights, dict):
        w = np.array(list(weights.values()), dtype=float)
    else:
        w = np.array(weights, dtype=float)

    stacked = np.stack([r.values for r in ranked], axis=-1)
    result = np.nansum(stacked * w, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_ic_weight(
    *alphas: Union[Expr, pd.DataFrame],
    returns: Union[Expr, pd.DataFrame],
    lookback: int = 20,
    min_periods: int = 5,
    lazy: bool = False,
) -> Union[Expr, pd.DataFrame]:
    """IC-based dynamic weight combination

    Args:
        *alphas: Alpha DataFrames
        returns: Returns DataFrame
        lookback: IC rolling window
        min_periods: Minimum periods
        lazy: If True, always return Expr

    Returns:
        IC weighted combined alpha
    """
    has_expr = any(isinstance(a, Expr) for a in alphas) or isinstance(returns, Expr)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        returns_expr = _ensure_expr(returns) if not isinstance(returns, Expr) else returns
        all_inputs = alpha_exprs + [returns_expr]
        return Expr(
            OpCode.CA_IC_WEIGHT,
            kwargs={"lookback": lookback, "min_periods": min_periods},
            inputs=all_inputs
        )

    # Eager execution: compute rolling IC for each alpha, then weighted sum
    returns_val = returns.values if isinstance(returns, pd.DataFrame) else returns
    n_alphas = len(alphas)

    # Compute rolling IC for each alpha
    ics = []
    for alpha in alphas:
        alpha_val = alpha.values if isinstance(alpha, pd.DataFrame) else alpha
        # Rolling correlation between alpha and returns
        ic = pd.DataFrame(alpha_val).rolling(lookback, min_periods=min_periods).corr(
            pd.DataFrame(returns_val)
        ).values
        ics.append(ic)

    # Stack ICs: (T, N, K)
    ic_stack = np.stack(ics, axis=-1)
    # Normalize ICs to weights (softmax-like, handle NaNs)
    ic_abs = np.abs(ic_stack)
    ic_sum = np.nansum(ic_abs, axis=-1, keepdims=True)
    weights = np.where(ic_sum > 1e-10, ic_abs / ic_sum, 1.0 / n_alphas)

    # Weighted combination
    alpha_stack = np.stack([a.values for a in alphas], axis=-1)
    result = np.nansum(alpha_stack * weights, axis=-1)

    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_corr(
    *alphas: Union[Expr, pd.DataFrame],
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Inter-alpha correlation matrix

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        K x K correlation matrix - DataFrame if all inputs are DataFrames and lazy=False

    Example:
        >>> corr_matrix = ca_corr(alpha1, alpha2, alpha3)
        >>> # Returns 3x3 correlation matrix
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)

    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_CORR, inputs=alpha_exprs)

    # Eager execution: compute K x K correlation matrix
    K = len(alphas)
    # Flatten each alpha to 1D vector (ignoring NaN)
    flattened = []
    for a in alphas:
        flat = a.values.flatten()
        flattened.append(flat)

    # Stack into (N_total, K) matrix
    stacked = np.column_stack(flattened)

    # Compute pairwise correlations
    corr_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            mask = ~(np.isnan(stacked[:, i]) | np.isnan(stacked[:, j]))
            if mask.sum() > 1:
                corr_matrix[i, j] = np.corrcoef(stacked[mask, i], stacked[mask, j])[0, 1]
            else:
                corr_matrix[i, j] = np.nan

    # Return as DataFrame with alpha indices
    labels = [f"alpha_{i}" for i in range(K)]
    return pd.DataFrame(corr_matrix, index=labels, columns=labels)


# --- Cross-Alpha Reduce Operations (WQ Brain: reduce_*) ---
#
# All ca_reduce_* operations are **pointwise**:
# - Input: K (T x N) alphas
# - Operation: Aggregate K values at each (t, n) position
# - Output: (T x N)
#
# If time-axis statistics are needed, apply ts_* operations first, then use ca_reduce_*.
# Example: To average the rolling IR of each alpha:
#     ir1 = ts_mean(alpha1, 60) / ts_std(alpha1, 60)
#     ir2 = ts_mean(alpha2, 60) / ts_std(alpha2, 60)
#     combined = ca_reduce_avg(ir1, ir2)

def ca_reduce_avg(
    *alphas: Union[Expr, pd.DataFrame],
    threshold: int = 0,
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha average (reduce_avg) - Pointwise

    Computes the average of K alpha values at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        threshold: Minimum number of valid values (0 for no limit)
        lazy: If True, always return Expr

    Returns:
        Average alpha (T x N) - DataFrame if all inputs are DataFrames and lazy=False

    Example:
        >>> # Simple average
        >>> combined = ca_reduce_avg(alpha1, alpha2, alpha3)

        >>> # Average after time-axis statistics (average of rolling means)
        >>> smoothed1 = ts_mean(alpha1, 20)
        >>> smoothed2 = ts_mean(alpha2, 20)
        >>> combined = ca_reduce_avg(smoothed1, smoothed2)
    """
    # Check if any input is Expr
    has_expr = any(isinstance(a, Expr) for a in alphas)

    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_AVG, kwargs={"threshold": threshold}, inputs=alpha_exprs)

    # Eager execution: all inputs are DataFrames
    stacked = np.stack([a.values for a in alphas], axis=-1)
    if threshold > 0:
        valid_count = np.sum(~np.isnan(stacked), axis=-1)
        result = np.where(valid_count >= threshold, np.nanmean(stacked, axis=-1), np.nan)
    else:
        result = np.nanmean(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_sum(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha sum (reduce_sum) - Pointwise

    Computes the sum of K alpha values at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Sum alpha (T x N)

    Example:
        >>> combined = ca_reduce_sum(alpha1, alpha2, alpha3)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_SUM, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nansum(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_max(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha maximum (reduce_max) - Pointwise

    Selects the maximum value among K alphas at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Maximum alpha (T x N)

    Example:
        >>> # Select strongest signal
        >>> strongest = ca_reduce_max(momentum_alpha, value_alpha, quality_alpha)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_MAX, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nanmax(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_min(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha minimum (reduce_min) - Pointwise

    Selects the minimum value among K alphas at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Minimum alpha (T x N)

    Example:
        >>> # Select most conservative signal
        >>> conservative = ca_reduce_min(alpha1, alpha2, alpha3)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_MIN, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nanmin(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_stddev(
    *alphas: Union[Expr, pd.DataFrame],
    threshold: float = 0,
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha standard deviation (reduce_stddev) - Pointwise

    Computes the standard deviation of K alpha values at each (t, n) position.
    Useful for measuring disagreement among alphas.

    Args:
        *alphas: Alpha DataFrames (T x N)
        threshold: Minimum valid ratio (0 for no limit)
        lazy: If True, always return Expr

    Returns:
        Standard deviation (T x N)

    Example:
        >>> # Measure disagreement among alphas
        >>> disagreement = ca_reduce_stddev(alpha1, alpha2, alpha3)
        >>> # Higher disagreement means higher uncertainty
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_STDDEV, kwargs={"threshold": threshold}, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    if threshold > 0:
        valid_count = np.sum(~np.isnan(stacked), axis=-1)
        valid_ratio = valid_count / len(alphas)
        result = np.where(valid_ratio >= threshold, np.nanstd(stacked, axis=-1, ddof=1), np.nan)
    else:
        result = np.nanstd(stacked, axis=-1, ddof=1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_ir(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Inter-alpha IR (reduce_ir) - Pointwise

    Computes mean/std of K alpha values at each (t, n) position.
    Measures strength relative to consensus among alphas.

    NOTE: This is NOT "time-axis IR"!
    For time-axis IR, combine ts_* operations.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Inter-alpha IR (T x N)

    Example:
        >>> # Pointwise IR (inter-alpha)
        >>> consensus_strength = ca_reduce_ir(alpha1, alpha2, alpha3)

        >>> # For time-axis IR (recommended pattern):
        >>> ir1 = ts_mean(alpha1, 60) / ts_std(alpha1, 60)
        >>> ir2 = ts_mean(alpha2, 60) / ts_std(alpha2, 60)
        >>> ir3 = ts_mean(alpha3, 60) / ts_std(alpha3, 60)
        >>> avg_ir = ca_reduce_avg(ir1, ir2, ir3)  # Average of each alpha's IR
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_IR, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    mean_val = np.nanmean(stacked, axis=-1)
    std_val = np.nanstd(stacked, axis=-1, ddof=1)
    result = np.where(std_val > 1e-10, mean_val / std_val, 0.0)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_skewness(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha skewness (reduce_skewness) - Pointwise

    Computes the skewness of K alpha values at each (t, n) position.
    Measures asymmetry of distribution among alphas.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Skewness (T x N)

    Example:
        >>> skew = ca_reduce_skewness(alpha1, alpha2, alpha3, alpha4, alpha5)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_SKEWNESS, inputs=alpha_exprs)

    from scipy import stats as scipy_stats
    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = scipy_stats.skew(stacked, axis=-1, nan_policy='omit')
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_kurtosis(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha kurtosis (reduce_kurtosis) - Pointwise

    Computes the kurtosis of K alpha values at each (t, n) position.
    Measures tail thickness of distribution among alphas.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Kurtosis (T x N)

    Example:
        >>> kurt = ca_reduce_kurtosis(alpha1, alpha2, alpha3, alpha4, alpha5)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_KURTOSIS, inputs=alpha_exprs)

    from scipy import stats as scipy_stats
    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = scipy_stats.kurtosis(stacked, axis=-1, nan_policy='omit')
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_range(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha range (reduce_range) - Pointwise

    Computes the range (max - min) of K alpha values at each (t, n) position.
    Measures spread among alphas.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Range (T x N)

    Example:
        >>> spread = ca_reduce_range(alpha1, alpha2, alpha3)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_RANGE, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nanmax(stacked, axis=-1) - np.nanmin(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_median(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha median (reduce_median) - Pointwise

    Computes the median of K alpha values at each (t, n) position.
    Creates outlier-robust combined alpha.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        Median alpha (T x N)

    Example:
        >>> # Outlier-robust combination
        >>> robust_combined = ca_reduce_median(alpha1, alpha2, alpha3, alpha4, alpha5)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_MEDIAN, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nanmedian(stacked, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_count(
    *alphas: Union[Expr, pd.DataFrame],
    threshold: float = 0,
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Valid alpha count (reduce_count) - Pointwise

    Counts alphas exceeding threshold at each (t, n) position.
    Measures the number of agreeing alphas.

    Args:
        *alphas: Alpha DataFrames (T x N)
        threshold: Count threshold (0 for counting non-NaN values)
        lazy: If True, always return Expr

    Returns:
        Valid count (T x N)

    Example:
        >>> # Count positive signals
        >>> bullish_count = ca_reduce_count(alpha1, alpha2, alpha3, threshold=0)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_COUNT, kwargs={"threshold": threshold}, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    if threshold != 0:
        result = np.sum(stacked > threshold, axis=-1)
    else:
        result = np.sum(~np.isnan(stacked), axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_norm(*alphas: Union[Expr, pd.DataFrame], lazy: bool = False) -> Union[Expr, pd.DataFrame]:
    """Alpha L2 norm (reduce_norm) - Pointwise

    Computes the L2 norm (sqrt(sum(x^2))) of K alpha values at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lazy: If True, always return Expr

    Returns:
        L2 norm (T x N)

    Example:
        >>> magnitude = ca_reduce_norm(alpha1, alpha2, alpha3)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_NORM, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.sqrt(np.nansum(stacked ** 2, axis=-1))
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_reduce_powersum(
    *alphas: Union[Expr, pd.DataFrame],
    power: float = 2,
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha power sum (reduce_powersum) - Pointwise

    Computes sum(|alpha|^power) at each (t, n) position.

    Args:
        *alphas: Alpha DataFrames (T x N)
        power: Power exponent (default: 2)
        lazy: If True, always return Expr

    Returns:
        Power sum (T x N)

    Example:
        >>> power_sum = ca_reduce_powersum(alpha1, alpha2, alpha3, power=2)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)
    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(OpCode.CA_REDUCE_POWERSUM, kwargs={"power": power}, inputs=alpha_exprs)

    stacked = np.stack([a.values for a in alphas], axis=-1)
    result = np.nansum(np.abs(stacked) ** power, axis=-1)
    return pd.DataFrame(result, index=alphas[0].index, columns=alphas[0].columns)


def ca_combo_a(
    *alphas: Union[Expr, pd.DataFrame],
    lookback: int = 250,
    mode: str = "algo1",
    lazy: bool = False
) -> Union[Expr, pd.DataFrame]:
    """Alpha combination (combo_a) - Includes temporal operations

    NOTE: This function uses time-axis statistics internally.
    Compatible with WQ Brain's combo_a: rolling IC-based dynamic weights.

    For pure pointwise combination, use ca_reduce_avg or ca_weighted_sum.

    Args:
        *alphas: Alpha DataFrames (T x N)
        lookback: IC calculation lookback period (default: 250 days)
        mode: Weight algorithm
            - algo1: Simple IR weights
            - algo2: Volatility-adjusted IR
            - algo3: Decayed IR
        lazy: If True, always return Expr

    Returns:
        Combined alpha (T x N) - DataFrame if all inputs are DataFrames and lazy=False

    Example:
        >>> # Use built-in temporal logic
        >>> combined = ca_combo_a(alpha1, alpha2, alpha3, lookback=60)

        >>> # Or manually compute IR then weighted average (more flexible):
        >>> ir1 = ts_mean(alpha1, 60) / ts_std(alpha1, 60)
        >>> ir2 = ts_mean(alpha2, 60) / ts_std(alpha2, 60)
        >>> weights = softmax(ca_stack(ir1, ir2))  # IR-based weights
        >>> combined = ca_weighted_sum(alpha1, alpha2, weights=weights)
    """
    has_expr = any(isinstance(a, Expr) for a in alphas)

    if has_expr or lazy:
        alpha_exprs = [_ensure_expr(a) if not isinstance(a, Expr) else a for a in alphas]
        return Expr(
            OpCode.CA_COMBO_A,
            kwargs={"lookback": lookback, "mode": mode},
            inputs=alpha_exprs
        )

    # Eager execution: compute rolling IR-weighted combination
    K = len(alphas)
    T, N = alphas[0].shape

    # Align all DataFrames
    common_index = alphas[0].index
    common_columns = alphas[0].columns
    for a in alphas[1:]:
        common_index = common_index.intersection(a.index)
        common_columns = common_columns.intersection(a.columns)

    aligned = [a.loc[common_index, common_columns] for a in alphas]
    T, N = len(common_index), len(common_columns)

    # Compute rolling IR for each alpha
    irs = []
    for a in aligned:
        # Rolling mean / rolling std
        rolling_mean = a.rolling(window=lookback, min_periods=1).mean()
        rolling_std = a.rolling(window=lookback, min_periods=1).std()
        ir = rolling_mean / (rolling_std + 1e-10)
        irs.append(ir.values)

    # Stack IRs: (T, N, K)
    ir_stack = np.stack(irs, axis=-1)

    if mode == "algo1":
        # Simple IR weights: softmax of IR
        # Avoid overflow by subtracting max
        ir_max = np.nanmax(ir_stack, axis=-1, keepdims=True)
        ir_shifted = ir_stack - np.where(np.isnan(ir_max), 0, ir_max)
        exp_ir = np.exp(np.clip(ir_shifted, -10, 10))
        exp_ir = np.where(np.isnan(ir_stack), 0, exp_ir)
        weights = exp_ir / (np.nansum(exp_ir, axis=-1, keepdims=True) + 1e-10)
    elif mode == "algo2":
        # Volatility-adjusted: IR / volatility
        vol_stack = np.stack([a.rolling(window=lookback, min_periods=1).std().values for a in aligned], axis=-1)
        adjusted_ir = ir_stack / (vol_stack + 1e-10)
        ir_max = np.nanmax(adjusted_ir, axis=-1, keepdims=True)
        ir_shifted = adjusted_ir - np.where(np.isnan(ir_max), 0, ir_max)
        exp_ir = np.exp(np.clip(ir_shifted, -10, 10))
        exp_ir = np.where(np.isnan(adjusted_ir), 0, exp_ir)
        weights = exp_ir / (np.nansum(exp_ir, axis=-1, keepdims=True) + 1e-10)
    elif mode == "algo3":
        # Decayed IR: exponential decay
        decay = 0.94
        decay_weights = np.array([decay ** i for i in range(lookback)][::-1])
        decay_weights = decay_weights / decay_weights.sum()
        # Use simple IR weights for now (full decay impl needs convolution)
        ir_max = np.nanmax(ir_stack, axis=-1, keepdims=True)
        ir_shifted = ir_stack - np.where(np.isnan(ir_max), 0, ir_max)
        exp_ir = np.exp(np.clip(ir_shifted, -10, 10))
        exp_ir = np.where(np.isnan(ir_stack), 0, exp_ir)
        weights = exp_ir / (np.nansum(exp_ir, axis=-1, keepdims=True) + 1e-10)
    else:
        # Default: equal weights
        weights = np.ones((T, N, K)) / K

    # Weighted sum
    alpha_stack = np.stack([a.values for a in aligned], axis=-1)
    result = np.nansum(alpha_stack * weights, axis=-1)

    return pd.DataFrame(result, index=common_index, columns=common_columns)
