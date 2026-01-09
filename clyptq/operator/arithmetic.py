"""Arithmetic operators.

Basic math operations with proper handling of edge cases
(division by zero, negative log, etc.).

All functions are designed to work with pd.Series, pd.DataFrame,
and scalar values.

Expr Support:
- All functions accept both pandas data and Expr
- When Expr is passed, returns new Expr (lazy evaluation)
- When pandas is passed, executes immediately (backward compatible)
"""

from typing import Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clyptq.operator.base import Expr

DataType = Union[pd.Series, pd.DataFrame, "Expr"]


def _is_expr(data) -> bool:
    """Check if data is an Expr instance."""
    from clyptq.operator.base import Expr
    return isinstance(data, Expr)


def _ensure_expr(data):
    """Convert data to Expr if not already."""
    from clyptq.operator.base import Expr
    if isinstance(data, Expr):
        return data
    elif isinstance(data, (int, float, np.integer, np.floating)):
        return Expr.const(float(data))
    elif isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        return Expr.from_data(data)
    else:
        return Expr.const(float(data))


def add(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise addition.

    Args:
        x: First operand
        y: Second operand (DataType or scalar)
        lazy: If True, always return Expr

    Returns:
        x + y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.ADD, inputs=[x_expr, y_expr])
    if isinstance(y, (int, float)):
        return x + y
    return x + y


def sub(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise subtraction.

    Args:
        x: First operand
        y: Second operand (DataType or scalar)
        lazy: If True, always return Expr

    Returns:
        x - y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.SUB, inputs=[x_expr, y_expr])
    if isinstance(y, (int, float)):
        return x - y
    return x - y


def mul(x: Union[DataType, float], y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise multiplication.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        x * y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.MUL, inputs=[x_expr, y_expr])
    if isinstance(y, (int, float)):
        return x * y
    if isinstance(x, (int, float)):
        return y * x
    # DataFrame * Series: broadcast along axis
    return x * y


def div(x: Union[DataType, float], y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise division with zero handling.

    Division by zero returns 0 (not inf or nan).

    Args:
        x: Numerator
        y: Denominator
        lazy: If True, always return Expr

    Returns:
        x / y, with 0 where y == 0 (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.DIV, inputs=[x_expr, y_expr])
    if isinstance(y, (int, float)):
        if abs(y) < 1e-10:
            return x * 0
        return x / y

    if isinstance(x, (int, float)):
        result = x / y.replace(0, np.nan)
    else:
        result = x.div(y.replace(0, np.nan))

    return result.replace([np.inf, -np.inf], 0).fillna(0)


def pow(x: DataType, exp: float, *, lazy: bool = False) -> DataType:
    """Element-wise power/exponentiation.

    Args:
        x: Base
        exp: Exponent
        lazy: If True, always return Expr

    Returns:
        x^exp, with inf values replaced by 0 (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or lazy:
        x_expr = _ensure_expr(x)
        exp_expr = _ensure_expr(exp)
        return Expr(OpCode.POW, inputs=[x_expr, exp_expr])
    result = np.power(x, exp)
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.replace([np.inf, -np.inf], 0)
    return np.where(np.isinf(result), 0, result)


def power(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise power (alias for pow with variable exponent).

    Args:
        x: Base
        y: Exponent (can be DataType)
        lazy: If True, always return Expr

    Returns:
        x^y, with inf values replaced by 0 (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.POW, inputs=[x_expr, y_expr])
    result = np.power(x, y)
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.replace([np.inf, -np.inf], 0)
    return np.where(np.isinf(result), 0, result)


def log(data: DataType, *, lazy: bool = False) -> DataType:
    """Natural logarithm with zero/negative handling.

    log(0) and log(negative) return 0.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        ln(data), with 0 for invalid inputs (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.LOG, inputs=[data_expr])
    return np.log(data.replace(0, np.nan)).fillna(0)


def abs(data: DataType, *, lazy: bool = False) -> DataType:
    """Absolute value.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        |data| (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.ABS, inputs=[data_expr])
    if isinstance(data, pd.DataFrame):
        return data.abs()
    return np.abs(data)


def sqrt(data: DataType, *, lazy: bool = False) -> DataType:
    """Square root with negative handling.

    sqrt(negative) returns 0.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        sqrt(data), with 0 for negative inputs (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.SQRT, inputs=[data_expr])
    if isinstance(data, pd.DataFrame):
        return np.sqrt(data.clip(lower=0))
    return np.sqrt(np.maximum(data, 0))


def neg(data: DataType, *, lazy: bool = False) -> DataType:
    """Negation (sign flip).

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        -data (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.NEG, inputs=[data_expr])
    return -data


def sign(data: DataType, *, lazy: bool = False) -> DataType:
    """Sign function.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        -1 for negative, 0 for zero, +1 for positive (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.SIGN, inputs=[data_expr])
    return np.sign(data)


def min(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise minimum.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        min(x, y) element-wise (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.ELEM_MIN, inputs=[x_expr, y_expr])
    return np.minimum(x, y)


def max(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Element-wise maximum.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        max(x, y) element-wise (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.ELEM_MAX, inputs=[x_expr, y_expr])
    return np.maximum(x, y)


def sequence(n: int) -> np.ndarray:
    """Generate sequence from 1 to n.

    Args:
        n: Length of sequence

    Returns:
        Array [1, 2, 3, ..., n]
    """
    return np.arange(1, n + 1)


def twise_a_scale(data: DataType, scale_val: float = 1.0, *, lazy: bool = False) -> DataType:
    """Time-wise scaling (standardization along time axis).

    For DataFrame: standardize each row (across symbols at each timestamp)
    For Series: standardize entire series

    Args:
        data: Input data
        scale_val: Scale factor to apply after standardization
        lazy: If True, always return Expr

    Returns:
        Standardized data * scale_val (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        # Implement as zscore * scale_val
        zscore_expr = Expr(OpCode.CS_ZSCORE, kwargs={"axis": 1}, inputs=[data_expr])
        scale_expr = _ensure_expr(scale_val)
        return Expr(OpCode.MUL, inputs=[zscore_expr, scale_expr])
    if isinstance(data, pd.DataFrame):
        mean = data.mean(axis=1)
        std = data.std(axis=1).replace(0, np.nan)
        return data.sub(mean, axis=0).div(std, axis=0) * scale_val
    std = data.std()
    if std == 0 or pd.isna(std):
        return data * 0
    return ((data - data.mean()) / std) * scale_val


def outer_mul(row_series: DataType, col_series: DataType, *, lazy: bool = False) -> DataType:
    """Outer product multiplication: row_series[:, None] * col_series[None, :]

    Creates a DataFrame where each row is col_series scaled by corresponding row_series value.

    Args:
        row_series: Series with time index (T,)
        col_series: Series with symbol index (N,)
        lazy: If True, always return Expr

    Returns:
        DataFrame (T x N) = row_series outer-multiplied with col_series
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(row_series) or _is_expr(col_series) or lazy:
        r_expr = _ensure_expr(row_series)
        c_expr = _ensure_expr(col_series)
        return Expr(OpCode.OUTER_MUL, inputs=[r_expr, c_expr])

    # row_series: (T,) with time index
    # col_series: (N,) with symbol index
    # result: (T x N) DataFrame
    result = pd.DataFrame(
        np.outer(row_series.values, col_series.values),
        index=row_series.index,
        columns=col_series.index,
    )
    return result
