"""Comparison and conditional operators.

Boolean operations for conditional logic in alpha construction.

Expr Support:
- All functions accept both pandas data and Expr
- When Expr is passed, returns new Expr (lazy evaluation)
- When pandas is passed, executes immediately (backward compatible)
"""

from typing import Union, TYPE_CHECKING
import pandas as pd
import numpy as np

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


def lt(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Less than comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x < y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.LT, inputs=[x_expr, y_expr])
    return x < y


def gt(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Greater than comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x > y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.GT, inputs=[x_expr, y_expr])
    return x > y


def le(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Less than or equal comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x <= y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.LE, inputs=[x_expr, y_expr])
    return x <= y


def ge(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Greater than or equal comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x >= y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.GE, inputs=[x_expr, y_expr])
    return x >= y


def eq(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Equality comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x == y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.EQ, inputs=[x_expr, y_expr])
    return x == y


def ne(x: DataType, y: Union[DataType, float], *, lazy: bool = False) -> DataType:
    """Not equal comparison.

    Args:
        x: First operand
        y: Second operand
        lazy: If True, always return Expr

    Returns:
        Boolean mask where x != y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.NE, inputs=[x_expr, y_expr])
    return x != y


def or_(x: DataType, y: DataType, *, lazy: bool = False) -> DataType:
    """Logical OR.

    Args:
        x: First boolean operand
        y: Second boolean operand
        lazy: If True, always return Expr

    Returns:
        x OR y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.OR, inputs=[x_expr, y_expr])
    return x | y


def and_(x: DataType, y: DataType, *, lazy: bool = False) -> DataType:
    """Logical AND.

    Args:
        x: First boolean operand
        y: Second boolean operand
        lazy: If True, always return Expr

    Returns:
        x AND y (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = _ensure_expr(x)
        y_expr = _ensure_expr(y)
        return Expr(OpCode.AND, inputs=[x_expr, y_expr])
    return x & y


def condition(
    cond: DataType,
    true_val: Union[DataType, float],
    false_val: Union[DataType, float],
    *,
    lazy: bool = False,
) -> DataType:
    """Conditional selection (if-then-else).

    Args:
        cond: Boolean condition
        true_val: Value when condition is True
        false_val: Value when condition is False
        lazy: If True, always return Expr

    Returns:
        true_val where cond is True, false_val otherwise (or Expr if input is Expr)

    Example:
        >>> # Set negative returns to zero
        >>> condition(returns < 0, 0, returns)

        >>> # Use volatility when returns are negative, else use close
        >>> condition(returns < 0, volatility, close)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(cond) or _is_expr(true_val) or _is_expr(false_val) or lazy:
        cond_expr = _ensure_expr(cond)
        true_expr = _ensure_expr(true_val)
        false_expr = _ensure_expr(false_val)
        # Use CONDITION op with true_val first, then apply where
        return Expr(OpCode.CONDITION, kwargs={"keep_self": True}, inputs=[true_expr, cond_expr, false_expr])

    if isinstance(cond, pd.DataFrame):
        if isinstance(true_val, (int, float)):
            true_val = cond * 0 + true_val
        if isinstance(false_val, (int, float)):
            false_val = cond * 0 + false_val
    else:
        if isinstance(true_val, (int, float)):
            true_val = pd.Series(true_val, index=cond.index)
        if isinstance(false_val, (int, float)):
            false_val = pd.Series(false_val, index=cond.index)

    return true_val.where(cond, false_val)


# Aliases for readability
logical_and = and_
logical_or = or_


def where(
    cond: DataType,
    x: Union[DataType, float],
    y: Union[DataType, float],
    *,
    lazy: bool = False,
) -> DataType:
    """Where condition is True, use x; otherwise use y.

    Alias for condition() with clearer semantics for masking.

    Args:
        cond: Boolean condition
        x: Value when condition is True
        y: Value when condition is False
        lazy: If True, always return Expr

    Returns:
        x where cond is True, y otherwise
    """
    return condition(cond, x, y, lazy=lazy)


def notna(data: DataType, *, lazy: bool = False) -> DataType:
    """Check for non-NaN values.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        Boolean mask where values are not NaN
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.NOTNA, inputs=[data_expr])

    if isinstance(data, pd.DataFrame):
        return data.notna()
    elif isinstance(data, pd.Series):
        return data.notna()
    else:
        return ~np.isnan(data)


def isna(data: DataType, *, lazy: bool = False) -> DataType:
    """Check for NaN values.

    Args:
        data: Input data
        lazy: If True, always return Expr

    Returns:
        Boolean mask where values are NaN
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.ISNA, inputs=[data_expr])

    if isinstance(data, pd.DataFrame):
        return data.isna()
    elif isinstance(data, pd.Series):
        return data.isna()
    else:
        return np.isnan(data)


def broadcast_mask(
    data: DataType,
    include_columns,
    *,
    lazy: bool = False,
) -> DataType:
    """Create a boolean mask based on column membership.

    For each column in data, returns True if the column name is in
    include_columns set, False otherwise. Broadcasts across all rows.

    This is useful for static filtering based on metadata (e.g., strata).

    Args:
        data: Input DataFrame (T x N)
        include_columns: Set/list of column names to include
        lazy: If True, always return Expr

    Returns:
        Boolean mask (T x N) - True for columns in include_columns

    Example:
        >>> # Include only BTC and ETH columns
        >>> mask = broadcast_mask(close, {"BTC", "ETH"})
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(
            OpCode.BROADCAST_MASK,
            inputs=[data_expr],
            kwargs={"include_columns": set(include_columns)}
        )

    if isinstance(data, pd.DataFrame):
        include_set = set(include_columns) if not isinstance(include_columns, set) else include_columns
        mask = pd.DataFrame(
            {col: col in include_set for col in data.columns},
            index=data.index,
        )
        return mask
    else:
        # For Series, check if name is in include_columns
        return data.name in include_columns
