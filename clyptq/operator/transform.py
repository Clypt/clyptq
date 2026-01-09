"""Transformational and Special operators.

This module provides data transformation and special operations.

WQ Brain Compatible Operations:
- trade_when: Conditional trading (maintain position only when condition is true)
- densify: Convert sparse data to dense data

Special Operations:
- self_corr: Autocorrelation
- vector_neut: Vector neutralization (orthogonalization)
"""

from typing import Union, Optional, TYPE_CHECKING

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
    """Ensure data is an Expr, wrapping if necessary."""
    from clyptq.operator.base import Expr
    if isinstance(data, Expr):
        return data
    return Expr.from_data(data)


# --- Transformational Operations ---

def trade_when(
    data: DataType,
    condition: DataType,
    *,
    lazy: bool = False,
) -> DataType:
    """Conditional trading operator.

    Only maintains position (non-zero) when condition is True.
    When condition is False, position becomes 0.

    Args:
        data: Input signal/position data (T x N DataFrame or Expr)
        condition: Boolean condition (T x N DataFrame or Expr)
        lazy: If True, always return Expr

    Returns:
        Data with values set to 0 where condition is False

    Example:
        >>> # Only trade when volatility is below threshold
        >>> vol_condition = volatility < 0.3
        >>> position = trade_when(alpha_signal, vol_condition)

        >>> # Only trade on positive momentum days
        >>> momentum_condition = momentum > 0
        >>> position = trade_when(weights, momentum_condition)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or _is_expr(condition) or lazy:
        data_expr = _ensure_expr(data)
        cond_expr = _ensure_expr(condition)
        return Expr(OpCode.TRADE_WHEN, inputs=[data_expr, cond_expr])

    # Immediate execution
    if isinstance(data, pd.DataFrame):
        if isinstance(condition, pd.DataFrame):
            return data.where(condition.astype(bool), 0)
        else:
            return data.where(condition, 0)
    else:
        return data.where(condition, 0)


def densify(
    data: DataType,
    method: str = "ffill",
    *,
    lazy: bool = False,
) -> DataType:
    """Convert sparse data to dense by filling missing values.

    Fills NaN values to create dense (continuous) data.
    Useful for converting sparse signals to continuous positions.

    Args:
        data: Sparse input data (T x N DataFrame or Expr)
        method: Filling method ("ffill", "bfill", "interpolate")
        lazy: If True, always return Expr

    Returns:
        Dense data with NaN filled

    Example:
        >>> # Forward fill sparse signals
        >>> dense_signal = densify(sparse_alpha, method="ffill")

        >>> # Interpolate missing values
        >>> smooth_signal = densify(sparse_data, method="interpolate")
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.DENSIFY, kwargs={"method": method}, inputs=[data_expr])

    # Immediate execution
    if isinstance(data, pd.DataFrame):
        if method == "ffill":
            return data.ffill()
        elif method == "bfill":
            return data.bfill()
        elif method == "interpolate":
            return data.interpolate()
        else:
            return data.ffill()
    else:
        if method == "ffill":
            return data.ffill()
        elif method == "bfill":
            return data.bfill()
        elif method == "interpolate":
            return data.interpolate()
        else:
            return data.ffill()


# --- Special Operations ---

def self_corr(
    data: DataType,
    lag: int = 1,
    window: int = 20,
    *,
    lazy: bool = False,
) -> DataType:
    """Rolling autocorrelation with specified lag.

    Computes rolling correlation between data and its lagged version.
    Useful for detecting persistence or mean-reversion patterns.

    Args:
        data: Time-series data (T x N DataFrame or Expr)
        lag: Number of periods to lag (default: 1)
        window: Rolling window for correlation calculation
        lazy: If True, always return Expr

    Returns:
        Rolling autocorrelation values

    Example:
        >>> # Detect persistence in returns
        >>> persistence = self_corr(returns, lag=1, window=20)

        >>> # Check for weekly seasonality
        >>> weekly_pattern = self_corr(returns, lag=5, window=60)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.SELF_CORR, kwargs={"lag": lag, "window": window}, inputs=[data_expr])

    # Immediate execution
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for col in data.columns:
            result[col] = data[col].rolling(window=window).apply(
                lambda vals: pd.Series(vals).autocorr(lag=min(lag, len(vals)-1)),
                raw=False
            )
        return result
    else:
        return data.rolling(window=window).apply(
            lambda vals: pd.Series(vals).autocorr(lag=min(lag, len(vals)-1)),
            raw=False
        )


def vector_neut(
    data: DataType,
    vector: DataType,
    *,
    lazy: bool = False,
) -> DataType:
    """Neutralize data against a vector (orthogonalization).

    Projects out the component of data along the vector direction.
    Result is orthogonal to the vector at each timestamp.

    Math: result = data - (data·vector / vector·vector) * vector

    Args:
        data: Input data to neutralize (T x N DataFrame or Expr)
        vector: Vector to neutralize against (T x N DataFrame or Expr)
        lazy: If True, always return Expr

    Returns:
        Data neutralized against vector

    Example:
        >>> # Neutralize alpha against market returns
        >>> market_neutral = vector_neut(alpha, market_returns)

        >>> # Remove sector exposure
        >>> sector_neutral = vector_neut(signal, sector_factor)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or _is_expr(vector) or lazy:
        data_expr = _ensure_expr(data)
        vector_expr = _ensure_expr(vector)
        return Expr(OpCode.VECTOR_NEUT, inputs=[data_expr, vector_expr])

    # Immediate execution
    def neutralize_against(d, v):
        if isinstance(d, pd.DataFrame):
            result = d.copy()
            for idx in d.index:
                row = d.loc[idx].values
                vec = v.loc[idx].values if isinstance(v, pd.DataFrame) else v
                # Orthogonalize: x - (x·v / v·v) * v
                dot_xv = np.nansum(row * vec)
                dot_vv = np.nansum(vec * vec)
                if dot_vv != 0:
                    result.loc[idx] = row - (dot_xv / dot_vv) * vec
            return result
        else:
            vec = vector.values if hasattr(vector, 'values') else vector
            dot_xv = np.nansum(d.values * vec)
            dot_vv = np.nansum(vec ** 2)
            if dot_vv != 0:
                return d - (dot_xv / dot_vv) * vector
            return d

    return neutralize_against(data, vector)


def purify(
    data: DataType,
    factors: list,
    *,
    lazy: bool = False,
) -> DataType:
    """Multi-factor neutralization (purification).

    Orthogonalizes data against multiple factor vectors sequentially.
    Result has zero exposure to all provided factors.

    Args:
        data: Input data to neutralize (T x N DataFrame or Expr)
        factors: List of factor vectors to neutralize against
        lazy: If True, always return Expr

    Returns:
        Data purified against all factors

    Example:
        >>> # Remove market and sector exposure
        >>> pure_alpha = purify(alpha, [market_returns, sector_factor])

        >>> # Multi-factor neutralization
        >>> purified = purify(signal, [factor1, factor2, factor3])
    """
    result = data
    for factor in factors:
        result = vector_neut(result, factor, lazy=lazy)
    return result


def clip_extreme(
    data: DataType,
    n_std: float = 3.0,
    *,
    lazy: bool = False,
) -> DataType:
    """Clip extreme values based on standard deviations.

    Clips values beyond n_std standard deviations from the mean.
    Applied cross-sectionally at each timestamp.

    Args:
        data: Input data (T x N DataFrame or Expr)
        n_std: Number of standard deviations for clipping threshold
        lazy: If True, always return Expr

    Returns:
        Data with extreme values clipped

    Example:
        >>> # Clip outliers beyond 3 std
        >>> clean_signal = clip_extreme(alpha, n_std=3.0)
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = _ensure_expr(data)
        return Expr(OpCode.CLIP_EXTREME, kwargs={"n_std": n_std}, inputs=[data_expr])

    # Immediate execution
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for timestamp in data.index:
            row = data.loc[timestamp]
            mean = row.mean()
            std = row.std()
            if std > 0:
                lower = mean - n_std * std
                upper = mean + n_std * std
                result.loc[timestamp] = row.clip(lower=lower, upper=upper)
        return result
    else:
        mean = data.mean()
        std = data.std()
        if std > 0:
            lower = mean - n_std * std
            upper = mean + n_std * std
            return data.clip(lower=lower, upper=upper)
        return data
