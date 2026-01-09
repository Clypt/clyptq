"""Time-series operators (ts_*).

Rolling calculations along the time axis. All functions are stateless
and work with both pd.Series and pd.DataFrame.

Convention:
- ts_* prefix indicates time-series operation
- window parameter specifies rolling window size
- Operations are applied along axis=0 (time axis)

Expr Support:
- All functions accept both pandas data and Expr
- When Expr is passed, returns new Expr (lazy evaluation)
- When pandas is passed, executes immediately (backward compatible)
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


def _make_expr(op_code, data, *args, **kwargs):
    """Create Expr node from operation."""
    from clyptq.operator.base import Expr, OpCode
    if _is_expr(data):
        return Expr(op_code, args=args, kwargs=kwargs, inputs=[data])
    # Wrap raw data and return Expr
    expr = Expr.from_data(data)
    return Expr(op_code, args=args, kwargs=kwargs, inputs=[expr])


# --- Rolling Statistics ---

def ts_mean(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling mean over time window.

    Args:
        data: Time-series data (Series, DataFrame, or Expr)
        window: Rolling window size
        lazy: If True, always return Expr (for explicit lazy mode)

    Returns:
        Rolling mean with same shape as input (or Expr if input is Expr)
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_MEAN, data, window)
    return data.rolling(window).mean()


def ts_sum(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling sum over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_SUM, data, window)
    return data.rolling(window).sum()


def ts_std(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling standard deviation over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_STD, data, window)
    return data.rolling(window).std()


def ts_min(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling minimum over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_MIN, data, window)
    return data.rolling(window).min()


def ts_max(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling maximum over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_MAX, data, window)
    return data.rolling(window).max()


def ts_rank(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling rank (percentile) over time window.

    Returns value between 0 and 1 indicating position within window.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_RANK, data, window)
    return data.rolling(window).rank(pct=True)


def ts_corr(x: DataType, y: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling correlation between two time series."""
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = x if _is_expr(x) else Expr.from_data(x)
        y_expr = y if _is_expr(y) else Expr.from_data(y)
        return Expr(OpCode.TS_CORR, args=(window,), inputs=[x_expr, y_expr])
    result = x.rolling(window).corr(y)
    return result.replace([np.inf, -np.inf], 0)


def ts_cov(x: DataType, y: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling covariance between two time series."""
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(x) or _is_expr(y) or lazy:
        x_expr = x if _is_expr(x) else Expr.from_data(x)
        y_expr = y if _is_expr(y) else Expr.from_data(y)
        return Expr(OpCode.TS_COV, args=(window,), inputs=[x_expr, y_expr])
    result = x.rolling(window).cov(y)
    return result.replace([np.inf, -np.inf], 0)


def ts_zscore(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling z-score over time window.

    Returns (value - mean) / std for rolling window.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_ZSCORE, data, window)
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / std.replace(0, np.nan)


def ts_scale(data: DataType, window: int, constant: float = 0, *, lazy: bool = False) -> DataType:
    """Rolling min-max scaling to [0, 1] range over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_SCALE, data, window, constant=constant)
    min_val = data.rolling(window).min()
    max_val = data.rolling(window).max()
    range_val = max_val - min_val
    return (data - min_val) / range_val.replace(0, np.nan) + constant


def ts_quantile(data: DataType, window: int, q: float = 0.5, *, lazy: bool = False) -> DataType:
    """Rolling quantile over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_QUANTILE, data, window, q=q)
    return data.rolling(window).quantile(q)


def ts_product(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling product over time window."""
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_PRODUCT, data, window)
    return data.rolling(window).apply(np.prod, raw=True)


def ts_decayed_linear(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Linearly decayed weighted average over time window.

    Recent values get higher weights (linear decay from window to 1).
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_DECAYED_LINEAR, data, window)
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    return data.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


# --- Lag and Difference ---

def ts_delta(data: DataType, period: int = 1, *, lazy: bool = False) -> DataType:
    """Time-series difference (current - lagged value).

    Args:
        data: Time-series data
        period: Number of periods to difference
        lazy: If True, always return Expr

    Returns:
        data[t] - data[t-period]
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_DELTA, data, period)
    return data.diff(periods=period)


def ts_delay(data: DataType, period: int, *, lazy: bool = False) -> DataType:
    """Lag (shift) time series by period.

    Args:
        data: Time-series data
        period: Number of periods to shift (positive = backward)
        lazy: If True, always return Expr

    Returns:
        data[t-period]
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_DELAY, data, period)
    return data.shift(period)


def delay(data: DataType, period: int, *, lazy: bool = False) -> DataType:
    """Alias for ts_delay."""
    return ts_delay(data, period, lazy=lazy)


# --- Argmax/Argmin ---

def ts_argmax(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling index of maximum value within window.

    Returns position (0 to window-1) of maximum value.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_ARGMAX, data, window)
    if isinstance(data, pd.DataFrame):
        return data.rolling(window).apply(lambda x: np.argmax(x), raw=True)
    # Series - optimized version
    result = np.full(len(data), np.nan)
    values = data.values
    for i in range(window - 1, len(data)):
        result[i] = np.argmax(values[i - window + 1 : i + 1])
    return pd.Series(result, index=data.index)


def ts_argmin(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling index of minimum value within window.

    Returns position (0 to window-1) of minimum value.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_ARGMIN, data, window)
    if isinstance(data, pd.DataFrame):
        return data.rolling(window).apply(lambda x: np.argmin(x), raw=True)
    result = np.full(len(data), np.nan)
    values = data.values
    for i in range(window - 1, len(data)):
        result[i] = np.argmin(values[i - window + 1 : i + 1])
    return pd.Series(result, index=data.index)


# --- Regression ---

def ts_slope(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Rolling linear regression slope over time window.

    Regresses data against sequential indices [0, 1, 2, ...].
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_SLOPE, data, window)
    def calc_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return data.rolling(window).apply(calc_slope, raw=True)


def ts_regbeta(data: DataType, x: np.ndarray, *, lazy: bool = False) -> DataType:
    """Rolling regression beta coefficient.

    Args:
        data: Dependent variable (y)
        x: Independent variable array (length = window)
        lazy: If True, always return Expr

    Returns:
        Regression coefficient from y = beta * x + alpha
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.PANDAS_METHOD, kwargs={"method": "ts_regbeta", "x": x.tolist()}, inputs=[data_expr])
    window = len(x)
    return data.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0], raw=True)


def ts_residual(data: DataType, window: int, x: Optional[DataType] = None, *, lazy: bool = False) -> DataType:
    """Rolling regression residual (actual - predicted).

    Args:
        data: Dependent variable
        window: Rolling window size
        x: Independent variable (if None, uses sequential indices)
        lazy: If True, always return Expr

    Returns:
        Residual at current time point
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.PANDAS_METHOD, kwargs={"method": "ts_residual", "window": window}, inputs=[data_expr])
    if x is None:
        def calc_residual(y):
            if len(y) < 2:
                return np.nan
            x_vals = np.arange(len(y))
            coeffs = np.polyfit(x_vals, y, 1)
            fitted = np.polyval(coeffs, x_vals)
            return y[-1] - fitted[-1]
        return data.rolling(window).apply(calc_residual, raw=True)
    else:
        # Two series regression - simplified
        return data - x


def ts_rsquare(data: DataType, window: int, y: Optional[DataType] = None, *, lazy: bool = False) -> DataType:
    """Rolling R-squared (coefficient of determination).

    Measures goodness of fit against sequential indices.
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.PANDAS_METHOD, kwargs={"method": "ts_rsquare", "window": window}, inputs=[data_expr])
    def calc_rsquare(vals):
        if len(vals) < 2:
            return np.nan
        x = np.arange(len(vals))
        if np.std(vals) == 0:
            return 0
        corr = np.corrcoef(x, vals)[0, 1]
        return corr ** 2 if not np.isnan(corr) else np.nan
    return data.rolling(window).apply(calc_rsquare, raw=True)


def ts_linear_reg(data: DataType, window: int, mode: int = 0, *, lazy: bool = False) -> DataType:
    """Rolling linear regression with sequential indices.

    Args:
        data: Time-series data
        window: Rolling window size
        mode: 0 = predicted value at end of window, 1 = slope
        lazy: If True, always return Expr

    Returns:
        Regression output based on mode
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.PANDAS_METHOD, kwargs={"method": "ts_linear_reg", "window": window, "mode": mode}, inputs=[data_expr])
    def calc_reg(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        if mode == 0:
            return coeffs[0] * (len(y) - 1) + coeffs[1]
        return coeffs[0]
    return data.rolling(window).apply(calc_reg, raw=True)


# --- Days Since Extrema ---

def lowday(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Days since lowest point within rolling window.

    Returns number of days elapsed since the minimum value.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.LOWDAY, data, window)
    return data.rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)


def highday(data: DataType, window: int, *, lazy: bool = False) -> DataType:
    """Days since highest point within rolling window.

    Returns number of days elapsed since the maximum value.
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.HIGHDAY, data, window)
    return data.rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)


# --- Returns & Cumulative Operations ---

def ts_returns(data: DataType, period: int = 1, *, lazy: bool = False) -> DataType:
    """Percentage change (returns) over period.

    Replaces raw pandas .pct_change() for Expr compatibility.

    Args:
        data: Price series or DataFrame
        period: Number of periods for return calculation
        lazy: If True, always return Expr

    Returns:
        Returns: (p_t - p_{t-period}) / p_{t-period}

    Example:
        ```python
        # Instead of close.pct_change(1)
        returns = ts_returns(close, 1)

        # 20-day returns
        returns_20d = ts_returns(close, period=20)
        ```
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_RETURNS, data, period)
    return data.pct_change(periods=period)


def ts_log_returns(data: DataType, period: int = 1, *, lazy: bool = False) -> DataType:
    """Log returns over period.

    Args:
        data: Price series or DataFrame
        period: Number of periods
        lazy: If True, always return Expr

    Returns:
        Log returns: log(p_t / p_{t-period})
    """
    from clyptq.operator.base import OpCode
    if _is_expr(data) or lazy:
        return _make_expr(OpCode.TS_LOG_RETURNS, data, period)
    return np.log(data / data.shift(period))


def ts_cumsum(data: DataType, *, axis: int = 0, lazy: bool = False) -> DataType:
    """Cumulative sum along time axis.

    Args:
        data: Time-series data
        axis: Axis along which to compute (default 0 = time)
        lazy: If True, always return Expr

    Returns:
        Cumulative sum
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_CUMSUM, kwargs={"axis": axis}, inputs=[data_expr])
    return data.cumsum(axis=axis)


def ts_cumprod(data: DataType, *, axis: int = 0, lazy: bool = False) -> DataType:
    """Cumulative product along time axis.

    Useful for compounding returns.

    Args:
        data: Time-series data (typically 1 + returns)
        axis: Axis along which to compute (default 0 = time)
        lazy: If True, always return Expr

    Returns:
        Cumulative product

    Example:
        ```python
        cumulative_return = ts_cumprod(1 + ts_returns(close))
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_CUMPROD, kwargs={"axis": axis}, inputs=[data_expr])
    return data.cumprod(axis=axis)


# --- Additional Time-Series Operations (WQ Brain compatible) ---

def ts_days_from_last_change(data: DataType, *, lazy: bool = False) -> DataType:
    """Count days since last value change.

    Returns the number of periods since the value last changed.
    Useful for detecting staleness or unchanged positions.

    Args:
        data: Time-series data
        lazy: If True, always return Expr

    Returns:
        Days since last change

    Example:
        ```python
        # Detect stale prices
        staleness = ts_days_from_last_change(close)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_DAYS_FROM_LAST_CHANGE, inputs=[data_expr])

    # Immediate execution
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for col in data.columns:
            changed = data[col].diff().ne(0)
            groups = changed.cumsum()
            result[col] = data[col].groupby(groups).cumcount()
        return result
    else:
        changed = data.diff().ne(0)
        groups = changed.cumsum()
        return data.groupby(groups).cumcount()


def ts_backfill(data: DataType, limit: Optional[int] = None, *, lazy: bool = False) -> DataType:
    """Backward fill NaN values.

    Args:
        data: Time-series data with NaN values
        limit: Maximum number of consecutive NaN to fill
        lazy: If True, always return Expr

    Returns:
        Data with NaN filled backward

    Example:
        ```python
        filled = ts_backfill(sparse_data, limit=5)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_BACKFILL, kwargs={"limit": limit}, inputs=[data_expr])
    return data.bfill(limit=limit)


def ts_decay_exp_window(
    data: DataType,
    window: int = 20,
    factor: float = 0.5,
    *,
    lazy: bool = False
) -> DataType:
    """Exponential decay weighted average over window.

    More recent values get higher weights with exponential decay.

    Args:
        data: Time-series data
        window: Rolling window size
        factor: Decay factor (0 < factor < 1, smaller = faster decay)
        lazy: If True, always return Expr

    Returns:
        Exponentially decayed weighted average

    Example:
        ```python
        decayed = ts_decay_exp_window(signal, window=10, factor=0.8)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_DECAY_EXP_WINDOW, kwargs={"window": window, "factor": factor}, inputs=[data_expr])

    # Immediate execution
    weights = np.array([factor ** i for i in range(window)])[::-1]
    weights = weights / weights.sum()

    def apply_decay(series):
        return series.rolling(window, min_periods=1).apply(
            lambda vals: np.sum(vals * weights[-len(vals):] / weights[-len(vals):].sum()),
            raw=True
        )

    if isinstance(data, pd.DataFrame):
        return data.apply(apply_decay)
    return apply_decay(data)


def ts_hump(data: DataType, window: int = 3, *, lazy: bool = False) -> DataType:
    """Detect local maxima (humps) in time-series.

    Returns 1 when the value is a local maximum within the window, 0 otherwise.

    Args:
        data: Time-series data
        window: Half-window for detecting local maxima
        lazy: If True, always return Expr

    Returns:
        Binary indicator for local maxima

    Example:
        ```python
        peaks = ts_hump(close, window=5)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_HUMP, kwargs={"window": window}, inputs=[data_expr])

    # Immediate execution
    def detect_hump(series):
        result = pd.Series(0, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                result.iloc[i] = 1
        return result

    if isinstance(data, pd.DataFrame):
        return data.apply(detect_hump)
    return detect_hump(data)


def ts_jump_decay(
    data: DataType,
    threshold: float = 2.0,
    decay: float = 0.9,
    *,
    lazy: bool = False
) -> DataType:
    """Detect jumps and apply decay.

    Identifies sudden price jumps (beyond threshold * std) and applies decay.

    Args:
        data: Time-series data
        threshold: Number of standard deviations to trigger jump detection
        decay: Decay factor applied each period after jump
        lazy: If True, always return Expr

    Returns:
        Jump indicator with decay

    Example:
        ```python
        jump_signal = ts_jump_decay(close, threshold=3.0, decay=0.8)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_JUMP_DECAY, kwargs={"threshold": threshold, "decay": decay}, inputs=[data_expr])

    # Immediate execution
    def jump_decay_series(series):
        result = pd.Series(0.0, index=series.index)
        returns = series.pct_change()
        std = returns.rolling(20).std()
        jumps = (returns.abs() > threshold * std).astype(float)

        decay_val = 0
        for i in range(len(series)):
            if jumps.iloc[i] == 1:
                decay_val = 1
            else:
                decay_val *= decay
            result.iloc[i] = decay_val
        return result

    if isinstance(data, pd.DataFrame):
        return data.apply(jump_decay_series)
    return jump_decay_series(data)


def ts_step_decay(data: DataType, step: int = 5, *, lazy: bool = False) -> DataType:
    """Apply step decay pattern.

    Creates a repeating decay pattern with period `step`.

    Args:
        data: Time-series data
        step: Period of the decay pattern
        lazy: If True, always return Expr

    Returns:
        Step decay pattern

    Example:
        ```python
        pattern = ts_step_decay(signal, step=10)
        ```
    """
    from clyptq.operator.base import OpCode, Expr
    if _is_expr(data) or lazy:
        data_expr = data if _is_expr(data) else Expr.from_data(data)
        return Expr(OpCode.TS_STEP_DECAY, kwargs={"step": step}, inputs=[data_expr])

    # Immediate execution
    def step_decay_series(series):
        result = pd.Series(0.0, index=series.index)
        for i in range(len(series)):
            result.iloc[i] = 1.0 / (1 + (i % step))
        return result

    if isinstance(data, pd.DataFrame):
        return data.apply(step_decay_series)
    return step_decay_series(data)


# --- Missing Value Handling ---

def ts_fillna(
    data: DataType,
    value: Union[float, int, None] = None,
    *,
    method: Optional[str] = None,
    limit: Optional[int] = None,
    lazy: bool = False,
) -> DataType:
    """Fill missing values along time axis.

    Args:
        data: Time-series data (Series, DataFrame, or Expr)
        value: Scalar value to use for filling (mutually exclusive with method)
        method: Fill method - 'ffill' or 'pad' only (backward fill not allowed)
        limit: Maximum consecutive NaN values to fill when using method
        lazy: If True, always return Expr

    Returns:
        Data with NaN values filled

    Examples:
        # Fill with 0
        ts_fillna(data, 0)

        # Forward fill (use previous value)
        ts_fillna(data, method='ffill')

        # Forward fill with limit
        ts_fillna(data, method='ffill', limit=3)

    Note:
        Backward fill (bfill) is not supported to prevent look-ahead bias.
    """
    from clyptq.operator.base import OpCode

    # Prevent look-ahead bias
    if method in ('bfill', 'backfill'):
        raise ValueError(
            "Backward fill (bfill) causes look-ahead bias. "
            "Use ffill or fill with a constant value instead."
        )

    if _is_expr(data) or lazy:
        return _make_expr(
            OpCode.FILLNA,
            data,
            value=value,
            method=method,
            limit=limit,
        )

    # Immediate execution
    if method is not None:
        return data.fillna(method=method, limit=limit)
    return data.fillna(value)


def ts_ffill(
    data: DataType,
    limit: Optional[int] = None,
    *,
    lazy: bool = False,
) -> DataType:
    """Forward fill missing values (use previous valid value).

    Safe for backtesting - only uses past data.

    Args:
        data: Time-series data
        limit: Maximum consecutive NaN values to fill
        lazy: If True, always return Expr

    Returns:
        Data with NaN values forward-filled
    """
    from clyptq.operator.base import OpCode

    if _is_expr(data) or lazy:
        return _make_expr(OpCode.FFILL, data, limit=limit)

    return data.ffill(limit=limit)
