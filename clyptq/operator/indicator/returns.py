"""Returns calculation indicators."""

from clyptq import operator


def returns(close, period: int = 1):
    """Simple returns.

    Args:
        close: Close prices
        period: Number of periods for return calculation

    Returns:
        (close[t] - close[t-period]) / close[t-period]
    """
    return operator.ts_returns(close, period=period)


def log_returns(close, period: int = 1):
    """Log returns.

    Args:
        close: Close prices
        period: Number of periods for return calculation

    Returns:
        ln(close[t] / close[t-period])
    """
    return operator.ts_log_returns(close, period=period)


def cumulative_returns(close):
    """Cumulative returns from start.

    Args:
        close: Close prices

    Returns:
        Cumulative return series
    """
    # (cumprod(1 + returns) - 1)
    returns_data = operator.ts_returns(close)
    returns_plus1 = operator.add(returns_data, 1)
    cumulative = operator.ts_cumprod(returns_plus1)
    return operator.sub(cumulative, 1)


def rolling_returns(close, window: int):
    """Rolling returns over window.

    Args:
        close: Close prices
        window: Rolling window size

    Returns:
        Rolling return over window periods
    """
    return operator.ts_returns(close, period=window)
