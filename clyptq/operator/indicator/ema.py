"""Exponential Moving Average indicator."""

from typing import Optional


def ema(data, span: int, alpha: Optional[float] = None):
    """Exponential Moving Average.

    Args:
        data: Time-series data
        span: Span for EMA calculation (decay factor = 2/(span+1))
        alpha: Optional explicit smoothing factor (0 < alpha <= 1)

    Returns:
        Exponentially weighted moving average
    """
    # EMA uses ewm which is a method on data (Expr/DataFrame both support it)
    if alpha is not None:
        return data.ewm(alpha=alpha, adjust=False).mean()
    return data.ewm(span=span, adjust=False).mean()
