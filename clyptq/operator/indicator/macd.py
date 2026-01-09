"""Moving Average Convergence Divergence indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema


def macd(
    close,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
):
    """Moving Average Convergence Divergence.

    Args:
        close: Close prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Dict with 'macd', 'signal', and 'histogram' values
    """
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)

    macd_val = operator.sub(fast_ema, slow_ema)
    signal_val = ema(macd_val, signal_period)
    histogram = operator.sub(macd_val, signal_val)

    return {
        "macd": macd_val,
        "signal": signal_val,
        "histogram": histogram,
    }


def macd_line(close, fast_period: int = 12, slow_period: int = 26):
    """MACD Line only.

    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        MACD line values
    """
    return operator.sub(ema(close, fast_period), ema(close, slow_period))


def macd_signal(
    close,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
):
    """MACD Signal line only.

    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Signal line values
    """
    macd_val = macd_line(close, fast_period, slow_period)
    return ema(macd_val, signal_period)


def macd_histogram(
    close,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
):
    """MACD Histogram only.

    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Histogram values
    """
    macd_val = macd_line(close, fast_period, slow_period)
    signal_val = ema(macd_val, signal_period)
    return operator.sub(macd_val, signal_val)
