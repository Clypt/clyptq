"""Simple Moving Average indicator."""

from clyptq import operator


def sma(data, window: int):
    """Simple Moving Average.

    Args:
        data: Time-series data
        window: Averaging window size

    Returns:
        Rolling mean
    """
    return operator.ts_mean(data, window)


def ma(data, window: int):
    """Moving Average (alias for SMA).

    Args:
        data: Time-series data
        window: Averaging window size

    Returns:
        Rolling mean
    """
    return sma(data, window)
