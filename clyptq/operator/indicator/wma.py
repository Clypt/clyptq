"""Weighted Moving Average indicator."""

from clyptq import operator


def wma(data, window: int):
    """Weighted Moving Average (linearly weighted).

    More recent values get higher weights.
    Uses linear decay: weights = [1, 2, 3, ..., window] normalized.

    Args:
        data: Time-series data
        window: Window size

    Returns:
        Linearly weighted moving average
    """
    return operator.ts_decayed_linear(data, window)
