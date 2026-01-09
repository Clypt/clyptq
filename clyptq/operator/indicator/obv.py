"""On-Balance Volume indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema


def obv(close, volume):
    """On-Balance Volume.

    OBV = cumsum(volume * sign(close.diff()))

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        OBV values
    """
    direction = operator.sign(operator.ts_delta(close, 1))
    return operator.ts_cumsum(operator.mul(volume, direction))


def obv_ema(close, volume, span: int = 20):
    """On-Balance Volume with EMA smoothing.

    Args:
        close: Close prices
        volume: Trading volume
        span: EMA span for smoothing

    Returns:
        Smoothed OBV values
    """
    obv_val = obv(close, volume)
    return ema(obv_val, span)
