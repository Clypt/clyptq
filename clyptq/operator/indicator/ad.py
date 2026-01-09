"""Accumulation/Distribution indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema


def ad(high, low, close, volume):
    """Accumulation/Distribution Line.

    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    A/D = cumsum(CLV * Volume)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume

    Returns:
        A/D Line values
    """
    hl_range = operator.sub(high, low)
    clv = operator.div(
        operator.sub(
            operator.sub(close, low),
            operator.sub(high, close)
        ),
        hl_range
    )

    return operator.ts_cumsum(operator.mul(clv, volume))


def ad_oscillator(high, low, close, volume, fast_window: int = 3, slow_window: int = 10):
    """Accumulation/Distribution Oscillator.

    Chaikin A/D Oscillator = EMA(A/D, fast) - EMA(A/D, slow)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        fast_window: Fast EMA period
        slow_window: Slow EMA period

    Returns:
        A/D Oscillator values
    """
    ad_line = ad(high, low, close, volume)
    return operator.sub(ema(ad_line, fast_window), ema(ad_line, slow_window))
