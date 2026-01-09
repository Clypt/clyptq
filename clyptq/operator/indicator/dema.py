"""Double Exponential Moving Average indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema


def dema(data, span: int):
    """Double Exponential Moving Average.

    DEMA = 2 * EMA(data) - EMA(EMA(data))
    Reduces lag compared to simple EMA.

    Args:
        data: Time-series data
        span: EMA span

    Returns:
        Double EMA
    """
    ema1 = ema(data, span)
    ema2 = ema(ema1, span)
    return operator.sub(operator.mul(2, ema1), ema2)
