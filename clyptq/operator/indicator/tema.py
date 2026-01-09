"""Triple Exponential Moving Average indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema


def tema(data, span: int):
    """Triple Exponential Moving Average.

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    Further reduces lag compared to DEMA.

    Args:
        data: Time-series data
        span: EMA span

    Returns:
        Triple EMA
    """
    ema1 = ema(data, span)
    ema2 = ema(ema1, span)
    ema3 = ema(ema2, span)
    return operator.add(
        operator.sub(operator.mul(3, ema1), operator.mul(3, ema2)),
        ema3
    )
