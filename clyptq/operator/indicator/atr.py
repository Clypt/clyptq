"""Average True Range indicator."""

from clyptq import operator
from clyptq.operator.indicator.sma import sma


def true_range(high, low, close):
    """True Range calculation.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        True Range values
    """
    prev_close = operator.ts_delay(close, 1)

    tr1 = operator.sub(high, low)
    tr2 = operator.abs(operator.sub(high, prev_close))
    tr3 = operator.abs(operator.sub(low, prev_close))

    # max of three values
    return operator.elem_max(tr1, operator.elem_max(tr2, tr3))


def atr(high, low, close, window: int = 14):
    """Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Averaging window

    Returns:
        ATR (measure of volatility)
    """
    tr = true_range(high, low, close)
    return sma(tr, window)


def natr(high, low, close, window: int = 14):
    """Normalized Average True Range.

    NATR = (ATR / Close) * 100

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Averaging window

    Returns:
        Normalized ATR as percentage
    """
    atr_val = atr(high, low, close, window)
    return operator.mul(operator.div(atr_val, close), 100)
