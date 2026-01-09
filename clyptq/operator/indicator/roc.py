"""Rate of Change indicator."""

from clyptq import operator


def roc(close, window: int = 12):
    """Rate of Change.

    ROC = ((Close - Close[n]) / Close[n]) * 100

    Args:
        close: Close prices
        window: Lookback period (default 12)

    Returns:
        ROC as percentage
    """
    prev_close = operator.ts_delay(close, window)
    return operator.mul(
        operator.div(operator.sub(close, prev_close), prev_close),
        100
    )


def momentum(close, window: int = 10):
    """Momentum indicator.

    Momentum = Close - Close[n]

    Args:
        close: Close prices
        window: Lookback period (default 10)

    Returns:
        Momentum values (price difference)
    """
    return operator.sub(close, operator.ts_delay(close, window))
