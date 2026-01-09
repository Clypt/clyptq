"""Chaikin Money Flow indicator."""

from clyptq import operator


def cmf(high, low, close, volume, window: int = 20):
    """Chaikin Money Flow.

    CMF = sum(MFV, n) / sum(Volume, n)
    where MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Lookback period (default 20)

    Returns:
        CMF values (-1 to 1)
    """
    hl_range = operator.sub(high, low)
    clv = operator.div(
        operator.sub(
            operator.sub(close, low),
            operator.sub(high, close)
        ),
        hl_range
    )

    mfv = operator.mul(clv, volume)

    cmf_val = operator.div(
        operator.ts_sum(mfv, window),
        operator.ts_sum(volume, window)
    )

    return cmf_val
