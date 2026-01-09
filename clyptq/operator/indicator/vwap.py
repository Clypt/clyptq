"""Volume Weighted Average Price indicator."""

from clyptq import operator


def vwap(high, low, close, volume):
    """Volume Weighted Average Price.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume

    Returns:
        Cumulative VWAP
    """
    # typical_price = (high + low + close) / 3
    typical_price = operator.div(
        operator.add(operator.add(high, low), close),
        3
    )

    # (typical_price * volume).cumsum() / volume.cumsum()
    tp_vol = operator.mul(typical_price, volume)

    return operator.div(
        operator.ts_cumsum(tp_vol),
        operator.ts_cumsum(volume)
    )


def vwap_rolling(high, low, close, volume, window: int = 20):
    """Rolling Volume Weighted Average Price.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Rolling window size

    Returns:
        Rolling VWAP
    """
    # typical_price = (high + low + close) / 3
    typical_price = operator.div(
        operator.add(operator.add(high, low), close),
        3
    )

    # tp_vol = typical_price * volume
    tp_vol = operator.mul(typical_price, volume)

    return operator.div(
        operator.ts_sum(tp_vol, window),
        operator.ts_sum(volume, window)
    )
