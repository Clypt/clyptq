"""Williams %R indicator."""

from clyptq import operator


def williams_r(high, low, close, window: int = 14):
    """Williams %R.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period (default 14)

    Returns:
        Williams %R values (-100 to 0)
    """
    highest_high = operator.ts_max(high, window)
    lowest_low = operator.ts_min(low, window)

    wr = operator.mul(
        operator.div(
            operator.sub(highest_high, close),
            operator.sub(highest_high, lowest_low)
        ),
        -100
    )

    return wr
