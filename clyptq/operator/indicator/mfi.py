"""Money Flow Index indicator."""

from clyptq import operator


def mfi(high, low, close, volume, window: int = 14):
    """Money Flow Index.

    MFI = 100 - (100 / (1 + Money Flow Ratio))
    where Money Flow Ratio = Positive MF / Negative MF

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Lookback period (default 14)

    Returns:
        MFI values (0-100)
    """
    # Typical Price = (H + L + C) / 3
    typical_price = operator.div(
        operator.add(operator.add(high, low), close),
        3
    )
    raw_money_flow = operator.mul(typical_price, volume)

    tp_diff = operator.ts_delta(typical_price, 1)

    # Positive/Negative money flow
    positive_mf = operator.condition(operator.gt(tp_diff, 0), raw_money_flow, 0)
    negative_mf = operator.condition(operator.lt(tp_diff, 0), raw_money_flow, 0)

    positive_mf_sum = operator.ts_sum(positive_mf, window)
    negative_mf_sum = operator.ts_sum(negative_mf, window)

    mf_ratio = operator.div(positive_mf_sum, negative_mf_sum)
    mfi_val = operator.sub(100, operator.div(100, operator.add(1, mf_ratio)))

    return mfi_val
