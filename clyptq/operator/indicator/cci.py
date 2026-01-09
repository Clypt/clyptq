"""Commodity Channel Index indicator."""

from clyptq import operator
from clyptq.operator.indicator.sma import sma


def cci(high, low, close, window: int = 20, constant: float = 0.015):
    """Commodity Channel Index.

    CCI = (Typical Price - SMA) / (constant * Mean Deviation)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period (default 20)
        constant: Lambert's constant (default 0.015)

    Returns:
        CCI values
    """
    # Typical Price = (H + L + C) / 3
    typical_price = operator.div(
        operator.add(operator.add(high, low), close),
        3
    )
    tp_sma = sma(typical_price, window)

    # Mean Absolute Deviation from SMA
    # Note: We approximate using ts_std * 0.8 (for normal distribution MAD ≈ 0.8 * std)
    # For exact MAD, would need a custom operator
    mean_dev = operator.mul(operator.ts_std(typical_price, window), 0.7979)

    cci_val = operator.div(
        operator.sub(typical_price, tp_sma),
        operator.mul(constant, mean_dev)
    )

    return cci_val
