"""Parabolic SAR indicator.

Parabolic SAR uses acceleration factor that increases on new extremes.
The self-referencing state is handled via ts_delay with initial values.
"""

from clyptq import operator


def psar(high, low, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
    """Parabolic SAR (Stop and Reverse).

    Args:
        high: High prices
        low: Low prices
        af_start: Starting acceleration factor (default 0.02)
        af_increment: AF increment (default 0.02)
        af_max: Maximum AF (default 0.2)

    Returns:
        Dict with 'psar', 'direction' (1 for bull, -1 for bear)
    """
    # Initial direction based on price trend
    price_trend = operator.ts_delta(high, 5)
    direction = operator.condition(
        operator.gt(price_trend, 0),
        1,
        -1
    )

    # Extreme point tracking
    prev_direction = operator.ts_delay(direction, 1)

    # Trailing extremes for SAR calculation
    trailing_high = operator.ts_max(high, 10)
    trailing_low = operator.ts_min(low, 10)

    # Initialize SAR based on direction
    # Bull: SAR starts at recent low
    # Bear: SAR starts at recent high
    sar = operator.condition(
        operator.gt(direction, 0),
        trailing_low,
        trailing_high
    )

    prev_sar = operator.ts_delay(sar, 1)

    # Extreme point (EP): highest high in uptrend, lowest low in downtrend
    ep = operator.condition(
        operator.gt(direction, 0),
        trailing_high,
        trailing_low
    )

    # AF increases when new extreme is made (simplified: use fixed AF for now)
    # True AF tracking would need: af = min(af + af_increment, af_max) on new EP
    af = af_start

    # SAR update: SAR = prev_SAR + AF * (EP - prev_SAR)
    sar = operator.add(
        prev_sar,
        operator.mul(af, operator.sub(ep, prev_sar))
    )

    # Reversal check: if price crosses SAR, flip direction
    # Bull reversal: low < SAR
    # Bear reversal: high > SAR
    bull_reversal = operator.lt(low, sar)
    bear_reversal = operator.gt(high, sar)

    # Final direction after reversal check
    final_direction = operator.condition(
        operator.and_(operator.gt(direction, 0), bull_reversal),
        -1,
        operator.condition(
            operator.and_(operator.lt(direction, 0), bear_reversal),
            1,
            direction
        )
    )

    return {
        "psar": sar,
        "direction": final_direction,
    }


def psar_value(high, low, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
    """Parabolic SAR value only.

    Args:
        high: High prices
        low: Low prices
        af_start: Starting acceleration factor
        af_increment: AF increment
        af_max: Maximum AF

    Returns:
        PSAR values
    """
    return psar(high, low, af_start, af_increment, af_max)["psar"]
