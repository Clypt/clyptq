"""SuperTrend indicator.

SuperTrend uses ATR-based bands with trend-following logic.
The self-referencing state is handled via ts_delay with initial value.
"""

from clyptq import operator
from clyptq.operator.indicator.atr import atr


def supertrend(high, low, close, atr_window: int = 10, multiplier: float = 3.0):
    """SuperTrend indicator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_window: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)

    Returns:
        Dict with 'supertrend', 'direction' (-1 for down, 1 for up)
    """
    atr_val = atr(high, low, close, atr_window)
    hl2 = operator.div(operator.add(high, low), 2)

    # ATR-based bands
    upper_band = operator.add(hl2, operator.mul(multiplier, atr_val))
    lower_band = operator.sub(hl2, operator.mul(multiplier, atr_val))

    # Initial supertrend value (start with upper band - bearish assumption)
    st = upper_band

    # Self-referencing via ts_delay: compare close with previous supertrend
    # If close > prev_st: trend is up → use lower_band
    # If close < prev_st: trend is down → use upper_band
    prev_st = operator.ts_delay(st, 1)

    st = operator.condition(
        operator.gt(close, prev_st),
        lower_band,
        upper_band
    )

    # Direction: 1 for up (using lower band), -1 for down (using upper band)
    direction = operator.condition(
        operator.gt(close, prev_st),
        1,
        -1
    )

    return {
        "supertrend": st,
        "direction": direction,
    }


def supertrend_value(high, low, close, atr_window: int = 10, multiplier: float = 3.0):
    """SuperTrend line value only.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_window: ATR period
        multiplier: ATR multiplier

    Returns:
        SuperTrend line values
    """
    return supertrend(high, low, close, atr_window, multiplier)["supertrend"]


def supertrend_direction(high, low, close, atr_window: int = 10, multiplier: float = 3.0):
    """SuperTrend direction only.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_window: ATR period
        multiplier: ATR multiplier

    Returns:
        Direction values (-1 for down, 1 for up)
    """
    return supertrend(high, low, close, atr_window, multiplier)["direction"]
