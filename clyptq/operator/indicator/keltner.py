"""Keltner Channel indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema
from clyptq.operator.indicator.atr import atr


def keltner_channel(
    high,
    low,
    close,
    ema_window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0
):
    """Keltner Channel.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_window: EMA period for middle line (default 20)
        atr_window: ATR period (default 10)
        atr_multiplier: ATR multiplier for bands (default 2.0)

    Returns:
        Dict with 'upper', 'middle', 'lower' values
    """
    middle = ema(close, ema_window)
    atr_val = atr(high, low, close, atr_window)

    upper = operator.add(middle, operator.mul(atr_val, atr_multiplier))
    lower = operator.sub(middle, operator.mul(atr_val, atr_multiplier))

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def keltner_upper(
    high,
    low,
    close,
    ema_window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0
):
    """Keltner Channel upper line.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_window: EMA period
        atr_window: ATR period
        atr_multiplier: ATR multiplier

    Returns:
        Upper channel values
    """
    return keltner_channel(high, low, close, ema_window, atr_window, atr_multiplier)["upper"]


def keltner_lower(
    high,
    low,
    close,
    ema_window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0
):
    """Keltner Channel lower line.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_window: EMA period
        atr_window: ATR period
        atr_multiplier: ATR multiplier

    Returns:
        Lower channel values
    """
    return keltner_channel(high, low, close, ema_window, atr_window, atr_multiplier)["lower"]
