"""Donchian Channel indicator."""

from clyptq import operator


def donchian_channel(high, low, window: int = 20):
    """Donchian Channel.

    Args:
        high: High prices
        low: Low prices
        window: Lookback period (default 20)

    Returns:
        Dict with 'upper', 'middle', 'lower' values
    """
    upper = operator.ts_max(high, window)
    lower = operator.ts_min(low, window)
    middle = operator.div(operator.add(upper, lower), 2)

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def donchian_upper(high, window: int = 20):
    """Donchian Channel upper line.

    Args:
        high: High prices
        window: Lookback period

    Returns:
        Upper channel values
    """
    return operator.ts_max(high, window)


def donchian_lower(low, window: int = 20):
    """Donchian Channel lower line.

    Args:
        low: Low prices
        window: Lookback period

    Returns:
        Lower channel values
    """
    return operator.ts_min(low, window)


def donchian_middle(high, low, window: int = 20):
    """Donchian Channel middle line.

    Args:
        high: High prices
        low: Low prices
        window: Lookback period

    Returns:
        Middle channel values
    """
    upper = operator.ts_max(high, window)
    lower = operator.ts_min(low, window)
    return operator.div(operator.add(upper, lower), 2)
