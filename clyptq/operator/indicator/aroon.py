"""Aroon indicator."""

from clyptq import operator


def aroon(high, low, window: int = 25):
    """Aroon indicator.

    Aroon Up = ((window - days since highest high) / window) * 100
    Aroon Down = ((window - days since lowest low) / window) * 100
    Aroon Oscillator = Aroon Up - Aroon Down

    Args:
        high: High prices
        low: Low prices
        window: Lookback period (default 25)

    Returns:
        Dict with 'aroon_up', 'aroon_down', 'oscillator'
    """
    # ts_argmax returns position of max within window (0 to window-1)
    # Aroon Up = (window - (window - 1 - argmax)) / window * 100
    #          = (argmax + 1) / window * 100
    days_since_high = operator.sub(window, operator.ts_argmax(high, window + 1))
    days_since_low = operator.sub(window, operator.ts_argmin(low, window + 1))

    up = operator.mul(operator.div(operator.sub(window, days_since_high), window), 100)
    down = operator.mul(operator.div(operator.sub(window, days_since_low), window), 100)
    oscillator = operator.sub(up, down)

    return {
        "aroon_up": up,
        "aroon_down": down,
        "oscillator": oscillator,
    }


def aroon_up(high, window: int = 25):
    """Aroon Up only.

    Args:
        high: High prices
        window: Lookback period

    Returns:
        Aroon Up values
    """
    days_since_high = operator.sub(window, operator.ts_argmax(high, window + 1))
    return operator.mul(operator.div(operator.sub(window, days_since_high), window), 100)


def aroon_down(low, window: int = 25):
    """Aroon Down only.

    Args:
        low: Low prices
        window: Lookback period

    Returns:
        Aroon Down values
    """
    days_since_low = operator.sub(window, operator.ts_argmin(low, window + 1))
    return operator.mul(operator.div(operator.sub(window, days_since_low), window), 100)


def aroon_oscillator(high, low, window: int = 25):
    """Aroon Oscillator only.

    Args:
        high: High prices
        low: Low prices
        window: Lookback period

    Returns:
        Aroon Oscillator values
    """
    return operator.sub(aroon_up(high, window), aroon_down(low, window))
