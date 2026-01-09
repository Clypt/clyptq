"""Bollinger Bands indicator."""

from clyptq import operator
from clyptq.operator.indicator.sma import sma


def bollinger_bands(close, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands.

    Args:
        close: Close prices
        window: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Dict with 'upper', 'middle', 'lower', 'bandwidth', 'percent_b'
    """
    middle = sma(close, window)
    std = operator.ts_std(close, window)

    upper = operator.add(middle, operator.mul(std, num_std))
    lower = operator.sub(middle, operator.mul(std, num_std))

    bandwidth = operator.mul(
        operator.div(operator.sub(upper, lower), middle),
        100
    )
    percent_b = operator.div(
        operator.sub(close, lower),
        operator.sub(upper, lower)
    )

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    }


def bollinger_upper(close, window: int = 20, num_std: float = 2.0):
    """Bollinger Band upper line.

    Args:
        close: Close prices
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        Upper band values
    """
    middle = sma(close, window)
    std = operator.ts_std(close, window)
    return operator.add(middle, operator.mul(std, num_std))


def bollinger_lower(close, window: int = 20, num_std: float = 2.0):
    """Bollinger Band lower line.

    Args:
        close: Close prices
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        Lower band values
    """
    middle = sma(close, window)
    std = operator.ts_std(close, window)
    return operator.sub(middle, operator.mul(std, num_std))


def bollinger_bandwidth(close, window: int = 20, num_std: float = 2.0):
    """Bollinger Bandwidth.

    Bandwidth = (Upper - Lower) / Middle * 100

    Args:
        close: Close prices
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        Bandwidth as percentage
    """
    bands = bollinger_bands(close, window, num_std)
    return bands["bandwidth"]


def bollinger_percent_b(close, window: int = 20, num_std: float = 2.0):
    """Bollinger %B.

    %B = (Price - Lower) / (Upper - Lower)

    Args:
        close: Close prices
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        %B values (0-1 typically)
    """
    bands = bollinger_bands(close, window, num_std)
    return bands["percent_b"]
