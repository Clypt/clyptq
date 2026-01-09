"""Average Directional Index indicator."""

from clyptq import operator
from clyptq.operator.indicator.atr import true_range
from clyptq.operator.indicator.ema import ema


def adx(high, low, close, window: int = 14):
    """Average Directional Index.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period (default 14)

    Returns:
        Dict with 'adx', 'plus_di', 'minus_di' values
    """
    # True Range
    tr = true_range(high, low, close)

    # Directional Movement
    up_move = operator.ts_delta(high, 1)
    down_move = operator.neg(operator.ts_delta(low, 1))

    plus_dm = operator.condition(
        operator.and_(operator.gt(up_move, down_move), operator.gt(up_move, 0)),
        up_move,
        0
    )
    minus_dm = operator.condition(
        operator.and_(operator.gt(down_move, up_move), operator.gt(down_move, 0)),
        down_move,
        0
    )

    # Smoothed values (Wilder's smoothing)
    alpha = 1 / window
    smoothed_tr = ema(tr, span=window, alpha=alpha)
    smoothed_plus_dm = ema(plus_dm, span=window, alpha=alpha)
    smoothed_minus_dm = ema(minus_dm, span=window, alpha=alpha)

    # Directional Indicators
    plus_di_val = operator.mul(operator.div(smoothed_plus_dm, smoothed_tr), 100)
    minus_di_val = operator.mul(operator.div(smoothed_minus_dm, smoothed_tr), 100)

    # DX and ADX
    di_sum = operator.add(plus_di_val, minus_di_val)
    di_diff = operator.abs(operator.sub(plus_di_val, minus_di_val))
    dx = operator.mul(operator.div(di_diff, di_sum), 100)

    adx_val = ema(dx, span=window, alpha=alpha)

    return {
        "adx": adx_val,
        "plus_di": plus_di_val,
        "minus_di": minus_di_val,
    }


def adx_value(high, low, close, window: int = 14):
    """ADX value only.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period

    Returns:
        ADX values
    """
    return adx(high, low, close, window)["adx"]


def plus_di(high, low, close, window: int = 14):
    """+DI (Plus Directional Indicator).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period

    Returns:
        +DI values
    """
    return adx(high, low, close, window)["plus_di"]


def minus_di(high, low, close, window: int = 14):
    """-DI (Minus Directional Indicator).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period

    Returns:
        -DI values
    """
    return adx(high, low, close, window)["minus_di"]
