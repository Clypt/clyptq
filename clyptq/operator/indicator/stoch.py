"""Stochastic Oscillator indicator."""

from clyptq import operator
from clyptq.operator.indicator.sma import sma


def stoch(
    high,
    low,
    close,
    k_window: int = 14,
    d_window: int = 3,
    smooth_k: int = 1
):
    """Stochastic Oscillator.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: %K lookback period (default 14)
        d_window: %D smoothing period (default 3)
        smooth_k: %K smoothing period (default 1, no smoothing)

    Returns:
        Dict with 'k' and 'd' values
    """
    lowest_low = operator.ts_min(low, k_window)
    highest_high = operator.ts_max(high, k_window)

    stoch_k = operator.mul(
        operator.div(
            operator.sub(close, lowest_low),
            operator.sub(highest_high, lowest_low)
        ),
        100
    )

    if smooth_k > 1:
        stoch_k = sma(stoch_k, smooth_k)

    stoch_d = sma(stoch_k, d_window)

    return {
        "k": stoch_k,
        "d": stoch_d,
    }


def fast_stoch(high, low, close, k_window: int = 14, d_window: int = 3):
    """Fast Stochastic Oscillator.

    Uses raw %K without smoothing.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: %K lookback period
        d_window: %D smoothing period

    Returns:
        Dict with 'k' and 'd' values
    """
    return stoch(high, low, close, k_window, d_window, smooth_k=1)


def slow_stoch(
    high,
    low,
    close,
    k_window: int = 14,
    d_window: int = 3,
    smooth_k: int = 3
):
    """Slow Stochastic Oscillator.

    Uses smoothed %K (typically 3-period SMA).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: %K lookback period
        d_window: %D smoothing period
        smooth_k: %K smoothing period

    Returns:
        Dict with 'k' and 'd' values
    """
    return stoch(high, low, close, k_window, d_window, smooth_k=smooth_k)
