"""Relative Strength Index indicator."""

from clyptq import operator
from clyptq.operator.indicator.ema import ema
from clyptq.operator.indicator.sma import sma


def rsi(close, window: int = 14, method: str = "ema"):
    """Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Args:
        close: Close prices
        window: Lookback period (default 14)
        method: Averaging method - "ema" (Wilder's) or "sma"

    Returns:
        RSI values (0-100)
    """
    delta = operator.ts_delta(close, 1)

    gain = operator.condition(operator.gt(delta, 0), delta, 0)
    loss = operator.condition(operator.lt(delta, 0), operator.neg(delta), 0)

    if method == "ema":
        # Wilder's smoothing (EMA with alpha = 1/window)
        avg_gain = ema(gain, span=window, alpha=1/window)
        avg_loss = ema(loss, span=window, alpha=1/window)
    else:
        avg_gain = sma(gain, window)
        avg_loss = sma(loss, window)

    rs = operator.div(avg_gain, avg_loss)
    rsi_val = operator.sub(100, operator.div(100, operator.add(1, rs)))

    return rsi_val


def stoch_rsi(
    close,
    rsi_window: int = 14,
    stoch_window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
):
    """Stochastic RSI.

    Applies stochastic calculation to RSI values.

    Args:
        close: Close prices
        rsi_window: RSI period
        stoch_window: Stochastic lookback
        smooth_k: %K smoothing period
        smooth_d: %D smoothing period

    Returns:
        Dict with 'k' and 'd' values
    """
    rsi_val = rsi(close, rsi_window)

    rsi_min = operator.ts_min(rsi_val, stoch_window)
    rsi_max = operator.ts_max(rsi_val, stoch_window)

    stoch_rsi_k = operator.mul(
        operator.div(
            operator.sub(rsi_val, rsi_min),
            operator.sub(rsi_max, rsi_min)
        ),
        100
    )

    if smooth_k > 1:
        stoch_rsi_k = sma(stoch_rsi_k, smooth_k)

    stoch_rsi_d = sma(stoch_rsi_k, smooth_d)

    return {
        "k": stoch_rsi_k,
        "d": stoch_rsi_d,
    }
