"""Momentum-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class MomentumFactor(BaseSignal):
    """Momentum Factor.

    Rate of price change over N days. High momentum = high loading.

    Formula: (close[t] - close[t-window]) / close[t-window]

    Example:
        ```python
        factor = MomentumFactor(lookback=20)
        loading = factor.compute(data)  # (T x N) DataFrame
        ```
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"Momentum_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        close = data.close if hasattr(data, 'close') else data['close']
        past_close = operator.ts_delay(close, self.lookback)
        momentum = operator.div(operator.ts_delta(close, self.lookback), past_close)
        return momentum


class ShortTermReversalFactor(BaseSignal):
    """Short-Term Reversal Factor.

    5-day return reversal. Recently declining assets = high loading (mean reversion).

    Formula: -ts_delta(close, 5) / ts_delay(close, 5)
    """

    role = SignalRole.FACTOR

    def __init__(self, **kwargs):
        super().__init__(
            name="ShortTermReversal",
            lookback=5,
            **kwargs
        )

    def compute(self, data):
        close = data.close if hasattr(data, 'close') else data['close']
        past_close = operator.ts_delay(close, 5)
        short_momentum = operator.div(operator.ts_delta(close, 5), past_close)
        return operator.neg(short_momentum)
