"""Mean reversion factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class MeanReversionFactor(BaseSignal):
    """Mean Reversion Factor.

    Degree of deviation from average price (Z-score).
    High Z-score = overvalued = high loading.

    Formula: ts_zscore(close, window)
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"MeanReversion_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        close = data.close if hasattr(data, 'close') else data['close']
        zscores = operator.ts_zscore(close, self.lookback)
        return zscores
