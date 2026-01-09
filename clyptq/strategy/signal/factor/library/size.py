"""Size-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class SizeFactor(BaseSignal):
    """Size Factor.

    Volume-based size proxy. High volume = high loading.
    (In crypto, volume is used instead of market capitalization)

    Formula: mean(volume, window)
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"Size_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        volume = data.volume if hasattr(data, 'volume') else data['volume']
        avg_volume = operator.ts_mean(volume, self.lookback)
        return avg_volume
