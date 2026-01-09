"""Quality-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class QualityFactor(BaseSignal):
    """Quality Factor (Volume Stability Proxy).

    Low volume volatility = high quality = high loading.

    Formula: -std(volume, window) / mean(volume, window)
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"Quality_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        volume = data.volume if hasattr(data, 'volume') else data['volume']
        vol_mean = operator.ts_mean(volume, self.lookback)
        vol_std = operator.ts_std(volume, self.lookback)

        # Coefficient of variation (inverted: low CV = high quality)
        cv = operator.div(vol_std, vol_mean)

        return operator.neg(cv)
