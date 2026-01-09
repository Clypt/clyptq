"""Volume-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class VolumeTrendFactor(BaseSignal):
    """Volume Trend Factor.

    Rate of change in volume. Increasing volume = high loading.

    Formula: (volume[t] - volume[t-window]) / volume[t-window]
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 10, **kwargs):
        super().__init__(
            name=f"VolumeTrend_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        volume = data.volume if hasattr(data, 'volume') else data['volume']
        past_volume = operator.ts_delay(volume, self.lookback)
        vol_change = operator.div(operator.ts_delta(volume, self.lookback), past_volume)
        return vol_change
