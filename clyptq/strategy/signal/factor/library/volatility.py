"""Volatility-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class VolatilityFactor(BaseSignal):
    """Volatility Factor.

    Standard deviation of returns over N days. High volatility = high loading.

    Formula: std(returns, window)
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"Volatility_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        returns = data.returns if hasattr(data, 'returns') else operator.ts_returns(data.close if hasattr(data, 'close') else data['close'], 1)
        volatility = operator.ts_std(returns, self.lookback)
        return volatility
