"""Liquidity-based factors."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class LiquidityFactor(BaseSignal):
    """Liquidity Factor (Inverse of Amihud Illiquidity).

    Low price impact relative to high volume = high liquidity = high loading.

    Formula: -mean(|return| / volume, window)
    """

    role = SignalRole.FACTOR

    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(
            name=f"Liquidity_{lookback}D",
            lookback=lookback,
            **kwargs
        )

    def compute(self, data):
        returns = operator.abs(data.returns if hasattr(data, 'returns') else operator.ts_returns(data.close if hasattr(data, 'close') else data['close'], 1))
        volume = data.volume if hasattr(data, 'volume') else data['volume']

        # Amihud illiquidity measure (inverted for liquidity)
        illiquidity = operator.div(returns, volume)
        avg_illiquidity = operator.ts_mean(illiquidity, self.lookback)

        # Invert: high liquidity = low illiquidity
        liquidity = operator.neg(avg_illiquidity)

        return liquidity
