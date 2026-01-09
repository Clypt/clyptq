"""Volatility-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class VolatilityAlpha(BaseSignal):
    """Inverse volatility alpha: prefer low volatility assets.

    Lower volatility = higher score (risk parity style).
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute inverse volatility scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of inverse volatility scores (ranked)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Calculate returns
        returns = operator.ts_returns(close, 1)

        # Calculate rolling std of returns
        volatility = operator.ts_std(returns, lookback)

        # Inverse: lower vol = higher score
        inv_vol = operator.mul(volatility, -1)

        # Rank cross-sectionally
        ranked = operator.rank(inv_vol)

        return operator.ts_fillna(ranked, 0)
