"""Quality-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class VolumeStabilityAlpha(BaseSignal):
    """Volume stability alpha: 1 / coefficient of variation.

    Higher stability (lower CV) = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute volume stability scores.

        Args:
            data: Dict or UniverseData with 'volume' DataFrame

        Returns:
            DataFrame of stability scores
        """
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Mean and std of volume
        mean_vol = operator.ts_mean(volume, lookback)
        std_vol = operator.ts_std(volume, lookback)

        # Coefficient of variation = std / mean
        cv = operator.div(std_vol, mean_vol)

        # Stability = 1 / cv (prefer low CV)
        stability = operator.div(1.0, cv)

        return operator.ts_fillna(stability, 0)


class PriceImpactAlpha(BaseSignal):
    """Price impact alpha: lower impact = higher score.

    Price impact = |return| / log(volume)
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute price impact scores.

        Args:
            data: Dict or UniverseData with 'close' and 'volume' DataFrames

        Returns:
            DataFrame of price impact scores (higher = lower impact)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Absolute returns
        returns = operator.ts_returns(close, 1)
        abs_returns = operator.abs(returns)

        # Log volume
        log_volume = operator.log(operator.add(volume, 1.0))

        # Impact = |return| / log(volume)
        impact = operator.div(abs_returns, log_volume)

        # Average impact
        avg_impact = operator.ts_mean(impact, lookback)

        # Negative: prefer low price impact
        score = operator.mul(avg_impact, -1)

        return operator.ts_fillna(score, 0)


class MarketDepthProxyAlpha(BaseSignal):
    """Market depth proxy alpha: avg_volume / volatility.

    Higher depth = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute market depth proxy scores.

        Args:
            data: Dict or UniverseData with 'close' and 'volume' DataFrames

        Returns:
            DataFrame of depth proxy scores
        """
        close = data['close'] if isinstance(data, dict) else data.close
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Calculate returns and volatility
        returns = operator.ts_returns(close, 1)
        volatility = operator.ts_std(returns, lookback)

        # Average volume
        avg_volume = operator.ts_mean(volume, lookback)

        # Depth proxy = volume / volatility
        depth_proxy = operator.div(avg_volume, volatility)

        return operator.ts_fillna(depth_proxy, 0)
