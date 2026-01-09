"""Liquidity-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class AmihudAlpha(BaseSignal):
    """Amihud illiquidity alpha.

    Higher liquidity (lower illiquidity) = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute Amihud liquidity scores.

        Args:
            data: Dict or UniverseData with 'close' and 'volume' DataFrames

        Returns:
            DataFrame of liquidity scores (higher = more liquid)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Calculate absolute returns
        returns = operator.ts_returns(close, 1)
        abs_returns = operator.abs(returns)

        # Dollar volume
        dollar_volume = operator.mul(close, volume)

        # Illiquidity = |return| / dollar_volume
        illiquidity = operator.div(abs_returns, dollar_volume)

        # Average illiquidity
        avg_illiquidity = operator.ts_mean(illiquidity, lookback)

        # Negative: prefer high liquidity (low illiquidity)
        score = operator.mul(avg_illiquidity, -1)

        return operator.ts_fillna(score, 0)


class EffectiveSpreadAlpha(BaseSignal):
    """Effective spread alpha (proxy using high-low range).

    Lower spread = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute effective spread scores.

        Args:
            data: Dict or UniverseData with 'high', 'low', 'close' DataFrames

        Returns:
            DataFrame of spread scores (higher = tighter spread)
        """
        high = data['high'] if isinstance(data, dict) else data.high
        low = data['low'] if isinstance(data, dict) else data.low
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Spread proxy = (high - low) / close
        range_val = operator.sub(high, low)
        spread = operator.div(range_val, close)

        # Average spread
        avg_spread = operator.ts_mean(spread, lookback)

        # Negative: prefer low spread
        score = operator.mul(avg_spread, -1)

        return operator.ts_fillna(score, 0)


class VolatilityOfVolatilityAlpha(BaseSignal):
    """Volatility of volatility alpha.

    Lower vol-of-vol = more stable = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20, "vol_window": 5}

    def compute(self, data):
        """Compute volatility-of-volatility scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame

        Returns:
            DataFrame of vol-of-vol scores (higher = more stable)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)
        vol_window = self.params.get("vol_window", 5)

        # Calculate returns
        returns = operator.ts_returns(close, 1)

        # Rolling volatility
        rolling_vol = operator.ts_std(returns, vol_window)

        # Volatility of volatility
        vol_of_vol = operator.ts_std(rolling_vol, lookback)

        # Negative: prefer stable volatility
        score = operator.mul(vol_of_vol, -1)

        return operator.ts_fillna(score, 0)
