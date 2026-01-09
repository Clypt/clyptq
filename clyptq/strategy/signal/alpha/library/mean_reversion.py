"""Mean reversion alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class BollingerAlpha(BaseSignal):
    """Bollinger mean reversion: (middle - price) / bandwidth.

    Positive when price is below middle band (oversold).
    Negative when price is above middle band (overbought).
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20, "num_std": 2.0}

    def compute(self, data):
        """Compute Bollinger band mean reversion scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of Bollinger scores clipped to [-1, 1]
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)
        num_std = self.params.get("num_std", 2.0)

        # Calculate middle band (SMA) and standard deviation
        middle = operator.ts_mean(close, lookback)
        std = operator.ts_std(close, lookback)

        # Bandwidth = num_std * std
        bandwidth = operator.mul(std, num_std)

        # Score = (middle - price) / bandwidth
        deviation = operator.sub(middle, close)
        score = operator.div(deviation, bandwidth)

        # Clip to [-1, 1]
        clipped = operator.clip(score, lower=-1.0, upper=1.0)
        return operator.ts_fillna(clipped, 0)


class ZScoreAlpha(BaseSignal):
    """Z-Score mean reversion: -(price - mean) / std.

    Positive when price is below mean (oversold).
    Negative when price is above mean (overbought).
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute Z-Score mean reversion scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of Z-scores clipped to [-3, 3]
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # Calculate mean and std
        mean = operator.ts_mean(close, lookback)
        std = operator.ts_std(close, lookback)

        # Z-score = -(price - mean) / std (negative for mean reversion)
        deviation = operator.sub(close, mean)
        zscore = operator.div(deviation, std)
        score = operator.mul(zscore, -1)

        # Clip to [-3, 3]
        clipped = operator.clip(score, lower=-3.0, upper=3.0)
        return operator.ts_fillna(clipped, 0)


class PercentileAlpha(BaseSignal):
    """Percentile mean reversion: -(percentile - 0.5) * 2.

    Positive when price is in lower percentile (oversold).
    Negative when price is in upper percentile (overbought).
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute percentile-based mean reversion scores.

        Args:
            data: Dict or UniverseData with 'close' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of percentile scores in [-1, 1]
        """
        close = data['close'] if isinstance(data, dict) else data.close
        lookback = self.params.get("lookback", self.lookback)

        # ts_rank gives percentile rank within lookback window
        percentile = operator.ts_rank(close, lookback)

        # Convert to [-1, 1]: -(percentile - 0.5) * 2
        centered = operator.sub(percentile, 0.5)
        score = operator.mul(centered, -2)

        return operator.ts_fillna(score, 0)
