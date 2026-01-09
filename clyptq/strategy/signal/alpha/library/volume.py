"""Volume-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class VolumeAlpha(BaseSignal):
    """Volume alpha: recent volume / average volume.

    Higher values indicate unusually high volume.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute volume ratio scores.

        Args:
            data: Dict or UniverseData with 'volume' DataFrame (timestamps x symbols)

        Returns:
            DataFrame of volume ratio scores
        """
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Average volume over lookback
        avg_volume = operator.ts_mean(volume, lookback)

        # Ratio = current / average
        score = operator.div(volume, avg_volume)

        return operator.ts_fillna(score, 0)


class DollarVolumeAlpha(BaseSignal):
    """Dollar volume alpha: average dollar volume.

    Higher values indicate more liquid assets.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute average dollar volume scores.

        Args:
            data: Dict or UniverseData with 'close' and 'volume' DataFrames

        Returns:
            DataFrame of dollar volume scores
        """
        close = data['close'] if isinstance(data, dict) else data.close
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Dollar volume = price * volume
        dollar_volume = operator.mul(close, volume)

        # Average dollar volume
        avg_dollar_volume = operator.ts_mean(dollar_volume, lookback)

        return operator.ts_fillna(avg_dollar_volume, 0)


class VolumeRatioAlpha(BaseSignal):
    """Volume ratio alpha: short-term / long-term average volume.

    Higher values indicate volume surge (potential momentum).
    """

    role = SignalRole.ALPHA
    default_params = {"short_window": 5, "long_window": 20}

    def compute(self, data):
        """Compute volume ratio scores.

        Args:
            data: Dict or UniverseData with 'volume' DataFrame

        Returns:
            DataFrame of volume ratio scores
        """
        volume = data['volume'] if isinstance(data, dict) else data.volume
        short_window = self.params.get("short_window", 5)
        long_window = self.params.get("long_window", 20)

        # Short-term and long-term averages
        short_avg = operator.ts_mean(volume, short_window)
        long_avg = operator.ts_mean(volume, long_window)

        # Ratio = short / long
        score = operator.div(short_avg, long_avg)

        return operator.ts_fillna(score, 0)
