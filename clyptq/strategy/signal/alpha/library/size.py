"""Size-based alpha signals."""

from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class DollarVolumeSizeAlpha(BaseSignal):
    """Dollar volume size alpha: log of average dollar volume.

    Larger market cap proxy = higher score.
    """

    role = SignalRole.ALPHA
    default_params = {"lookback": 20}

    def compute(self, data):
        """Compute dollar volume size scores.

        Args:
            data: Dict or UniverseData with 'close' and 'volume' DataFrames

        Returns:
            DataFrame of size scores (log dollar volume)
        """
        close = data['close'] if isinstance(data, dict) else data.close
        volume = data['volume'] if isinstance(data, dict) else data.volume
        lookback = self.params.get("lookback", self.lookback)

        # Dollar volume = price * volume
        dollar_volume = operator.mul(close, volume)

        # Average dollar volume
        avg_dollar_volume = operator.ts_mean(dollar_volume, lookback)

        # Log scale
        score = operator.log(operator.add(avg_dollar_volume, 1.0))

        return operator.ts_fillna(score, 0)
