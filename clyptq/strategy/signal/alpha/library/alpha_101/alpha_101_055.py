"""Alpha 101_055: Stochastic-volume correlation signal.

Formula: mul(-1,ts_corr(rank(div(sub({disk:close},ts_min({disk:low},12)),sub(ts_max({disk:high},12),ts_min({disk:low},12)))),rank({disk:volume}),6))

Negative of 6-period correlation between ranked stochastic %K and ranked volume.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_055(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_055: Stochastic-volume correlation.

    Negates the correlation between ranked 12-period stochastic %K and ranked volume.
    """

    default_params = {"stoch_window": 12, "corr_window": 6}

    @property
    def name(self) -> str:
        return "alpha_101_055"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_055."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        stoch_window = self.params["stoch_window"]

        # Stochastic %K calculation
        # ts_min(low, 12)
        low_min = operator.ts_min(low, stoch_window)

        # ts_max(high, 12)
        high_max = operator.ts_max(high, stoch_window)

        # close - low_min
        numerator = operator.sub(close, low_min)

        # high_max - low_min
        denominator = operator.sub(high_max, low_min)

        # Stochastic %K
        stochastic_k = operator.div(numerator, denominator)

        # rank(stochastic_k)
        stoch_rank = operator.rank(stochastic_k)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(stoch_rank, volume_rank, 6)
        corr = operator.ts_corr(stoch_rank, volume_rank, self.params["corr_window"])

        # mul(-1, corr)
        alpha = operator.mul(corr, -1)

        return operator.ts_fillna(alpha, 0)
