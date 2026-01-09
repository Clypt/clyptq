"""Alpha 101_015: High-volume correlation ranking signal.

Formula: mul(-1,ts_sum(rank(ts_corr(rank({disk:high}),rank({disk:volume}),3)),3))

Negative sum of ranked correlations between high price and volume rankings.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_015(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_015: High-volume correlation sum.

    Negative sum of ranked correlations between high and volume rankings.
    """

    default_params = {"corr_window": 3, "sum_window": 3}

    @property
    def name(self) -> str:
        return "alpha_101_015"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_015."""
        high = data["high"]
        volume = data["volume"]

        # rank(high)
        high_rank = operator.rank(high)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(rank(high), rank(volume), 3)
        corr = operator.ts_corr(high_rank, volume_rank, self.params["corr_window"])

        # rank(ts_corr(...))
        corr_rank = operator.rank(corr)

        # ts_sum(rank(...), 3)
        sum_rank = operator.ts_sum(corr_rank, self.params["sum_window"])

        # mul(-1, ts_sum(...))
        alpha = operator.mul(sum_rank, -1)

        return operator.ts_fillna(alpha, 0)
