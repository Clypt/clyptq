"""Alpha 101_026: Volume-high time-series correlation signal.

Formula: mul(-1,ts_max(ts_corr(ts_rank({disk:volume},5),ts_rank({disk:high},5),5),3))

Negative of 3-period max of 5-period correlation between volume and high rankings.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_026(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_026: Volume-high time-series correlation.

    Uses negative max of correlation between time-series ranked volume and high.
    """

    default_params = {"ts_rank_window": 5, "corr_window": 5, "max_window": 3}

    @property
    def name(self) -> str:
        return "alpha_101_026"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_026."""
        volume = data["volume"]
        high = data["high"]

        # ts_rank(volume, 5)
        volume_rank = operator.ts_rank(volume, self.params["ts_rank_window"])

        # ts_rank(high, 5)
        high_rank = operator.ts_rank(high, self.params["ts_rank_window"])

        # ts_corr(volume_rank, high_rank, 5)
        corr = operator.ts_corr(volume_rank, high_rank, self.params["corr_window"])

        # ts_max(corr, 3)
        max_corr = operator.ts_max(corr, self.params["max_window"])

        # mul(-1, max_corr)
        alpha = operator.mul(max_corr, -1)

        return operator.ts_fillna(alpha, 0)
