"""Alpha 101_013: Close-volume covariance signal.

Formula: mul(-1,rank(ts_cov(rank({disk:close}),rank({disk:volume}),5)))

Negative rank of covariance between close and volume rankings.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_013(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_013: Negative close-volume covariance ranking.

    Uses negative rank of covariance between ranked close and volume.
    """

    default_params = {"cov_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_013"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_013."""
        close = data["close"]
        volume = data["volume"]

        # rank(close)
        close_rank = operator.rank(close)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_cov(rank(close), rank(volume), 5)
        cov = operator.ts_cov(close_rank, volume_rank, self.params["cov_window"])

        # rank(ts_cov(...))
        ranked_cov = operator.rank(cov)

        # -1 * rank(...)
        alpha = operator.mul(ranked_cov, -1)

        return operator.ts_fillna(alpha, 0)
