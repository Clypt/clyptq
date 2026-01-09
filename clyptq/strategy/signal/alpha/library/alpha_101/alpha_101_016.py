"""Alpha 101_016: High-volume covariance ranking signal.

Formula: mul(-1,rank(ts_cov(rank({disk:high}),rank({disk:volume}),5)))

Negative rank of covariance between high price and volume rankings.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_016(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_016: Negative high-volume covariance ranking.

    Uses negative rank of covariance between ranked high prices and volumes.
    """

    default_params = {"cov_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_016"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_016."""
        high = data["high"]
        volume = data["volume"]

        # rank(high)
        high_rank = operator.rank(high)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_cov(rank(high), rank(volume), 5)
        cov = operator.ts_cov(high_rank, volume_rank, self.params["cov_window"])

        # rank(ts_cov(...))
        cov_rank = operator.rank(cov)

        # mul(-1, rank(...))
        alpha = operator.mul(cov_rank, -1)

        return operator.ts_fillna(alpha, 0)
