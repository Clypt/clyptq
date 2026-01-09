"""Alpha 101_004: Low price time-series rank signal.

Formula: mul(-1,ts_rank(rank({disk:low}),9))

Inverted time-series rank of low price rankings.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_004(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_004: Inverted time-series rank of low prices.

    Uses negative of time-series rank of cross-sectional low price rankings.
    """

    default_params = {"ts_rank_window": 9}

    @property
    def name(self) -> str:
        return "alpha_101_004"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_004."""
        low = data["low"]

        # rank(low)
        low_rank = operator.rank(low)

        # ts_rank(low_rank, 9)
        ts_ranked = operator.ts_rank(low_rank, self.params["ts_rank_window"])

        # mul(-1, ts_ranked)
        alpha = operator.mul(ts_ranked, -1)

        return operator.ts_fillna(alpha, 0)
