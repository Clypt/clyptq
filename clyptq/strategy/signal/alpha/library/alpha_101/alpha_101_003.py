"""Alpha 101_003: Open-volume correlation signal.

Formula: mul(-1,ts_corr(rank({disk:open}),rank({disk:volume}),10))

Negative correlation between open price rank and volume rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_003(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_003: Open-volume negative correlation.

    Uses negative correlation between ranked open prices and ranked volumes.
    """

    default_params = {"corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_003"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_003."""
        open_ = data["open"]
        volume = data["volume"]

        # rank(open)
        open_rank = operator.rank(open_)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(open_rank, volume_rank, 10)
        corr = operator.ts_corr(open_rank, volume_rank, self.params["corr_window"])

        # mul(-1, corr)
        alpha = operator.mul(corr, -1)

        return operator.ts_fillna(alpha, 0)
