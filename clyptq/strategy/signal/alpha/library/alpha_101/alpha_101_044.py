"""Alpha 101_044: High-volume rank correlation signal.

Formula: mul(-1,ts_corr({disk:high},rank({disk:volume}),5))

Negative 5-period correlation between high price and volume rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_044(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_044: Negative high-volume rank correlation.

    Uses negative correlation between high price and cross-sectional volume rank.
    """

    default_params = {"corr_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_044"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_044."""
        high = data["high"]
        volume = data["volume"]

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(high, volume_rank, 5)
        corr = operator.ts_corr(high, volume_rank, self.params["corr_window"])

        # -1 * corr
        alpha = operator.mul(corr, -1)

        return operator.ts_fillna(alpha, 0)
