"""Alpha 101_006: Open-volume correlation signal.

Formula: mul(-1,ts_corr({disk:open},{disk:volume},10))

Negative correlation between open price and volume.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_006(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_006: Open-volume negative correlation.

    Uses negative correlation between open prices and volumes directly.
    """

    default_params = {"corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_006"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_006."""
        open_ = data["open"]
        volume = data["volume"]

        # ts_corr(open, volume, 10)
        corr = operator.ts_corr(open_, volume, self.params["corr_window"])

        # mul(-1, corr)
        alpha = operator.mul(corr, -1)

        return operator.ts_fillna(alpha, 0)
