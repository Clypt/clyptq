"""Alpha 101_040: High volatility with high-volume correlation signal.

Formula: mul(mul(-1,rank(ts_std({disk:high},10))),ts_corr({disk:high},{disk:volume},10))

Negative product of high price volatility rank and high-volume correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_040(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_040: High volatility with correlation.

    Multiplies negative high volatility rank with high-volume correlation.
    """

    default_params = {"std_window": 10, "corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_040"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_040."""
        high = data["high"]
        volume = data["volume"]

        # ts_std(high, 10)
        high_std = operator.ts_std(high, self.params["std_window"])

        # rank(high_std) * -1
        std_rank = operator.rank(high_std)
        neg_rank = operator.mul(std_rank, -1)

        # ts_corr(high, volume, 10)
        corr = operator.ts_corr(high, volume, self.params["corr_window"])

        # neg_rank * corr
        alpha = operator.mul(neg_rank, corr)

        return operator.ts_fillna(alpha, 0)
