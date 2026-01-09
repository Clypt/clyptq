"""Alpha 101_022: High-volume correlation change signal.

Formula: mul(-1,mul(ts_delta(ts_corr({disk:high},{disk:volume},5),5),rank(ts_std({disk:close},20))))

Negative product of high-volume correlation change and close volatility ranking.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_022(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_022: High-volume correlation change with volatility.

    Combines change in high-volume correlation with close volatility ranking.
    """

    default_params = {"corr_window": 5, "delta_window": 5, "std_window": 20}

    @property
    def name(self) -> str:
        return "alpha_101_022"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_022."""
        high = data["high"]
        close = data["close"]
        volume = data["volume"]

        # ts_corr(high, volume, 5)
        corr = operator.ts_corr(high, volume, self.params["corr_window"])

        # ts_delta(corr, 5)
        corr_delta = operator.ts_delta(corr, self.params["delta_window"])

        # rank(ts_std(close, 20))
        close_std = operator.ts_std(close, self.params["std_window"])
        std_rank = operator.rank(close_std)

        # mul(corr_delta, std_rank)
        product = operator.mul(corr_delta, std_rank)

        # mul(-1, product)
        alpha = operator.mul(product, -1)

        return operator.ts_fillna(alpha, 0)
