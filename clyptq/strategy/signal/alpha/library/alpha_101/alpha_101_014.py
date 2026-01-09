"""Alpha 101_014: Returns delta and open-volume correlation signal.

Formula: mul(mul(-1,rank(ts_delta({disk:returns},3))),ts_corr({disk:open},{disk:volume},10))

Combination of returns change ranking and open-volume correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_014(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_014: Returns delta with open-volume correlation.

    Multiplies negative returns delta rank with open-volume correlation.
    """

    default_params = {"delta_window": 3, "corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_014"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_014."""
        open_ = data["open"]
        close = data["close"]
        volume = data["volume"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # ts_delta(returns, 3)
        returns_delta = operator.ts_delta(returns, self.params["delta_window"])

        # rank(ts_delta(returns, 3))
        rank_returns_delta = operator.rank(returns_delta)

        # -1 * rank(...)
        neg_rank_returns = operator.mul(rank_returns_delta, -1)

        # ts_corr(open, volume, 10)
        open_volume_corr = operator.ts_corr(open_, volume, self.params["corr_window"])

        # (-1 * rank(...)) * ts_corr(...)
        alpha = operator.mul(neg_rank_returns, open_volume_corr)

        return operator.ts_fillna(alpha, 0)
