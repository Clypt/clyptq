"""Alpha 101_019: Price direction and long-term returns signal.

Formula: mul(mul(-1,sign(add(sub({disk:close},delay({disk:close},7)),ts_delta({disk:close},7)))),add(1,rank(add(1,ts_sum({disk:returns},250)))))

Combines 7-day price direction with long-term cumulative returns ranking.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_019(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_019: Price direction with long-term returns.

    Combines negative sign of 7-day price change with long-term returns ranking.
    """

    default_params = {"delay_window": 7, "delta_window": 7, "returns_sum_window": 250}

    @property
    def name(self) -> str:
        return "alpha_101_019"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_019."""
        close = data["close"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # close - delay(close, 7)
        close_lag = operator.delay(close, self.params["delay_window"])
        close_diff = operator.sub(close, close_lag)

        # ts_delta(close, 7)
        close_delta = operator.ts_delta(close, self.params["delta_window"])

        # add(close_diff, close_delta)
        sum_changes = operator.add(close_diff, close_delta)

        # sign(...) * -1
        sign_changes = operator.sign(sum_changes)
        neg_sign = operator.mul(sign_changes, -1)

        # ts_sum(returns, 250)
        returns_sum = operator.ts_sum(returns, self.params["returns_sum_window"])

        # add(1, returns_sum)
        returns_plus1 = operator.add(returns_sum, 1)

        # rank(...) + 1
        returns_rank = operator.rank(returns_plus1)
        rank_plus1 = operator.add(returns_rank, 1)

        # Final multiplication
        alpha = operator.mul(neg_sign, rank_plus1)

        return operator.ts_fillna(alpha, 0)
