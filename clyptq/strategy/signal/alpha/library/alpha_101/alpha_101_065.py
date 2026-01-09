"""Alpha 101_065: Weighted open-VWAP amount correlation vs open range signal.

Formula: mul(lt(rank(ts_corr(add(mul({disk:open},0.00817205),mul({disk:vwap},sub(1,0.00817205))),ts_sum(ts_mean({disk:amount},60),8.6911),6.40374)),rank(sub({disk:open},ts_min({disk:open},13.635)))),-1)

Negative comparison between weighted open-VWAP amount correlation rank and open range rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_065(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_065: Weighted open-VWAP amount correlation vs open range.

    Negates the comparison between weighted price-amount correlation rank and open range rank.
    """

    default_params = {
        "weight": 0.00817205,
        "amount_window": 60,
        "sum_window": 9,
        "corr_window": 6,
        "min_window": 14,
    }

    @property
    def name(self) -> str:
        return "alpha_101_065"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_065."""
        open_ = data["open"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        weight = self.params["weight"]

        # Part 1: Weighted open-VWAP amount correlation
        # Weighted open-VWAP: open * weight + vwap * (1-weight)
        open_part = operator.mul(open_, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_open_vwap = operator.add(open_part, vwap_part)

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 9)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(weighted_open_vwap, amount_sum, 6)
        first_corr = operator.ts_corr(weighted_open_vwap, amount_sum, self.params["corr_window"])

        # rank(first_corr)
        first_rank = operator.rank(first_corr)

        # Part 2: Open range
        # ts_min(open, 14)
        open_min = operator.ts_min(open_, self.params["min_window"])

        # sub(open, open_min)
        open_diff = operator.sub(open_, open_min)

        # rank(open_diff)
        second_rank = operator.rank(open_diff)

        # lt(first_rank, second_rank)
        condition = operator.lt(first_rank, second_rank)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
