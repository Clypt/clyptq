"""Alpha 101_095: Open-min open difference rank less than mid-amount correlation rank power ts_rank signal.

Formula: lt(rank(sub({disk:open},ts_min({disk:open},12.4105))),ts_rank(pow(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),19.1351),ts_sum(ts_mean({disk:amount},40),19.1351),12.8742)),5),11.7584))

Comparison of open-min open difference rank and mid-amount correlation rank power ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_095(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_095: Open-min open difference rank less than mid-amount correlation rank power ts_rank.

    Compares open-min open difference rank to mid-amount correlation rank power ts_rank.
    """

    default_params = {
        "min_window": 12,
        "sum_window": 19,
        "amount_window": 40,
        "corr_window": 13,
        "power_exp": 5,
        "final_rank_window": 12,
    }

    @property
    def name(self) -> str:
        return "alpha_101_095"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_095."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Open - min open difference rank
        # ts_min(open, 12)
        open_min = operator.ts_min(open_, self.params["min_window"])

        # sub(open, open_min)
        open_diff = operator.sub(open_, open_min)

        # rank(open_diff)
        first_part = operator.rank(open_diff)

        # Part 2: Mid-amount correlation rank power ts_rank
        # div(add(high, low), 2) - mid price
        mid_price = operator.div(operator.add(high, low), 2)

        # ts_sum(mid_price, 19)
        mid_sum = operator.ts_sum(mid_price, self.params["sum_window"])

        # ts_mean(amount, 40)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 19)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(mid_sum, amount_sum, 13)
        corr_result = operator.ts_corr(mid_sum, amount_sum, self.params["corr_window"])

        # rank(corr_result)
        corr_rank = operator.rank(corr_result)

        # pow(corr_rank, 5)
        corr_powered = operator.pow(corr_rank, self.params["power_exp"])

        # ts_rank(corr_powered, 12)
        second_part = operator.ts_rank(corr_powered, self.params["final_rank_window"])

        # lt(first_part, second_part)
        alpha = operator.lt(first_part, second_part)

        # Convert boolean to float
        alpha = alpha.astype(float)

        return operator.ts_fillna(alpha, 0)
