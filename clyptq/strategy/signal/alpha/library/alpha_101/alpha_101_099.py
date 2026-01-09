"""Alpha 101_099: Mid-amount sum correlation rank less than low-volume correlation rank comparison signal.

Formula: mul(lt(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),19.8975),ts_sum(ts_mean({disk:amount},60),19.8975),8.8136)),rank(ts_corr({disk:low},{disk:volume},6.28259))),-1)

Negative of comparison between mid-amount sum correlation rank and low-volume correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_099(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_099: Mid-amount sum correlation rank less than low-volume correlation rank comparison.

    Negates the comparison between mid-amount sum correlation rank and low-volume correlation rank.
    """

    default_params = {
        "sum_window": 20,
        "amount_window": 60,
        "corr_window1": 9,
        "corr_window2": 6,
    }

    @property
    def name(self) -> str:
        return "alpha_101_099"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_099."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Mid-amount sum correlation rank
        # div(add(high, low), 2) - mid price
        mid_price = operator.div(operator.add(high, low), 2)

        # ts_sum(mid_price, 20)
        mid_sum = operator.ts_sum(mid_price, self.params["sum_window"])

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 20)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(mid_sum, amount_sum, 9)
        mid_amount_corr = operator.ts_corr(mid_sum, amount_sum, self.params["corr_window1"])

        # rank(mid_amount_corr)
        first_rank = operator.rank(mid_amount_corr)

        # Part 2: Low-volume correlation rank
        # ts_corr(low, volume, 6)
        low_vol_corr = operator.ts_corr(low, volume, self.params["corr_window2"])

        # rank(low_vol_corr)
        second_rank = operator.rank(low_vol_corr)

        # lt(first_rank, second_rank)
        condition = operator.lt(first_rank, second_rank)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
