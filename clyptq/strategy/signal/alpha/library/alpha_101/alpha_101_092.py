"""Alpha 101_092: Mid-close vs low-open comparison decay rank min with low-amount rank correlation decay rank signal.

Formula: min(ts_rank(ts_decayed_linear(lt(add(div(add({disk:high},{disk:low}),2),{disk:close}),add({disk:low},{disk:open})),14.7221),18.8683),ts_rank(ts_decayed_linear(ts_corr(rank({disk:low}),rank(ts_mean({disk:amount},30)),7.58555),6.94024),6.80584))

Minimum of price comparison decay ts_rank and low-amount rank correlation decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_092(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_092: Mid-close vs low-open comparison decay rank min with low-amount rank correlation decay rank.

    Returns minimum of price comparison condition decay ts_rank and low-amount rank correlation decay ts_rank.
    """

    default_params = {
        "decay_window1": 15,
        "rank_window1": 19,
        "amount_window": 30,
        "corr_window": 8,
        "decay_window2": 7,
        "rank_window2": 7,
    }

    @property
    def name(self) -> str:
        return "alpha_101_092"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_092."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Price comparison condition decay ts_rank
        # div(add(high, low), 2) - mid price
        mid_price = operator.div(operator.add(high, low), 2)

        # add(mid_price, close)
        mid_close_sum = operator.add(mid_price, close)

        # add(low, open)
        low_open_sum = operator.add(low, open_)

        # lt(mid_close_sum, low_open_sum)
        price_condition = operator.lt(mid_close_sum, low_open_sum)

        # ts_decayed_linear(price_condition, 15)
        condition_decayed = operator.ts_decayed_linear(
            price_condition.astype(float), self.params["decay_window1"]
        )

        # ts_rank(condition_decayed, 19)
        first_part = operator.ts_rank(condition_decayed, self.params["rank_window1"])

        # Part 2: Low-amount rank correlation decay ts_rank
        # rank(low)
        low_rank = operator.rank(low)

        # ts_mean(amount, 30)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # rank(amount_mean)
        amount_rank = operator.rank(amount_mean)

        # ts_corr(low_rank, amount_rank, 8)
        low_amount_corr = operator.ts_corr(low_rank, amount_rank, self.params["corr_window"])

        # ts_decayed_linear(low_amount_corr, 7)
        corr_decayed = operator.ts_decayed_linear(low_amount_corr, self.params["decay_window2"])

        # ts_rank(corr_decayed, 7)
        second_part = operator.ts_rank(corr_decayed, self.params["rank_window2"])

        # min(first_part, second_part)
        alpha = operator.elem_min(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
