"""Alpha 101_088: Price rank sum difference decay rank vs close-amount ts_rank correlation decay rank min signal.

Formula: min(rank(ts_decayed_linear(sub(add(rank({disk:open}),rank({disk:low})),add(rank({disk:high}),rank({disk:close}))),8.06882)),ts_rank(ts_decayed_linear(ts_corr(ts_rank({disk:close},8.44728),ts_rank(ts_mean({disk:amount},60),20.6966),8.01266),6.65053),2.61957))

Minimum of price rank sum difference decay rank and close-amount ts_rank correlation decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_088(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_088: Price rank sum difference decay rank vs close-amount ts_rank correlation decay rank min.

    Returns the minimum of two complex decay rank values.
    """

    default_params = {
        "decay_window1": 8,
        "close_rank_window": 8,
        "amount_window": 60,
        "amount_rank_window": 21,
        "corr_window": 8,
        "decay_window2": 7,
        "final_rank_window": 3,
    }

    @property
    def name(self) -> str:
        return "alpha_101_088"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_088."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Price rank sum difference decay rank
        # rank(open)
        open_rank = operator.rank(open_)

        # rank(low)
        low_rank = operator.rank(low)

        # rank(high)
        high_rank = operator.rank(high)

        # rank(close)
        close_rank = operator.rank(close)

        # (open_rank + low_rank)
        open_low_sum = operator.add(open_rank, low_rank)

        # (high_rank + close_rank)
        high_close_sum = operator.add(high_rank, close_rank)

        # (open_rank + low_rank) - (high_rank + close_rank)
        rank_diff = operator.sub(open_low_sum, high_close_sum)

        # ts_decayed_linear(rank_diff, 8)
        diff_decayed = operator.ts_decayed_linear(rank_diff, self.params["decay_window1"])

        # rank(diff_decayed)
        first_part = operator.rank(diff_decayed)

        # Part 2: Close-amount ts_rank correlation decay ts_rank
        # ts_rank(close, 8)
        close_tsrank = operator.ts_rank(close, self.params["close_rank_window"])

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 21)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(close_tsrank, amount_tsrank, 8)
        corr = operator.ts_corr(close_tsrank, amount_tsrank, self.params["corr_window"])

        # ts_decayed_linear(corr, 7)
        corr_decayed = operator.ts_decayed_linear(corr, self.params["decay_window2"])

        # ts_rank(corr_decayed, 3)
        second_part = operator.ts_rank(corr_decayed, self.params["final_rank_window"])

        # min(first_part, second_part)
        alpha = operator.elem_min(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
