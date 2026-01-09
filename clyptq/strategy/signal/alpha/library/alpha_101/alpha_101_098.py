"""Alpha 101_098: VWAP-amount correlation decay rank minus open-amount rank correlation argmin decay rank signal.

Formula: sub(rank(ts_decayed_linear(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},5),26.4719),4.58418),7.18088)),rank(ts_decayed_linear(ts_rank(ts_argmin(ts_corr(rank({disk:open}),rank(ts_mean({disk:amount},15)),20.8187),8.62571),6.95668),8.07206)))

Difference between VWAP-amount correlation decay rank and open-amount rank correlation argmin decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_098(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_098: VWAP-amount correlation decay rank minus open-amount rank correlation argmin decay rank.

    Subtracts open-amount rank correlation argmin decay rank from VWAP-amount correlation decay rank.
    """

    default_params = {
        "amount_window1": 5,
        "sum_window": 26,
        "corr_window1": 5,
        "decay_window1": 7,
        "amount_window2": 15,
        "corr_window2": 21,
        "argmin_window": 9,
        "argmin_rank_window": 7,
        "decay_window2": 8,
    }

    @property
    def name(self) -> str:
        return "alpha_101_098"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_098."""
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

        # Part 1: VWAP-amount correlation decay rank
        # ts_mean(amount, 5)
        amount_mean_5 = operator.ts_mean(amount, self.params["amount_window1"])

        # ts_sum(amount_mean_5, 26)
        amount_sum = operator.ts_sum(amount_mean_5, self.params["sum_window"])

        # ts_corr(vwap, amount_sum, 5)
        vwap_corr = operator.ts_corr(vwap, amount_sum, self.params["corr_window1"])

        # ts_decayed_linear(vwap_corr, 7)
        vwap_decayed = operator.ts_decayed_linear(vwap_corr, self.params["decay_window1"])

        # rank(vwap_decayed)
        first_part = operator.rank(vwap_decayed)

        # Part 2: Open-amount rank correlation argmin decay rank
        # rank(open)
        open_rank = operator.rank(open_)

        # ts_mean(amount, 15)
        amount_mean_15 = operator.ts_mean(amount, self.params["amount_window2"])

        # rank(amount_mean_15)
        amount_rank = operator.rank(amount_mean_15)

        # ts_corr(open_rank, amount_rank, 21)
        open_amount_corr = operator.ts_corr(open_rank, amount_rank, self.params["corr_window2"])

        # ts_argmin(open_amount_corr, 9)
        corr_argmin = operator.ts_argmin(open_amount_corr, self.params["argmin_window"])

        # ts_rank(corr_argmin, 7)
        argmin_ranked = operator.ts_rank(corr_argmin, self.params["argmin_rank_window"])

        # ts_decayed_linear(argmin_ranked, 8)
        argmin_decayed = operator.ts_decayed_linear(argmin_ranked, self.params["decay_window2"])

        # rank(argmin_decayed)
        second_part = operator.rank(argmin_decayed)

        # sub(first_part, second_part)
        alpha = operator.sub(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
