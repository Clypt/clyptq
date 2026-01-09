"""Alpha 101_087: Weighted close-VWAP delta decay rank vs demeaned amount-close correlation abs decay rank max signal.

Formula: mul(max(rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.369701),mul({disk:vwap},sub(1,0.369701))),1.91233),2.65461)),ts_rank(ts_decayed_linear(abs(ts_corr(grouped_demean(ts_mean({disk:amount},81),{disk:industry_group_lv2}),{disk:close},13.4132)),4.89768),14.4535)),-1)

Negative of max between weighted close-VWAP delta decay rank and demeaned amount-close correlation abs decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_087(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_087: Weighted close-VWAP delta decay rank vs demeaned amount-close correlation abs decay rank max.

    Negates the maximum of two complex decay rank values.
    """

    default_params = {
        "weight": 0.369701,
        "delta_window": 2,
        "decay_window1": 3,
        "amount_window": 81,
        "corr_window": 13,
        "decay_window2": 5,
        "rank_window": 14,
    }

    @property
    def name(self) -> str:
        return "alpha_101_087"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_087."""
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

        # Part 1: Weighted close-VWAP delta decay rank
        # Weighted close-VWAP: close * weight + vwap * (1-weight)
        close_part = operator.mul(close, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_close_vwap = operator.add(close_part, vwap_part)

        # ts_delta(weighted_close_vwap, 2)
        weighted_delta = operator.ts_delta(weighted_close_vwap, self.params["delta_window"])

        # ts_decayed_linear(weighted_delta, 3)
        delta_decayed = operator.ts_decayed_linear(weighted_delta, self.params["decay_window1"])

        # rank(delta_decayed)
        first_part = operator.rank(delta_decayed)

        # Part 2: Demeaned amount-close correlation abs decay ts_rank
        # ts_mean(amount, 81)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # Demean amount_mean (cross-sectional mean removal)
        amount_demeaned = operator.demean(amount_mean)

        # ts_corr(amount_demeaned, close, 13)
        amount_corr = operator.ts_corr(amount_demeaned, close, self.params["corr_window"])

        # abs(amount_corr)
        abs_corr = operator.abs(amount_corr)

        # ts_decayed_linear(abs_corr, 5)
        corr_decayed = operator.ts_decayed_linear(abs_corr, self.params["decay_window2"])

        # ts_rank(corr_decayed, 14)
        second_part = operator.ts_rank(corr_decayed, self.params["rank_window"])

        # max(first_part, second_part)
        max_result = operator.elem_max(first_part, second_part)

        # mul(max_result, -1)
        alpha = operator.mul(max_result, -1)

        return operator.ts_fillna(alpha, 0)
