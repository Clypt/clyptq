"""Alpha 101_076: VWAP delta decay rank vs demeaned low-amount correlation decay rank max signal.

Formula: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},1.24383),11.8259)),ts_rank(ts_decayed_linear(ts_rank(ts_corr(grouped_demean({disk:low},{disk:industry_group_lv1}),ts_mean({disk:amount},81),8.14941),19.569),17.1543),19.383)),-1)

Negative of max between VWAP delta decay rank and demeaned low-amount correlation decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_076(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_076: VWAP delta decay rank vs demeaned low-amount correlation decay rank max.

    Negates the maximum of VWAP delta decay rank and demeaned low-amount correlation decay ts_rank.
    """

    default_params = {
        "delta_window": 1,
        "decay_window1": 12,
        "amount_window": 81,
        "corr_window": 8,
        "rank_window1": 20,
        "decay_window2": 17,
        "rank_window2": 19,
    }

    @property
    def name(self) -> str:
        return "alpha_101_076"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_076."""
        low = data["low"]
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

        # Part 1: VWAP delta decay rank
        # ts_delta(vwap, 1)
        vwap_delta = operator.ts_delta(vwap, self.params["delta_window"])

        # ts_decayed_linear(vwap_delta, 12)
        vwap_decayed = operator.ts_decayed_linear(vwap_delta, self.params["decay_window1"])

        # rank(vwap_decayed)
        first_part = operator.rank(vwap_decayed)

        # Part 2: Demeaned low-amount correlation decay ts_rank
        # Demean low (cross-sectional mean removal)
        low_demeaned = operator.demean(low)

        # ts_mean(amount, 81)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(low_demeaned, amount_mean, 8)
        low_corr = operator.ts_corr(low_demeaned, amount_mean, self.params["corr_window"])

        # ts_rank(low_corr, 20)
        corr_ranked = operator.ts_rank(low_corr, self.params["rank_window1"])

        # ts_decayed_linear(corr_ranked, 17)
        corr_decayed = operator.ts_decayed_linear(corr_ranked, self.params["decay_window2"])

        # ts_rank(corr_decayed, 19)
        second_part = operator.ts_rank(corr_decayed, self.params["rank_window2"])

        # max(first_part, second_part)
        max_result = operator.elem_max(first_part, second_part)

        # mul(max_result, -1)
        alpha = operator.mul(max_result, -1)

        return operator.ts_fillna(alpha, 0)
