"""Alpha 101_091: Double decayed close-volume correlation ts_rank minus VWAP-amount correlation decay rank signal.

Formula: mul(sub(ts_rank(ts_decayed_linear(ts_decayed_linear(ts_corr(grouped_demean({disk:close},{disk:industry_group_lv2}),{disk:volume},9.74928),16.398),3.83219),4.8667),rank(ts_decayed_linear(ts_corr({disk:vwap},ts_mean({disk:amount},30),4.01303),2.6809))),-1)

Negative of double decayed demeaned close-volume correlation ts_rank minus VWAP-amount correlation decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_091(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_091: Double decayed close-volume correlation ts_rank minus VWAP-amount correlation decay rank.

    Negates the difference between double decayed demeaned close-volume correlation ts_rank and VWAP-amount correlation decay rank.
    """

    default_params = {
        "corr_window1": 10,
        "decay_window1": 16,
        "decay_window2": 4,
        "rank_window1": 5,
        "amount_window": 30,
        "corr_window2": 4,
        "decay_window3": 3,
    }

    @property
    def name(self) -> str:
        return "alpha_101_091"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_091."""
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

        # Part 1: Double decayed demeaned close-volume correlation ts_rank
        # Demean close (cross-sectional mean removal as proxy for industry demeaning)
        close_demeaned = operator.demean(close)

        # ts_corr(close_demeaned, volume, 10)
        close_vol_corr = operator.ts_corr(close_demeaned, volume, self.params["corr_window1"])

        # ts_decayed_linear(close_vol_corr, 16)
        first_decayed = operator.ts_decayed_linear(close_vol_corr, self.params["decay_window1"])

        # ts_decayed_linear(first_decayed, 4)
        second_decayed = operator.ts_decayed_linear(first_decayed, self.params["decay_window2"])

        # ts_rank(second_decayed, 5)
        first_part = operator.ts_rank(second_decayed, self.params["rank_window1"])

        # Part 2: VWAP-amount correlation decay rank
        # ts_mean(amount, 30)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(vwap, amount_mean, 4)
        vwap_amount_corr = operator.ts_corr(vwap, amount_mean, self.params["corr_window2"])

        # ts_decayed_linear(vwap_amount_corr, 3)
        vwap_decayed = operator.ts_decayed_linear(vwap_amount_corr, self.params["decay_window3"])

        # rank(vwap_decayed)
        second_part = operator.rank(vwap_decayed)

        # sub(first_part, second_part)
        diff = operator.sub(first_part, second_part)

        # mul(diff, -1)
        alpha = operator.mul(diff, -1)

        return operator.ts_fillna(alpha, 0)
