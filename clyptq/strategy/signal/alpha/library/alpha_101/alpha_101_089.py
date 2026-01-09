"""Alpha 101_089: Weighted low-amount correlation decay rank minus demeaned VWAP delta decay rank signal.

Formula: sub(ts_rank(ts_decayed_linear(ts_corr(add(mul({disk:low},0.967285),mul({disk:low},sub(1,0.967285))),ts_mean({disk:amount},10),6.94279),5.51607),3.79744),ts_rank(ts_decayed_linear(ts_delta(grouped_demean({disk:vwap},{disk:industry_group_lv2}),3.48158),10.1466),15.3012))

Difference between low-amount correlation decay ts_rank and demeaned VWAP delta decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_089(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_089: Weighted low-amount correlation decay rank minus demeaned VWAP delta decay rank.

    Subtracts demeaned VWAP delta decay ts_rank from low-amount correlation decay ts_rank.
    """

    default_params = {
        "amount_window": 10,
        "corr_window": 7,
        "decay_window1": 6,
        "rank_window1": 4,
        "delta_window": 3,
        "decay_window2": 10,
        "rank_window2": 15,
    }

    @property
    def name(self) -> str:
        return "alpha_101_089"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_089."""
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

        # Part 1: Low-amount correlation decay ts_rank
        # Weighted low: low * 0.967285 + low * (1-0.967285) = low
        weighted_low = low

        # ts_mean(amount, 10)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(weighted_low, amount_mean, 7)
        low_corr = operator.ts_corr(weighted_low, amount_mean, self.params["corr_window"])

        # ts_decayed_linear(low_corr, 6)
        low_decayed = operator.ts_decayed_linear(low_corr, self.params["decay_window1"])

        # ts_rank(low_decayed, 4)
        first_part = operator.ts_rank(low_decayed, self.params["rank_window1"])

        # Part 2: Demeaned VWAP delta decay ts_rank
        # Demean VWAP (cross-sectional mean removal)
        vwap_demeaned = operator.demean(vwap)

        # ts_delta(vwap_demeaned, 3)
        vwap_delta = operator.ts_delta(vwap_demeaned, self.params["delta_window"])

        # ts_decayed_linear(vwap_delta, 10)
        vwap_decayed = operator.ts_decayed_linear(vwap_delta, self.params["decay_window2"])

        # ts_rank(vwap_decayed, 15)
        second_part = operator.ts_rank(vwap_decayed, self.params["rank_window2"])

        # sub(first_part, second_part)
        alpha = operator.sub(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
