"""Alpha 101_097: Demeaned weighted low-VWAP delta decay rank minus low-amount ts_rank correlation decay ts_rank signal.

Formula: mul(sub(rank(ts_decayed_linear(ts_delta(grouped_demean(add(mul({disk:low},0.721001),mul({disk:vwap},sub(1,0.721001))),{disk:industry_group_lv2}),3.3705),20.4523)),ts_rank(ts_decayed_linear(ts_rank(ts_corr(ts_rank({disk:low},7.87871),ts_rank(ts_mean({disk:amount},60),17.255),4.97547),18.5925),15.7152),6.71659)),-1)

Negative of demeaned weighted low-VWAP delta decay rank minus low-amount ts_rank correlation decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_097(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_097: Demeaned weighted low-VWAP delta decay rank minus low-amount ts_rank correlation decay ts_rank.

    Negates the difference between demeaned weighted low-VWAP delta decay rank and low-amount ts_rank correlation decay ts_rank.
    """

    default_params = {
        "weight": 0.721001,
        "delta_window": 3,
        "decay_window1": 20,
        "low_rank_window": 8,
        "amount_window": 60,
        "amount_rank_window": 17,
        "corr_window": 5,
        "corr_rank_window": 19,
        "decay_window2": 16,
        "final_rank_window": 7,
    }

    @property
    def name(self) -> str:
        return "alpha_101_097"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_097."""
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

        weight = self.params["weight"]

        # Part 1: Demeaned weighted low-VWAP delta decay rank
        # Weighted low-VWAP: low * weight + vwap * (1-weight)
        weighted_low_vwap = operator.add(
            operator.mul(low, weight),
            operator.mul(vwap, 1 - weight)
        )

        # Demean (cross-sectional mean removal as proxy for industry demeaning)
        weighted_demeaned = operator.demean(weighted_low_vwap)

        # ts_delta(weighted_demeaned, 3)
        weighted_delta = operator.ts_delta(weighted_demeaned, self.params["delta_window"])

        # ts_decayed_linear(weighted_delta, 20)
        delta_decayed = operator.ts_decayed_linear(weighted_delta, self.params["decay_window1"])

        # rank(delta_decayed)
        first_part = operator.rank(delta_decayed)

        # Part 2: Low-amount ts_rank correlation decay ts_rank
        # ts_rank(low, 8)
        low_tsrank = operator.ts_rank(low, self.params["low_rank_window"])

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 17)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(low_tsrank, amount_tsrank, 5)
        low_amount_corr = operator.ts_corr(low_tsrank, amount_tsrank, self.params["corr_window"])

        # ts_rank(low_amount_corr, 19)
        corr_ranked = operator.ts_rank(low_amount_corr, self.params["corr_rank_window"])

        # ts_decayed_linear(corr_ranked, 16)
        corr_decayed = operator.ts_decayed_linear(corr_ranked, self.params["decay_window2"])

        # ts_rank(corr_decayed, 7)
        second_part = operator.ts_rank(corr_decayed, self.params["final_rank_window"])

        # sub(first_part, second_part)
        diff = operator.sub(first_part, second_part)

        # mul(diff, -1)
        alpha = operator.mul(diff, -1)

        return operator.ts_fillna(alpha, 0)
