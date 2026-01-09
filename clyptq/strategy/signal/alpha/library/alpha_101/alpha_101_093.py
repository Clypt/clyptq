"""Alpha 101_093: Demeaned VWAP-amount correlation decay ts_rank divided by weighted close-VWAP delta decay rank signal.

Formula: div(ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv2}),ts_mean({disk:amount},81),17.4193),19.848),7.54455),rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.524434),mul({disk:vwap},sub(1,0.524434))),2.77377),16.2664)))

Ratio of demeaned VWAP-amount correlation decay ts_rank to weighted close-VWAP delta decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_093(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_093: Demeaned VWAP-amount correlation decay ts_rank divided by weighted close-VWAP delta decay rank.

    Divides demeaned VWAP-amount correlation decay ts_rank by weighted close-VWAP delta decay rank.
    """

    default_params = {
        "amount_window": 81,
        "corr_window": 17,
        "decay_window1": 20,
        "rank_window": 8,
        "weight": 0.524434,
        "delta_window": 3,
        "decay_window2": 16,
    }

    @property
    def name(self) -> str:
        return "alpha_101_093"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_093."""
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

        # Numerator: Demeaned VWAP-amount correlation decay ts_rank
        # Demean VWAP (cross-sectional mean removal as proxy for industry demeaning)
        vwap_demeaned = operator.demean(vwap)

        # ts_mean(amount, 81)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(vwap_demeaned, amount_mean, 17)
        vwap_amount_corr = operator.ts_corr(vwap_demeaned, amount_mean, self.params["corr_window"])

        # ts_decayed_linear(vwap_amount_corr, 20)
        corr_decayed = operator.ts_decayed_linear(vwap_amount_corr, self.params["decay_window1"])

        # ts_rank(corr_decayed, 8)
        numerator = operator.ts_rank(corr_decayed, self.params["rank_window"])

        # Denominator: Weighted close-VWAP delta decay rank
        weight = self.params["weight"]

        # Weighted close-VWAP: close * weight + vwap * (1-weight)
        weighted_close_vwap = operator.add(
            operator.mul(close, weight),
            operator.mul(vwap, 1 - weight)
        )

        # ts_delta(weighted_close_vwap, 3)
        weighted_delta = operator.ts_delta(weighted_close_vwap, self.params["delta_window"])

        # ts_decayed_linear(weighted_delta, 16)
        delta_decayed = operator.ts_decayed_linear(weighted_delta, self.params["decay_window2"])

        # rank(delta_decayed)
        denominator = operator.rank(delta_decayed)

        # div(numerator, denominator)
        alpha = operator.div(numerator, denominator)

        return operator.ts_fillna(alpha, 0)
