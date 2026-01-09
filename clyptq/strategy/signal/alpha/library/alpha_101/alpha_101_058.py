"""Alpha 101_058: Demeaned VWAP-volume correlation decay rank signal.

Formula: mul(-1,ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv1}),{disk:volume},3.92795),7.89291),5.50322))

Negative of time-series rank of decayed correlation between demeaned VWAP and volume.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_058(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_058: Demeaned VWAP-volume correlation decay rank.

    Negates the ts_rank of decayed linear correlation between demeaned VWAP and volume.
    """

    default_params = {"corr_window": 4, "decay_window": 8, "rank_window": 6}

    @property
    def name(self) -> str:
        return "alpha_101_058"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_058."""
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Demean VWAP (cross-sectional mean removal as proxy for industry demean)
        vwap_demeaned = operator.demean(vwap)

        # ts_corr(vwap_demeaned, volume, 4)
        corr = operator.ts_corr(vwap_demeaned, volume, self.params["corr_window"])

        # ts_decayed_linear(corr, 8)
        decayed = operator.ts_decayed_linear(corr, self.params["decay_window"])

        # ts_rank(decayed, 6)
        ranked = operator.ts_rank(decayed, self.params["rank_window"])

        # mul(-1, ranked)
        alpha = operator.mul(ranked, -1)

        return operator.ts_fillna(alpha, 0)
