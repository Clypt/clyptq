"""Alpha 101_059: Weighted VWAP-volume correlation decay rank signal.

Formula: mul(-1,ts_rank(ts_decayed_linear(ts_corr(grouped_demean(add(mul({disk:vwap},0.728317),mul({disk:vwap},sub(1,0.728317))),{disk:industry_group_lv2}),{disk:volume},4.25197),16.2289),8.19648))

Negative of time-series rank of decayed correlation between weighted demeaned VWAP and volume.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_059(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_059: Weighted VWAP-volume correlation decay rank.

    Negates the ts_rank of decayed linear correlation between weighted demeaned VWAP and volume.
    """

    default_params = {"corr_window": 4, "decay_window": 16, "rank_window": 8}

    @property
    def name(self) -> str:
        return "alpha_101_059"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_059."""
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Weighted VWAP: vwap * 0.728317 + vwap * (1-0.728317) = vwap
        # This simplifies to just vwap
        weighted_vwap = vwap

        # Demean VWAP (cross-sectional mean removal as proxy for industry demean)
        vwap_demeaned = operator.demean(weighted_vwap)

        # ts_corr(vwap_demeaned, volume, 4)
        corr = operator.ts_corr(vwap_demeaned, volume, self.params["corr_window"])

        # ts_decayed_linear(corr, 16)
        decayed = operator.ts_decayed_linear(corr, self.params["decay_window"])

        # ts_rank(decayed, 8)
        ranked = operator.ts_rank(decayed, self.params["rank_window"])

        # mul(-1, ranked)
        alpha = operator.mul(ranked, -1)

        return operator.ts_fillna(alpha, 0)
