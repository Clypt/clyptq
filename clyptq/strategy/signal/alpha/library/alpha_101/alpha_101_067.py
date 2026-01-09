"""Alpha 101_067: High range rank power by demeaned VWAP-amount correlation rank signal.

Formula: mul(pow(rank(sub({disk:high},ts_min({disk:high},2.14593))),rank(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv1}),grouped_demean(ts_mean({disk:amount},20),{disk:industry_group_lv3}),6.02936))),-1)

Negative of high range rank raised to demeaned VWAP-amount correlation rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_067(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_067: High range rank power by demeaned VWAP-amount correlation rank.

    Negates the high range rank raised to the power of demeaned VWAP-amount correlation rank.
    """

    default_params = {"high_min_window": 2, "amount_window": 20, "corr_window": 6}

    @property
    def name(self) -> str:
        return "alpha_101_067"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_067."""
        high = data["high"]
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

        # Part 1: High range
        # ts_min(high, 2)
        high_min = operator.ts_min(high, self.params["high_min_window"])

        # sub(high, high_min)
        high_diff = operator.sub(high, high_min)

        # rank(high_diff)
        base_rank = operator.rank(high_diff)

        # Part 2: Demeaned VWAP-amount correlation
        # Demean VWAP (cross-sectional mean removal)
        vwap_demeaned = operator.demean(vwap)

        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # Demean amount_mean (cross-sectional mean removal)
        amount_demeaned = operator.demean(amount_mean)

        # ts_corr(vwap_demeaned, amount_demeaned, 6)
        corr = operator.ts_corr(vwap_demeaned, amount_demeaned, self.params["corr_window"])

        # rank(corr)
        power_rank = operator.rank(corr)

        # pow(base_rank, power_rank)
        powered = operator.pow(base_rank, power_rank)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
