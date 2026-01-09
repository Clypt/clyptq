"""Alpha 101_094: VWAP-min VWAP difference rank power by VWAP-amount ts_rank correlation ts_rank signal.

Formula: mul(pow(rank(sub({disk:vwap},ts_min({disk:vwap},11.5783))),ts_rank(ts_corr(ts_rank({disk:vwap},19.6462),ts_rank(ts_mean({disk:amount},60),4.02992),18.0926),2.70756)),-1)

Negative of VWAP-min VWAP difference rank raised to VWAP-amount ts_rank correlation ts_rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_094(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_094: VWAP-min VWAP difference rank power by VWAP-amount ts_rank correlation ts_rank.

    Negates the power of VWAP-min VWAP difference rank by VWAP-amount ts_rank correlation ts_rank.
    """

    default_params = {
        "min_window": 12,
        "vwap_rank_window": 20,
        "amount_window": 60,
        "amount_rank_window": 4,
        "corr_window": 18,
        "final_rank_window": 3,
    }

    @property
    def name(self) -> str:
        return "alpha_101_094"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_094."""
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

        # Base: VWAP - min VWAP difference rank
        # ts_min(vwap, 12)
        vwap_min = operator.ts_min(vwap, self.params["min_window"])

        # sub(vwap, vwap_min)
        vwap_diff = operator.sub(vwap, vwap_min)

        # rank(vwap_diff)
        base = operator.rank(vwap_diff)

        # Exponent: VWAP-amount ts_rank correlation ts_rank
        # ts_rank(vwap, 20)
        vwap_tsrank = operator.ts_rank(vwap, self.params["vwap_rank_window"])

        # ts_mean(amount, 60)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 4)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(vwap_tsrank, amount_tsrank, 18)
        corr_result = operator.ts_corr(vwap_tsrank, amount_tsrank, self.params["corr_window"])

        # ts_rank(corr_result, 3)
        power = operator.ts_rank(corr_result, self.params["final_rank_window"])

        # pow(base, power)
        powered = operator.pow(base, power)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
