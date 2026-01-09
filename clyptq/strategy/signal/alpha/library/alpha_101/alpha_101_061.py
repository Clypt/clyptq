"""Alpha 101_061: VWAP range vs amount correlation rank signal.

Formula: lt(rank(sub({disk:vwap},ts_min({disk:vwap},16.1219))),rank(ts_corr({disk:vwap},ts_mean({disk:amount},180),17.9282)))

Compares VWAP-min VWAP difference rank with VWAP-amount correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_061(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_061: VWAP range vs amount correlation rank.

    Returns 1 when VWAP range rank is less than VWAP-amount correlation rank.
    """

    default_params = {"vwap_min_window": 16, "amount_window": 180, "corr_window": 18}

    @property
    def name(self) -> str:
        return "alpha_101_061"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_061."""
        volume = data["volume"]
        close = data["close"]
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

        # Part 1: VWAP - min VWAP
        # ts_min(vwap, 16)
        vwap_min = operator.ts_min(vwap, self.params["vwap_min_window"])

        # sub(vwap, vwap_min)
        vwap_diff = operator.sub(vwap, vwap_min)

        # rank(vwap_diff)
        first_rank = operator.rank(vwap_diff)

        # Part 2: VWAP-amount correlation
        # ts_mean(amount, 180)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(vwap, amount_mean, 18)
        corr = operator.ts_corr(vwap, amount_mean, self.params["corr_window"])

        # rank(corr)
        second_rank = operator.rank(corr)

        # lt(first_rank, second_rank)
        alpha = operator.lt(first_rank, second_rank)

        # Convert boolean to float
        alpha = alpha.astype(float)

        return operator.ts_fillna(alpha, 0)
