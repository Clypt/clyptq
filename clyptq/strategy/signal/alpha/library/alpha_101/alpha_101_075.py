"""Alpha 101_075: VWAP-volume correlation rank vs low-amount rank correlation rank signal.

Formula: lt(rank(ts_corr({disk:vwap},{disk:volume},4.24304)),rank(ts_corr(rank({disk:low}),rank(ts_mean({disk:amount},50)),12.4413)))

Compares VWAP-volume correlation rank with low-amount rank correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_075(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_075: VWAP-volume correlation rank vs low-amount rank correlation rank.

    Returns 1 when VWAP-volume correlation rank is less than low-amount correlation rank.
    """

    default_params = {"corr_window1": 4, "amount_window": 50, "corr_window2": 12}

    @property
    def name(self) -> str:
        return "alpha_101_075"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_075."""
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

        # Part 1: VWAP-volume correlation
        # ts_corr(vwap, volume, 4)
        vwap_corr = operator.ts_corr(vwap, volume, self.params["corr_window1"])

        # rank(vwap_corr)
        first_rank = operator.rank(vwap_corr)

        # Part 2: Low-amount rank correlation
        # rank(low)
        low_rank = operator.rank(low)

        # ts_mean(amount, 50)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # rank(amount_mean)
        amount_rank = operator.rank(amount_mean)

        # ts_corr(low_rank, amount_rank, 12)
        low_corr = operator.ts_corr(low_rank, amount_rank, self.params["corr_window2"])

        # rank(low_corr)
        second_rank = operator.rank(low_corr)

        # lt(first_rank, second_rank)
        alpha = operator.lt(first_rank, second_rank)

        # Convert boolean to float
        alpha = alpha.astype(float)

        return operator.ts_fillna(alpha, 0)
