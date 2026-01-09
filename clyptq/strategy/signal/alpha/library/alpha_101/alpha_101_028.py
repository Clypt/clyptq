"""Alpha 101_028: Amount-low correlation with mid-price signal.

Formula: twise_a_scale(sub(add(ts_corr(ts_mean({disk:amount},20),{disk:low},5),div(add({disk:high},{disk:low}),2)),{disk:close}),1)

Scaled factor combining amount-low correlation, mid-price, and close.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_028(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_028: Amount-low correlation with mid-price.

    Combines average amount-low correlation with mid-price minus close, then scales.
    """

    default_params = {"amount_window": 20, "corr_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_028"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_028."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(amount_mean, low, 5)
        corr = operator.ts_corr(amount_mean, low, self.params["corr_window"])

        # (high + low) / 2 - mid price
        mid_price = operator.div(operator.add(high, low), 2)

        # corr + mid_price - close
        sum_part = operator.add(corr, mid_price)
        diff = operator.sub(sum_part, close)

        # twise_a_scale(diff, 1)
        alpha = operator.twise_a_scale(diff, 1)

        return operator.ts_fillna(alpha, 0)
