"""Alpha 101_077: Price difference decay rank vs mid-amount correlation decay rank min signal.

Formula: min(rank(ts_decayed_linear(sub(add(div(add({disk:high},{disk:low}),2),{disk:high}),add({disk:vwap},{disk:high})),20.0451)),rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:amount},40),3.1614),5.64125)))

Minimum of price combination difference decay rank and mid price-amount correlation decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_077(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_077: Price difference decay rank vs mid-amount correlation decay rank min.

    Returns the minimum of price combination difference decay rank and mid-amount correlation decay rank.
    """

    default_params = {
        "decay_window1": 20,
        "amount_window": 40,
        "corr_window": 3,
        "decay_window2": 6,
    }

    @property
    def name(self) -> str:
        return "alpha_101_077"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_077."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(high, low, close, volume)

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Price difference decay rank
        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # mid_price + high
        mid_high_sum = operator.add(mid_price, high)

        # vwap + high
        vwap_high_sum = operator.add(vwap, high)

        # (mid_price + high) - (vwap + high)
        price_diff = operator.sub(mid_high_sum, vwap_high_sum)

        # ts_decayed_linear(price_diff, 20)
        price_decayed = operator.ts_decayed_linear(price_diff, self.params["decay_window1"])

        # rank(price_decayed)
        first_rank = operator.rank(price_decayed)

        # Part 2: Mid price-amount correlation decay rank
        # ts_mean(amount, 40)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(mid_price, amount_mean, 3)
        corr = operator.ts_corr(mid_price, amount_mean, self.params["corr_window"])

        # ts_decayed_linear(corr, 6)
        corr_decayed = operator.ts_decayed_linear(corr, self.params["decay_window2"])

        # rank(corr_decayed)
        second_rank = operator.rank(corr_decayed)

        # min(first_rank, second_rank)
        alpha = operator.elem_min(first_rank, second_rank)

        return operator.ts_fillna(alpha, 0)
