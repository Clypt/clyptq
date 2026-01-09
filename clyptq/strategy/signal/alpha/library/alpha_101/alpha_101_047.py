"""Alpha 101_047: Complex price-volume-VWAP signal.

Formula: sub(mul(div(mul(rank(div(1,{disk:close})),{disk:volume}),ts_mean({disk:amount},20)),div(mul({disk:high},rank(sub({disk:high},{disk:close}))),div(ts_sum({disk:high},5),5))),rank(sub({disk:vwap},delay({disk:vwap},5))))

Complex factor combining inverse price rank, volume ratio, high price relationships, and VWAP change.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_047(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_047: Complex price-volume-VWAP.

    Combines inverse price rank with volume, high price relationships, minus VWAP change rank.
    """

    default_params = {"amount_window": 20, "high_window": 5, "vwap_delay": 5}

    @property
    def name(self) -> str:
        return "alpha_101_047"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_047."""
        close = data["close"]
        high = data["high"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: rank(1/close) * volume / ts_mean(amount, 20)
        inverse_close = operator.div(1, close)
        inverse_rank = operator.rank(inverse_close)
        rank_volume = operator.mul(inverse_rank, volume)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        first_ratio = operator.div(rank_volume, amount_mean)

        # Part 2: high * rank(high - close) / mean(high, 5)
        high_close_diff = operator.sub(high, close)
        diff_rank = operator.rank(high_close_diff)
        high_rank_product = operator.mul(high, diff_rank)
        high_mean = operator.div(
            operator.ts_sum(high, self.params["high_window"]),
            self.params["high_window"],
        )
        second_ratio = operator.div(high_rank_product, high_mean)

        # Product of first and second parts
        product = operator.mul(first_ratio, second_ratio)

        # Part 3: rank(vwap - delay(vwap, 5))
        vwap_lag = operator.delay(vwap, self.params["vwap_delay"])
        vwap_diff = operator.sub(vwap, vwap_lag)
        vwap_rank = operator.rank(vwap_diff)

        # product - vwap_rank
        alpha = operator.sub(product, vwap_rank)

        return operator.ts_fillna(alpha, 0)
