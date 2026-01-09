"""Alpha 101_064: Weighted open-low amount correlation vs mid-VWAP delta signal.

Formula: mul(lt(rank(ts_corr(ts_sum(add(mul({disk:open},0.178404),mul({disk:low},sub(1,0.178404))),12.7054),ts_sum(ts_mean({disk:amount},120),12.7054),16.6208)),rank(ts_delta(add(mul(div(add({disk:high},{disk:low}),2),0.178404),mul({disk:vwap},sub(1,0.178404))),3.69741))),-1)

Negative comparison between weighted open-low amount correlation rank and weighted mid-VWAP delta rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_064(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_064: Weighted open-low amount correlation vs mid-VWAP delta.

    Negates the comparison between two price-amount relationship ranks.
    """

    default_params = {
        "weight": 0.178404,
        "sum_window": 13,
        "amount_window": 120,
        "corr_window": 17,
        "delta_window": 4,
    }

    @property
    def name(self) -> str:
        return "alpha_101_064"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_064."""
        open_ = data["open"]
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

        weight = self.params["weight"]

        # Part 1: Weighted open-low amount correlation
        # Weighted open-low: open * weight + low * (1-weight)
        open_part = operator.mul(open_, weight)
        low_part = operator.mul(low, 1 - weight)
        weighted_open_low = operator.add(open_part, low_part)

        # ts_sum(weighted_open_low, 13)
        weighted_sum = operator.ts_sum(weighted_open_low, self.params["sum_window"])

        # ts_mean(amount, 120)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 13)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(weighted_sum, amount_sum, 17)
        first_corr = operator.ts_corr(weighted_sum, amount_sum, self.params["corr_window"])

        # rank(first_corr)
        first_rank = operator.rank(first_corr)

        # Part 2: Weighted mid-VWAP delta
        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # Weighted mid-VWAP: mid_price * weight + vwap * (1-weight)
        mid_part = operator.mul(mid_price, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_mid_vwap = operator.add(mid_part, vwap_part)

        # ts_delta(weighted_mid_vwap, 4)
        price_delta = operator.ts_delta(weighted_mid_vwap, self.params["delta_window"])

        # rank(price_delta)
        second_rank = operator.rank(price_delta)

        # lt(first_rank, second_rank)
        condition = operator.lt(first_rank, second_rank)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
