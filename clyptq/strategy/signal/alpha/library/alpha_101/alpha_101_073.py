"""Alpha 101_073: VWAP delta decay rank vs weighted price change rate decay rank max signal.

Formula: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},4.72775),2.91864)),ts_rank(ts_decayed_linear(mul(div(ts_delta(add(mul({disk:open},0.147155),mul({disk:low},sub(1,0.147155))),2.03608),add(mul({disk:open},0.147155),mul({disk:low},sub(1,0.147155)))),-1),3.33829),16.7411)),-1)

Negative of max between VWAP delta decay rank and weighted open-low change rate decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_073(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_073: VWAP delta decay rank vs weighted price change rate decay rank max.

    Negates the maximum of VWAP delta decay rank and weighted price change rate decay ts_rank.
    """

    default_params = {
        "vwap_delta_window": 5,
        "decay_window1": 3,
        "weight": 0.147155,
        "price_delta_window": 2,
        "decay_window2": 3,
        "rank_window": 17,
    }

    @property
    def name(self) -> str:
        return "alpha_101_073"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_073."""
        open_ = data["open"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        weight = self.params["weight"]

        # Part 1: VWAP delta decay rank
        # ts_delta(vwap, 5)
        vwap_delta = operator.ts_delta(vwap, self.params["vwap_delta_window"])

        # ts_decayed_linear(vwap_delta, 3)
        vwap_decayed = operator.ts_decayed_linear(vwap_delta, self.params["decay_window1"])

        # rank(vwap_decayed)
        first_part = operator.rank(vwap_decayed)

        # Part 2: Weighted open-low change rate decay ts_rank
        # Weighted open-low: open * weight + low * (1-weight)
        open_part = operator.mul(open_, weight)
        low_part = operator.mul(low, 1 - weight)
        weighted_open_low = operator.add(open_part, low_part)

        # ts_delta(weighted_open_low, 2)
        weighted_delta = operator.ts_delta(weighted_open_low, self.params["price_delta_window"])

        # div(weighted_delta, weighted_open_low) - change rate
        change_rate = operator.div(weighted_delta, weighted_open_low)

        # mul(change_rate, -1)
        neg_change_rate = operator.mul(change_rate, -1)

        # ts_decayed_linear(neg_change_rate, 3)
        rate_decayed = operator.ts_decayed_linear(neg_change_rate, self.params["decay_window2"])

        # ts_rank(rate_decayed, 17)
        second_part = operator.ts_rank(rate_decayed, self.params["rank_window"])

        # max(first_part, second_part)
        max_result = operator.elem_max(first_part, second_part)

        # mul(max_result, -1)
        alpha = operator.mul(max_result, -1)

        return operator.ts_fillna(alpha, 0)
