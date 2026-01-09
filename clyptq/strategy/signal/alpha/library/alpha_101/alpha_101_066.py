"""Alpha 101_066: VWAP delta decay rank plus low-VWAP ratio ts_rank signal.

Formula: mul(add(rank(ts_decayed_linear(ts_delta({disk:vwap},3.51013),7.23052)),ts_rank(ts_decayed_linear(div(sub(add(mul({disk:low},0.96633),mul({disk:low},sub(1,0.96633))),{disk:vwap}),sub({disk:open},div(add({disk:high},{disk:low}),2))),11.4157),6.72611)),-1)

Negative of sum of VWAP delta decay rank and low-VWAP ratio decay ts_rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_066(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_066: VWAP delta decay rank plus low-VWAP ratio ts_rank.

    Negates the sum of VWAP delta decay rank and price ratio decay ts_rank.
    """

    default_params = {
        "delta_window": 4,
        "decay_window1": 7,
        "decay_window2": 11,
        "rank_window": 7,
    }

    @property
    def name(self) -> str:
        return "alpha_101_066"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_066."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(high, low, close, volume)

        # Part 1: VWAP delta decay rank
        # ts_delta(vwap, 4)
        vwap_delta = operator.ts_delta(vwap, self.params["delta_window"])

        # ts_decayed_linear(vwap_delta, 7)
        vwap_decayed = operator.ts_decayed_linear(vwap_delta, self.params["decay_window1"])

        # rank(vwap_decayed)
        first_part = operator.rank(vwap_decayed)

        # Part 2: Low-VWAP ratio decay ts_rank
        # Weighted low: low * 0.96633 + low * (1-0.96633) = low
        weighted_low = low

        # sub(weighted_low, vwap)
        low_vwap_diff = operator.sub(weighted_low, vwap)

        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # sub(open, mid_price)
        open_mid_diff = operator.sub(open_, mid_price)

        # div(low_vwap_diff, open_mid_diff)
        ratio = operator.div(low_vwap_diff, open_mid_diff)

        # ts_decayed_linear(ratio, 11)
        ratio_decayed = operator.ts_decayed_linear(ratio, self.params["decay_window2"])

        # ts_rank(ratio_decayed, 7)
        second_part = operator.ts_rank(ratio_decayed, self.params["rank_window"])

        # add(first_part, second_part)
        sum_parts = operator.add(first_part, second_part)

        # mul(sum_parts, -1)
        alpha = operator.mul(sum_parts, -1)

        return operator.ts_fillna(alpha, 0)
