"""Alpha 101_074: Close-amount correlation rank vs weighted high-VWAP volume correlation rank signal.

Formula: mul(lt(rank(ts_corr({disk:close},ts_sum(ts_mean({disk:amount},30),37.4843),15.1365)),rank(ts_corr(rank(add(mul({disk:high},0.0261661),mul({disk:vwap},sub(1,0.0261661)))),rank({disk:volume}),11.4791))),-1)

Negative comparison between close-amount correlation rank and weighted high-VWAP volume correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_074(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_074: Close-amount correlation rank vs weighted high-VWAP volume correlation rank.

    Negates the comparison between two correlation rank values.
    """

    default_params = {
        "amount_window": 30,
        "sum_window": 37,
        "corr_window1": 15,
        "weight": 0.0261661,
        "corr_window2": 11,
    }

    @property
    def name(self) -> str:
        return "alpha_101_074"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_074."""
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

        # Part 1: Close-amount correlation
        # ts_mean(amount, 30)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 37)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(close, amount_sum, 15)
        close_corr = operator.ts_corr(close, amount_sum, self.params["corr_window1"])

        # rank(close_corr)
        first_rank = operator.rank(close_corr)

        # Part 2: Weighted high-VWAP volume correlation
        weight = self.params["weight"]

        # Weighted high-VWAP: high * weight + vwap * (1-weight)
        high_part = operator.mul(high, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_high_vwap = operator.add(high_part, vwap_part)

        # rank(weighted_high_vwap)
        weighted_rank = operator.rank(weighted_high_vwap)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(weighted_rank, volume_rank, 11)
        weighted_corr = operator.ts_corr(weighted_rank, volume_rank, self.params["corr_window2"])

        # rank(weighted_corr)
        second_rank = operator.rank(weighted_corr)

        # lt(first_rank, second_rank)
        condition = operator.lt(first_rank, second_rank)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
