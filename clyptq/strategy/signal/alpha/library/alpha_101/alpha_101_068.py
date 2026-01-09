"""Alpha 101_068: High-amount correlation ts_rank vs weighted close-low delta signal.

Formula: mul(lt(ts_rank(ts_corr(rank({disk:high}),rank(ts_mean({disk:amount},15)),8.91644),13.9333),rank(ts_delta(add(mul({disk:close},0.518371),mul({disk:low},sub(1,0.518371))),1.06157))),-1)

Negative comparison between high-amount correlation ts_rank and weighted close-low delta rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_068(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_068: High-amount correlation ts_rank vs weighted close-low delta.

    Negates the comparison between high-amount correlation ts_rank and price delta rank.
    """

    default_params = {
        "amount_window": 15,
        "corr_window": 9,
        "rank_window": 14,
        "weight": 0.518371,
        "delta_window": 1,
    }

    @property
    def name(self) -> str:
        return "alpha_101_068"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_068."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: High-amount correlation ts_rank
        # rank(high)
        high_rank = operator.rank(high)

        # ts_mean(amount, 15)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # rank(amount_mean)
        amount_rank = operator.rank(amount_mean)

        # ts_corr(high_rank, amount_rank, 9)
        corr = operator.ts_corr(high_rank, amount_rank, self.params["corr_window"])

        # ts_rank(corr, 14)
        first_part = operator.ts_rank(corr, self.params["rank_window"])

        # Part 2: Weighted close-low delta
        weight = self.params["weight"]

        # Weighted close-low: close * weight + low * (1-weight)
        close_part = operator.mul(close, weight)
        low_part = operator.mul(low, 1 - weight)
        weighted_close_low = operator.add(close_part, low_part)

        # ts_delta(weighted_close_low, 1)
        price_delta = operator.ts_delta(weighted_close_low, self.params["delta_window"])

        # rank(price_delta)
        second_part = operator.rank(price_delta)

        # lt(first_part, second_part)
        condition = operator.lt(first_part, second_part)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
