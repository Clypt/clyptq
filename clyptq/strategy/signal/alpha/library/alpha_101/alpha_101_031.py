"""Alpha 101_031: Multi-ranking decay with amount-low correlation signal.

Formula: add(add(rank(rank(rank(ts_decayed_linear(mul(-1,rank(rank(ts_delta({disk:close},10)))),10)))),rank(mul(-1,ts_delta({disk:close},3)))),sign(twise_a_scale(ts_corr(ts_mean({disk:amount},20),{disk:low},12))))

Complex factor combining multiple close delta rankings, decay, and amount-low correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_031(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_031: Multi-ranking decay with correlation.

    Combines triple-ranked decayed close delta with short-term delta and amount-low correlation.
    """

    default_params = {
        "long_delta": 10,
        "short_delta": 3,
        "decay_window": 10,
        "amount_window": 20,
        "corr_window": 12,
    }

    @property
    def name(self) -> str:
        return "alpha_101_031"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_031."""
        close = data["close"]
        low = data["low"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Complex ranking chain
        # ts_delta(close, 10) -> rank -> rank -> mul(-1) -> decay -> rank -> rank -> rank
        close_delta10 = operator.ts_delta(close, self.params["long_delta"])
        rank1 = operator.rank(close_delta10)
        rank2 = operator.rank(rank1)
        neg_rank = operator.mul(rank2, -1)
        decayed = operator.ts_decayed_linear(neg_rank, self.params["decay_window"])
        rank3 = operator.rank(decayed)
        rank4 = operator.rank(rank3)
        first_part = operator.rank(rank4)

        # Part 2: Short-term delta
        close_delta3 = operator.ts_delta(close, self.params["short_delta"])
        neg_delta3 = operator.mul(close_delta3, -1)
        second_part = operator.rank(neg_delta3)

        # Part 3: Amount-low correlation
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        corr = operator.ts_corr(amount_mean, low, self.params["corr_window"])
        scaled_corr = operator.twise_a_scale(corr, 1)
        third_part = operator.sign(scaled_corr)

        # Sum all parts
        alpha = operator.add(operator.add(first_part, second_part), third_part)

        return operator.ts_fillna(alpha, 0)
