"""Alpha 101_039: Price delta with decayed volume ratio signal.

Formula: mul(mul(-1,rank(mul(ts_delta({disk:close},7),sub(1,rank(ts_decayed_linear(div({disk:volume},ts_mean({disk:amount},20)),9)))))),add(1,rank(ts_sum({disk:returns},250))))

Complex factor combining 7-day price delta, decayed volume ratio, and long-term returns.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_039(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_039: Price delta with decayed volume ratio.

    Combines price delta, decayed volume ratio ranking, and long-term cumulative returns.
    """

    default_params = {
        "delta_window": 7,
        "amount_window": 20,
        "decay_window": 9,
        "returns_window": 250,
    }

    @property
    def name(self) -> str:
        return "alpha_101_039"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_039."""
        close = data["close"]
        volume = data["volume"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # ts_delta(close, 7)
        close_delta = operator.ts_delta(close, self.params["delta_window"])

        # volume / ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        volume_ratio = operator.div(volume, amount_mean)

        # ts_decayed_linear(volume_ratio, 9)
        decayed_ratio = operator.ts_decayed_linear(
            volume_ratio, self.params["decay_window"]
        )

        # 1 - rank(decayed_ratio)
        ratio_rank = operator.rank(decayed_ratio)
        one_minus_rank = operator.sub(1, ratio_rank)

        # close_delta * one_minus_rank
        product = operator.mul(close_delta, one_minus_rank)

        # -1 * rank(product)
        product_rank = operator.rank(product)
        neg_rank = operator.mul(product_rank, -1)

        # 1 + rank(ts_sum(returns, 250))
        returns_sum = operator.ts_sum(returns, self.params["returns_window"])
        returns_rank = operator.rank(returns_sum)
        returns_plus1 = operator.add(1, returns_rank)

        # neg_rank * returns_plus1
        alpha = operator.mul(neg_rank, returns_plus1)

        return operator.ts_fillna(alpha, 0)
