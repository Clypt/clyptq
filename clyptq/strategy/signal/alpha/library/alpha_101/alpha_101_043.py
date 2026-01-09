"""Alpha 101_043: Volume ratio and price delta time-series ranking signal.

Formula: mul(ts_rank(div({disk:volume},ts_mean({disk:amount},20)),20),ts_rank(mul(-1,ts_delta({disk:close},7)),8))

Product of volume ratio rank and negative 7-day price delta rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_043(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_043: Volume ratio with price delta ranking.

    Multiplies volume ratio time-series rank with negative close delta rank.
    """

    default_params = {
        "amount_window": 20,
        "volume_rank_window": 20,
        "delta_window": 7,
        "delta_rank_window": 8,
    }

    @property
    def name(self) -> str:
        return "alpha_101_043"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_043."""
        close = data["close"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # volume / ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        volume_ratio = operator.div(volume, amount_mean)

        # ts_rank(volume_ratio, 20)
        volume_rank = operator.ts_rank(volume_ratio, self.params["volume_rank_window"])

        # -1 * ts_delta(close, 7)
        close_delta = operator.ts_delta(close, self.params["delta_window"])
        neg_delta = operator.mul(close_delta, -1)

        # ts_rank(neg_delta, 8)
        delta_rank = operator.ts_rank(neg_delta, self.params["delta_rank_window"])

        # volume_rank * delta_rank
        alpha = operator.mul(volume_rank, delta_rank)

        return operator.ts_fillna(alpha, 0)
