"""Alpha 101_017: Complex close-volume momentum signal.

Formula: mul(mul(mul(-1,rank(ts_rank({disk:close},10))),rank(ts_delta(ts_delta({disk:close},1),1))),rank(ts_rank(div({disk:volume},ts_mean({disk:amount},20)),5)))

Complex factor combining close ranking, second derivative, and volume ratio.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_017(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_017: Complex close-volume momentum.

    Combines close time-series rank, close acceleration, and volume-to-amount ratio.
    """

    default_params = {
        "ts_rank_window1": 10,
        "ts_rank_window2": 5,
        "amount_mean_window": 20,
    }

    @property
    def name(self) -> str:
        return "alpha_101_017"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_017."""
        close = data["close"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # rank(ts_rank(close, 10)) * -1
        close_ts_rank = operator.ts_rank(close, self.params["ts_rank_window1"])
        close_rank = operator.rank(close_ts_rank)
        neg_close_rank = operator.mul(close_rank, -1)

        # rank(ts_delta(ts_delta(close, 1), 1))
        close_delta1 = operator.ts_delta(close, 1)
        close_delta2 = operator.ts_delta(close_delta1, 1)
        delta_rank = operator.rank(close_delta2)

        # rank(ts_rank(volume / ts_mean(amount, 20), 5))
        amount_mean = operator.ts_mean(amount, self.params["amount_mean_window"])
        volume_ratio = operator.div(volume, amount_mean)
        volume_ts_rank = operator.ts_rank(volume_ratio, self.params["ts_rank_window2"])
        volume_rank = operator.rank(volume_ts_rank)

        # Multiply all components
        alpha = operator.mul(operator.mul(neg_close_rank, delta_rank), volume_rank)

        return operator.ts_fillna(alpha, 0)
