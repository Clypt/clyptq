"""Alpha 101_020: Opening gap ranking signal.

Formula: mul(mul(mul(-1,rank(sub({disk:open},delay({disk:high},1)))),rank(sub({disk:open},delay({disk:close},1)))),rank(sub({disk:open},delay({disk:low},1))))

Combines rankings of opening gaps from previous high, close, and low.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_020(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_020: Opening gap ranking.

    Multiplies ranked differences between open and previous day's high, close, and low.
    """

    default_params = {"delay_window": 1}

    @property
    def name(self) -> str:
        return "alpha_101_020"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_020."""
        open_ = data["open"]
        high = data["high"]
        close = data["close"]
        low = data["low"]

        delay = self.params["delay_window"]

        # rank(open - delay(high, 1)) * -1
        high_lag = operator.delay(high, delay)
        open_high_diff = operator.sub(open_, high_lag)
        open_high_rank = operator.rank(open_high_diff)
        neg_open_high_rank = operator.mul(open_high_rank, -1)

        # rank(open - delay(close, 1))
        close_lag = operator.delay(close, delay)
        open_close_diff = operator.sub(open_, close_lag)
        open_close_rank = operator.rank(open_close_diff)

        # rank(open - delay(low, 1))
        low_lag = operator.delay(low, delay)
        open_low_diff = operator.sub(open_, low_lag)
        open_low_rank = operator.rank(open_low_diff)

        # Multiply all components
        alpha = operator.mul(
            operator.mul(neg_open_high_rank, open_close_rank), open_low_rank
        )

        return operator.ts_fillna(alpha, 0)
