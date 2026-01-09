"""Alpha 101_030: Price direction pattern with volume ratio signal.

Formula: div(mul(sub(1.0,rank(add(add(sign(sub({disk:close},delay({disk:close},1))),sign(sub(delay({disk:close},1),delay({disk:close},2)))),sign(sub(delay({disk:close},2),delay({disk:close},3)))))),ts_sum({disk:volume},5)),ts_sum({disk:volume},20))

Combines 3-day price direction pattern ranking with volume ratio.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_030(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_030: Price direction pattern with volume ratio.

    Uses 3-day consecutive price direction signs with short-to-long volume ratio.
    """

    default_params = {"short_volume_window": 5, "long_volume_window": 20}

    @property
    def name(self) -> str:
        return "alpha_101_030"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_030."""
        close = data["close"]
        volume = data["volume"]

        # Delayed closes
        close_lag1 = operator.delay(close, 1)
        close_lag2 = operator.delay(close, 2)
        close_lag3 = operator.delay(close, 3)

        # Direction signs for each period
        sign1 = operator.sign(operator.sub(close, close_lag1))
        sign2 = operator.sign(operator.sub(close_lag1, close_lag2))
        sign3 = operator.sign(operator.sub(close_lag2, close_lag3))

        # Sum of all direction signs
        sign_sum = operator.add(operator.add(sign1, sign2), sign3)

        # rank(sign_sum)
        sign_rank = operator.rank(sign_sum)

        # 1.0 - sign_rank
        one_minus_rank = operator.sub(1.0, sign_rank)

        # ts_sum(volume, 5)
        volume_sum_5 = operator.ts_sum(volume, self.params["short_volume_window"])

        # ts_sum(volume, 20)
        volume_sum_20 = operator.ts_sum(volume, self.params["long_volume_window"])

        # (1 - rank) * volume_sum_5 / volume_sum_20
        numerator = operator.mul(one_minus_rank, volume_sum_5)
        alpha = operator.div(numerator, volume_sum_20)

        return operator.ts_fillna(alpha, 0)
