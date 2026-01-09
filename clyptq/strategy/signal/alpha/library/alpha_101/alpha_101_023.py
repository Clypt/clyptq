"""Alpha 101_023: High price breakout signal.

Formula: condition(lt(div(ts_sum({disk:high},20),20),{disk:high}),mul(-1,ts_delta({disk:high},2)),0)

Negative high delta when current high exceeds 20-period average, else 0.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_023(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_023: High price breakout.

    Returns negative high change when current high exceeds average, else zero.
    """

    default_params = {"avg_window": 20, "delta_window": 2}

    @property
    def name(self) -> str:
        return "alpha_101_023"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_023."""
        high = data["high"]

        # 20-period high average
        high_mean = operator.div(
            operator.ts_sum(high, self.params["avg_window"]),
            self.params["avg_window"],
        )

        # Condition: high_mean < high
        condition = operator.lt(high_mean, high)

        # ts_delta(high, 2) * -1
        high_delta = operator.ts_delta(high, self.params["delta_window"])
        neg_delta = operator.mul(high_delta, -1)

        # condition(condition, neg_delta, 0)
        alpha = operator.condition(condition, neg_delta, 0)

        return operator.ts_fillna(alpha, 0)
