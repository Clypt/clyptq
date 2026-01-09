"""Alpha 101_051: Slope comparison with threshold signal.

Formula: condition(lt(sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10)),mul(-1,0.05)),1,mul(mul(-1,1),sub({disk:close},delay({disk:close},1))))

Conditional factor based on multi-period slope comparison with -0.05 threshold.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_051(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_051: Slope comparison with threshold.

    Returns 1 when slope difference is below -0.05, else negative daily change.
    """

    default_params = {"threshold": -0.05}

    @property
    def name(self) -> str:
        return "alpha_101_051"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_051."""
        close = data["close"]

        # Delayed closes
        close_lag20 = operator.delay(close, 20)
        close_lag10 = operator.delay(close, 10)
        close_lag1 = operator.delay(close, 1)

        # First slope: (close_lag20 - close_lag10) / 10
        slope1_num = operator.sub(close_lag20, close_lag10)
        slope1 = operator.div(slope1_num, 10)

        # Second slope: (close_lag10 - close) / 10
        slope2_num = operator.sub(close_lag10, close)
        slope2 = operator.div(slope2_num, 10)

        # Slope difference
        slope_diff = operator.sub(slope1, slope2)

        # Condition: slope_diff < -0.05
        condition = operator.lt(slope_diff, self.params["threshold"])

        # -(close - delay(close, 1))
        daily_change = operator.sub(close, close_lag1)
        neg_daily_change = operator.mul(daily_change, -1)

        # Conditional
        alpha = operator.condition(condition, 1, neg_daily_change)

        return operator.ts_fillna(alpha, 0)
