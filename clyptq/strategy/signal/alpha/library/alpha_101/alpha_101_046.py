"""Alpha 101_046: Multi-period slope comparison signal.

Formula: condition(lt(0.25,sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10))),mul(-1,1),condition(lt(sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10)),0),1,mul(mul(-1,1),sub({disk:close},delay({disk:close},1)))))

Conditional factor based on comparison of multi-period price change slopes.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_046(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_046: Multi-period slope comparison.

    Returns conditional signal based on the difference between 10-20 day and 0-10 day slopes.
    """

    default_params = {}

    @property
    def name(self) -> str:
        return "alpha_101_046"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_046."""
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

        # Condition 1: slope_diff > 0.25
        condition1 = operator.lt(0.25, slope_diff)

        # Condition 2: slope_diff < 0
        condition2 = operator.lt(slope_diff, 0)

        # close - delay(close, 1)
        daily_change = operator.sub(close, close_lag1)
        neg_daily_change = operator.mul(daily_change, -1)

        # Nested conditions
        inner_condition = operator.condition(condition2, 1, neg_daily_change)
        alpha = operator.condition(condition1, -1, inner_condition)

        return operator.ts_fillna(alpha, 0)
