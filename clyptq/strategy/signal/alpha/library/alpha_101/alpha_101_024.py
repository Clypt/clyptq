"""Alpha 101_024: Long-term average change rate signal.

Formula: condition(or_(lt(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05),eq(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05)),mul(-1,sub({disk:close},ts_min({disk:close},100))),mul(-1,ts_delta({disk:close},3)))

Conditional factor based on 100-period average long-term change rate.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_024(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_024: Long-term average change rate.

    Uses 100-period average change rate to determine trading signal.
    """

    default_params = {"long_window": 100, "short_delta": 3, "threshold": 0.05}

    @property
    def name(self) -> str:
        return "alpha_101_024"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_024."""
        close = data["close"]
        window = self.params["long_window"]

        # 100-period average
        close_mean = operator.div(operator.ts_sum(close, window), window)

        # ts_delta(close_mean, 100)
        mean_delta = operator.ts_delta(close_mean, window)

        # delay(close, 100)
        close_lag = operator.delay(close, window)

        # Change rate = mean_delta / close_lag
        rate = operator.div(mean_delta, close_lag)

        # Condition: rate <= 0.05
        condition1 = operator.lt(rate, self.params["threshold"])
        condition2 = operator.eq(rate, self.params["threshold"])
        main_condition = operator.or_(condition1, condition2)

        # ts_min(close, 100)
        close_min = operator.ts_min(close, window)

        # -(close - close_min)
        close_min_diff = operator.sub(close, close_min)
        neg_close_min = operator.mul(close_min_diff, -1)

        # -ts_delta(close, 3)
        close_delta = operator.ts_delta(close, self.params["short_delta"])
        neg_delta = operator.mul(close_delta, -1)

        # Final condition
        alpha = operator.condition(main_condition, neg_close_min, neg_delta)

        return operator.ts_fillna(alpha, 0)
