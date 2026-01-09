"""Alpha 101_021: Close volatility and volume ratio signal.

Formula: condition(lt(add(div(ts_sum({disk:close},8),8),ts_std({disk:close},8)),div(ts_sum({disk:close},2),2)),mul(-1,1),condition(lt(div(ts_sum({disk:close},2),2),sub(div(ts_sum({disk:close},8),8),ts_std({disk:close},8))),1,condition(or_(lt(1,div({disk:volume},ts_mean({disk:amount},20))),eq(div({disk:volume},ts_mean({disk:amount},20)),1)),1,mul(-1,1))))

Conditional factor using 8-period average, standard deviation, 2-period average, and volume ratio.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_021(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_021: Close volatility with volume ratio.

    Multi-condition factor based on close price statistics and volume ratio.
    """

    default_params = {"short_window": 2, "long_window": 8, "amount_window": 20}

    @property
    def name(self) -> str:
        return "alpha_101_021"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_021."""
        close = data["close"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # 8-period average
        close_mean_8 = operator.div(
            operator.ts_sum(close, self.params["long_window"]),
            self.params["long_window"],
        )

        # 8-period std
        close_std_8 = operator.ts_std(close, self.params["long_window"])

        # 2-period average
        close_mean_2 = operator.div(
            operator.ts_sum(close, self.params["short_window"]),
            self.params["short_window"],
        )

        # volume / ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        volume_ratio = operator.div(volume, amount_mean)

        # Condition 1: (close_mean_8 + close_std_8) < close_mean_2
        condition1 = operator.lt(operator.add(close_mean_8, close_std_8), close_mean_2)

        # Condition 2: close_mean_2 < (close_mean_8 - close_std_8)
        condition2 = operator.lt(close_mean_2, operator.sub(close_mean_8, close_std_8))

        # Condition 3: volume_ratio >= 1
        condition3a = operator.lt(1, volume_ratio)
        condition3b = operator.eq(volume_ratio, 1)
        condition3 = operator.or_(condition3a, condition3b)

        # Nested conditions
        inner_condition = operator.condition(condition3, 1, -1)
        middle_condition = operator.condition(condition2, 1, inner_condition)
        alpha = operator.condition(condition1, -1, middle_condition)

        return operator.ts_fillna(alpha, 0)
