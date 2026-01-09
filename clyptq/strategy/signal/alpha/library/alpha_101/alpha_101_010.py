"""Alpha 101_010: Price change direction consistency ranking.

Formula: rank(condition(lt(0,ts_min(ts_delta({disk:close},1),4)),ts_delta({disk:close},1),condition(lt(ts_max(ts_delta({disk:close},1),4),0),ts_delta({disk:close},1),mul(-1,ts_delta({disk:close},1)))))

Ranked conditional reversal based on 4-day price direction consistency.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_010(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_010: Ranked price change direction consistency.

    Ranks conditional price change based on 4-day direction consistency.
    """

    default_params = {"delta_window": 1, "consistency_window": 4}

    @property
    def name(self) -> str:
        return "alpha_101_010"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_010."""
        close = data["close"]

        # ts_delta(close, 1)
        close_delta = operator.ts_delta(close, self.params["delta_window"])

        # ts_min(close_delta, 4)
        min_delta = operator.ts_min(close_delta, self.params["consistency_window"])

        # ts_max(close_delta, 4)
        max_delta = operator.ts_max(close_delta, self.params["consistency_window"])

        # Condition 1: 0 < min_delta (all rising in last 4 days)
        cond1 = operator.gt(min_delta, 0)

        # Condition 2: max_delta < 0 (all falling in last 4 days)
        cond2 = operator.lt(max_delta, 0)

        # Negated close_delta
        neg_close_delta = operator.mul(close_delta, -1)

        # Inner condition: if cond2 then close_delta else neg_close_delta
        inner_condition = operator.condition(cond2, close_delta, neg_close_delta)

        # Outer condition: if cond1 then close_delta else inner_condition
        condition_result = operator.condition(cond1, close_delta, inner_condition)

        # rank(condition_result)
        alpha = operator.rank(condition_result)

        return operator.ts_fillna(alpha, 0)
