"""Alpha 101_034: Volatility ratio and price change signal.

Formula: rank(add(sub(1,rank(div(ts_std({disk:returns},2),ts_std({disk:returns},5)))),sub(1,rank(ts_delta({disk:close},1)))))

Ranking of inverse returns volatility ratio plus inverse price change rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_034(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_034: Volatility ratio and price change.

    Combines inverse rankings of returns volatility ratio and close delta.
    """

    default_params = {"short_std": 2, "long_std": 5, "delta_window": 1}

    @property
    def name(self) -> str:
        return "alpha_101_034"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_034."""
        close = data["close"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # Part 1: Volatility ratio
        returns_std_short = operator.ts_std(returns, self.params["short_std"])
        returns_std_long = operator.ts_std(returns, self.params["long_std"])
        std_ratio = operator.div(returns_std_short, returns_std_long)
        std_rank = operator.rank(std_ratio)
        first_part = operator.sub(1, std_rank)

        # Part 2: Price change
        close_delta = operator.ts_delta(close, self.params["delta_window"])
        delta_rank = operator.rank(close_delta)
        second_part = operator.sub(1, delta_rank)

        # Sum and rank
        sum_parts = operator.add(first_part, second_part)
        alpha = operator.rank(sum_parts)

        return operator.ts_fillna(alpha, 0)
