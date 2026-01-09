"""Alpha 101_056: Returns ratio and cap product signal.

Formula: sub(0,mul(1,mul(rank(div(ts_sum({disk:returns},10),ts_sum(ts_sum({disk:returns},2),3))),rank(mul({disk:returns},{disk:cap})))))

Negative of product of returns ratio rank and returns-cap product rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_056(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_056: Returns ratio and cap product.

    Negates the product of returns sum ratio rank and returns-cap product rank.
    """

    default_params = {"returns_window1": 10, "returns_window2": 2, "nested_window": 3}

    @property
    def name(self) -> str:
        return "alpha_101_056"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_056."""
        close = data["close"]
        volume = data["volume"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.returns(close)

        # Calculate cap (market cap proxy) if not available
        cap = data.get("cap")
        if cap is None:
            cap = operator.mul(volume, close)

        # Part 1: Returns sum ratio
        # ts_sum(returns, 10)
        returns_sum_10 = operator.ts_sum(returns, self.params["returns_window1"])

        # ts_sum(returns, 2)
        returns_sum_2 = operator.ts_sum(returns, self.params["returns_window2"])

        # ts_sum(returns_sum_2, 3)
        returns_sum_nested = operator.ts_sum(returns_sum_2, self.params["nested_window"])

        # div(returns_sum_10, returns_sum_nested)
        returns_ratio = operator.div(returns_sum_10, returns_sum_nested)

        # rank(returns_ratio)
        ratio_rank = operator.rank(returns_ratio)

        # Part 2: Returns-cap product
        # mul(returns, cap)
        returns_cap = operator.mul(returns, cap)

        # rank(returns_cap)
        cap_rank = operator.rank(returns_cap)

        # mul(ratio_rank, cap_rank)
        product = operator.mul(ratio_rank, cap_rank)

        # Negate
        alpha = operator.mul(product, -1)

        return operator.ts_fillna(alpha, 0)
