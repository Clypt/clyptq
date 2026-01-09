"""Alpha 101_008: Open-returns delayed comparison signal.

Formula: mul(-1,rank(sub(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),delay(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),10))))

Ranking of change in open-returns product over 10 periods.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_008(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_008: Open-returns delayed comparison.

    Negative rank of difference between current and delayed open-returns product.
    """

    default_params = {"sum_window": 5, "delay_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_008"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_008."""
        open_ = data["open"]
        close = data["close"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # ts_sum(open, 5)
        open_sum = operator.ts_sum(open_, self.params["sum_window"])

        # ts_sum(returns, 5)
        returns_sum = operator.ts_sum(returns, self.params["sum_window"])

        # mul(ts_sum(open, 5), ts_sum(returns, 5))
        current_product = operator.mul(open_sum, returns_sum)

        # delay(current_product, 10)
        delayed_product = operator.delay(current_product, self.params["delay_window"])

        # sub(current_product, delayed_product)
        diff = operator.sub(current_product, delayed_product)

        # rank(diff)
        ranked = operator.rank(diff)

        # mul(-1, ranked)
        alpha = operator.mul(ranked, -1)

        return operator.ts_fillna(alpha, 0)
