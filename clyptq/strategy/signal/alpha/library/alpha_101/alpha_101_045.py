"""Alpha 101_045: Delayed close mean with correlations signal.

Formula: mul(-1,mul(mul(rank(div(ts_sum(delay({disk:close},5),20),20)),ts_corr({disk:close},{disk:volume},2)),rank(ts_corr(ts_sum({disk:close},5),ts_sum({disk:close},20),2))))

Negative product of delayed close mean rank, close-volume correlation, and multi-period close sum correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_045(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_045: Delayed close mean with correlations.

    Combines ranked delayed close mean with close-volume and multi-period sum correlations.
    """

    default_params = {
        "delay_window": 5,
        "sum_window": 20,
        "short_sum": 5,
        "long_sum": 20,
        "corr_window": 2,
    }

    @property
    def name(self) -> str:
        return "alpha_101_045"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_045."""
        close = data["close"]
        volume = data["volume"]

        # Part 1: rank(mean of delayed close)
        close_lag = operator.delay(close, self.params["delay_window"])
        sum_delayed = operator.ts_sum(close_lag, self.params["sum_window"])
        mean_delayed = operator.div(sum_delayed, self.params["sum_window"])
        first_part = operator.rank(mean_delayed)

        # Part 2: ts_corr(close, volume, 2)
        second_part = operator.ts_corr(close, volume, self.params["corr_window"])

        # Part 3: rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
        sum_5 = operator.ts_sum(close, self.params["short_sum"])
        sum_20 = operator.ts_sum(close, self.params["long_sum"])
        corr_sums = operator.ts_corr(sum_5, sum_20, self.params["corr_window"])
        third_part = operator.rank(corr_sums)

        # Multiply all parts
        product = operator.mul(operator.mul(first_part, second_part), third_part)

        # -1 * product
        alpha = operator.mul(product, -1)

        return operator.ts_fillna(alpha, 0)
