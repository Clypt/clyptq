"""Alpha 101_018: Close-open volatility and correlation signal.

Formula: mul(-1,rank(add(add(ts_std(abs(sub({disk:close},{disk:open})),5),sub({disk:close},{disk:open})),ts_corr({disk:close},{disk:open},10))))

Complex factor combining close-open volatility, difference, and correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_018(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_018: Close-open volatility and correlation.

    Combines close-open difference volatility, level, and correlation.
    """

    default_params = {"std_window": 5, "corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_018"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_018."""
        close = data["close"]
        open_ = data["open"]

        # close - open
        close_open_diff = operator.sub(close, open_)

        # ts_std(abs(close - open), 5)
        abs_diff = operator.abs(close_open_diff)
        std = operator.ts_std(abs_diff, self.params["std_window"])

        # ts_corr(close, open, 10)
        corr = operator.ts_corr(close, open_, self.params["corr_window"])

        # add(add(std, close_open_diff), corr)
        sum_all = operator.add(operator.add(std, close_open_diff), corr)

        # rank(...)
        ranked = operator.rank(sum_all)

        # mul(-1, ...)
        alpha = operator.mul(ranked, -1)

        return operator.ts_fillna(alpha, 0)
