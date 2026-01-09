"""Alpha 101_037: Long-term open-close correlation signal.

Formula: add(rank(ts_corr(delay(sub({disk:open},{disk:close}),1),{disk:close},200)),rank(sub({disk:open},{disk:close})))

Combines 200-period correlation of lagged open-close diff with current close and open-close ranking.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_037(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_037: Long-term open-close correlation.

    Adds ranked long-term correlation with current open-close difference ranking.
    """

    default_params = {"corr_window": 200, "delay_window": 1}

    @property
    def name(self) -> str:
        return "alpha_101_037"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_037."""
        close = data["close"]
        open_ = data["open"]

        # open - close
        open_close_diff = operator.sub(open_, close)

        # delay(open_close_diff, 1)
        delayed_diff = operator.delay(open_close_diff, self.params["delay_window"])

        # ts_corr(delayed_diff, close, 200)
        corr = operator.ts_corr(delayed_diff, close, self.params["corr_window"])

        # rank(corr)
        first_part = operator.rank(corr)

        # rank(open_close_diff)
        second_part = operator.rank(open_close_diff)

        # add(first_part, second_part)
        alpha = operator.add(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
