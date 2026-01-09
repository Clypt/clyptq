"""Alpha 101_033: Open-close ratio momentum signal.

Formula: rank(mul(-1,pow(sub(1,div({disk:open},{disk:close})),1)))

Ranking of negative open/close ratio deviation from 1.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_033(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_033: Open-close ratio momentum.

    Ranks negative deviation of open/close ratio from 1.
    """

    default_params = {}

    @property
    def name(self) -> str:
        return "alpha_101_033"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_033."""
        open_ = data["open"]
        close = data["close"]

        # open / close
        open_close_ratio = operator.div(open_, close)

        # 1 - (open / close)
        one_minus_ratio = operator.sub(1, open_close_ratio)

        # pow(..., 1) is identity
        powered = operator.pow(one_minus_ratio, 1)

        # -1 * powered
        neg_powered = operator.mul(powered, -1)

        # rank(...)
        alpha = operator.rank(neg_powered)

        return operator.ts_fillna(alpha, 0)
