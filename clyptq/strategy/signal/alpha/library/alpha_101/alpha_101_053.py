"""Alpha 101_053: Price position delta signal.

Formula: mul(-1,ts_delta(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:close},{disk:low})),9))

Negative of 9-period change in Williams %R-like price position indicator.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_053(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_053: Price position delta.

    Negates the 9-period delta of a Williams %R-like price position indicator.
    """

    default_params = {"delta_window": 9}

    @property
    def name(self) -> str:
        return "alpha_101_053"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_053."""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # close - low
        close_low_diff = operator.sub(close, low)

        # high - close
        high_close_diff = operator.sub(high, close)

        # (close - low) - (high - close)
        numerator = operator.sub(close_low_diff, high_close_diff)

        # div(numerator, close - low) - Williams %R-like indicator
        price_position = operator.div(numerator, close_low_diff)

        # ts_delta(price_position, 9)
        delta = operator.ts_delta(price_position, self.params["delta_window"])

        # mul(-1, delta)
        alpha = operator.mul(delta, -1)

        return operator.ts_fillna(alpha, 0)
