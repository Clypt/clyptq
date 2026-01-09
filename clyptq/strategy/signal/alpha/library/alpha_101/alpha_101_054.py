"""Alpha 101_054: Price ratio with power signal.

Formula: div(mul(-1,mul(sub({disk:low},{disk:close}),pow({disk:open},5))),mul(sub({disk:low},{disk:high}),pow({disk:close},5)))

Ratio of (low-close)*open^5 to (low-high)*close^5, negated.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_054(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_054: Price ratio with power.

    Computes ratio of price differences weighted by 5th power of open and close.
    """

    default_params = {"power": 5}

    @property
    def name(self) -> str:
        return "alpha_101_054"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_054."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        power = self.params["power"]

        # Numerator: -1 * (low - close) * open^5
        low_close_diff = operator.sub(low, close)
        open_power = operator.pow(open_, power)
        numerator_product = operator.mul(low_close_diff, open_power)
        numerator = operator.mul(numerator_product, -1)

        # Denominator: (low - high) * close^5
        low_high_diff = operator.sub(low, high)
        close_power = operator.pow(close, power)
        denominator = operator.mul(low_high_diff, close_power)

        # div(numerator, denominator)
        alpha = operator.div(numerator, denominator)

        return operator.ts_fillna(alpha, 0)
