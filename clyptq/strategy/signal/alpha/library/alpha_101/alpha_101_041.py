"""Alpha 101_041: Geometric mean minus VWAP signal.

Formula: sub(pow(mul({disk:high},{disk:low}),0.5),{disk:vwap})

Geometric mean of high and low minus VWAP.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_041(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_041: Geometric mean minus VWAP.

    Subtracts VWAP from the geometric mean of high and low prices.
    """

    default_params = {}

    @property
    def name(self) -> str:
        return "alpha_101_041"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_041."""
        high = data["high"]
        low = data["low"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # high * low
        high_low_product = operator.mul(high, low)

        # sqrt(high * low) - geometric mean
        geometric_mean = operator.pow(high_low_product, 0.5)

        # geometric_mean - vwap
        alpha = operator.sub(geometric_mean, vwap)

        return operator.ts_fillna(alpha, 0)
