"""Alpha 101_101: Price change divided by price range signal.

Formula: div(sub({disk:close},{disk:open}),add(sub({disk:high},{disk:low}),.001))

Simple price change divided by price range (normalized return).
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_101(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_101: Price change divided by price range.

    Calculates price change normalized by price range.
    """

    default_params = {
        "epsilon": 0.001,
    }

    @property
    def name(self) -> str:
        return "alpha_101_101"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_101."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # sub(close, open) - price change
        price_change = operator.sub(close, open_)

        # sub(high, low) - price range
        price_range = operator.sub(high, low)

        # add(price_range, 0.001) - prevent division by zero
        adjusted_range = operator.add(price_range, self.params["epsilon"])

        # div(price_change, adjusted_range) - normalized return
        alpha = operator.div(price_change, adjusted_range)

        return operator.ts_fillna(alpha, 0)
