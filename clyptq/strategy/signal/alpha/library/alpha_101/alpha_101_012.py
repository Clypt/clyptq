"""Alpha 101_012: Volume-price divergence signal.

Formula: mul(sign(ts_delta({disk:volume},1)),mul(-1,ts_delta({disk:close},1)))

Uses opposite relationship between volume change direction and price change.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_012(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_012: Volume-price divergence.

    Multiplies volume change sign with negated price change.
    """

    default_params = {"delta_window": 1}

    @property
    def name(self) -> str:
        return "alpha_101_012"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_012."""
        close = data["close"]
        volume = data["volume"]

        # sign(ts_delta(volume, 1))
        volume_delta = operator.ts_delta(volume, self.params["delta_window"])
        volume_sign = operator.sign(volume_delta)

        # -1 * ts_delta(close, 1)
        close_delta = operator.ts_delta(close, self.params["delta_window"])
        close_neg = operator.mul(close_delta, -1)

        # volume_sign * close_neg
        alpha = operator.mul(volume_sign, close_neg)

        return operator.ts_fillna(alpha, 0)
