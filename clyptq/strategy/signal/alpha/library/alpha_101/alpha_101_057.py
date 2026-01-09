"""Alpha 101_057: Close-VWAP with argmax decay signal.

Formula: sub(0,mul(1,div(sub({disk:close},{disk:vwap}),ts_decayed_linear(rank(ts_argmax({disk:close},30)),2))))

Negative of close-VWAP difference divided by decayed rank of argmax position.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_057(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_057: Close-VWAP with argmax decay.

    Negates close-VWAP difference divided by decayed linear rank of 30-period argmax.
    """

    default_params = {"argmax_window": 30, "decay_window": 2}

    @property
    def name(self) -> str:
        return "alpha_101_057"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_057."""
        close = data["close"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Numerator: close - vwap
        close_vwap_diff = operator.sub(close, vwap)

        # Denominator: ts_decayed_linear(rank(ts_argmax(close, 30)), 2)
        # ts_argmax(close, 30)
        argmax = operator.ts_argmax(close, self.params["argmax_window"])

        # rank(argmax)
        argmax_rank = operator.rank(argmax)

        # ts_decayed_linear(argmax_rank, 2)
        decayed_rank = operator.ts_decayed_linear(argmax_rank, self.params["decay_window"])

        # div(close_vwap_diff, decayed_rank)
        ratio = operator.div(close_vwap_diff, decayed_rank)

        # Negate
        alpha = operator.mul(ratio, -1)

        return operator.ts_fillna(alpha, 0)
