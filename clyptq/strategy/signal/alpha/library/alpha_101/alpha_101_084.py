"""Alpha 101_084: VWAP max difference ts_rank power by close delta signal.

Formula: pow(ts_rank(sub({disk:vwap},ts_max({disk:vwap},15.3217)),20.7127),ts_delta({disk:close},4.96796))

VWAP-max VWAP difference ts_rank raised to close delta power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_084(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_084: VWAP max difference ts_rank power by close delta.

    Raises VWAP-max VWAP difference ts_rank to close delta power.
    """

    default_params = {"max_window": 15, "rank_window": 21, "delta_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_084"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_084."""
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Base: VWAP - max VWAP ts_rank
        # ts_max(vwap, 15)
        vwap_max = operator.ts_max(vwap, self.params["max_window"])

        # sub(vwap, vwap_max)
        vwap_diff = operator.sub(vwap, vwap_max)

        # ts_rank(vwap_diff, 21)
        base = operator.ts_rank(vwap_diff, self.params["rank_window"])

        # Exponent: close delta
        # ts_delta(close, 5)
        power = operator.ts_delta(close, self.params["delta_window"])

        # pow(base, power)
        alpha = operator.pow(base, power)

        return operator.ts_fillna(alpha, 0)
