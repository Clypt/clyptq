"""Alpha 101_042: VWAP-close difference to sum ratio signal.

Formula: div(rank(sub({disk:vwap},{disk:close})),rank(add({disk:vwap},{disk:close})))

Ratio of VWAP-close difference rank to sum rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_042(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_042: VWAP-close difference to sum ratio.

    Divides ranked VWAP-close difference by ranked VWAP-close sum.
    """

    default_params = {}

    @property
    def name(self) -> str:
        return "alpha_101_042"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_042."""
        close = data["close"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # vwap - close
        vwap_close_diff = operator.sub(vwap, close)

        # rank(vwap - close)
        diff_rank = operator.rank(vwap_close_diff)

        # vwap + close
        vwap_close_sum = operator.add(vwap, close)

        # rank(vwap + close)
        sum_rank = operator.rank(vwap_close_sum)

        # diff_rank / sum_rank
        alpha = operator.div(diff_rank, sum_rank)

        return operator.ts_fillna(alpha, 0)
