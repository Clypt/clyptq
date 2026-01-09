"""Alpha 101_011: VWAP-close deviation and volume change signal.

Formula: mul(add(rank(ts_max(sub({disk:vwap},{disk:close}),3)),rank(ts_min(sub({disk:vwap},{disk:close}),3))),rank(ts_delta({disk:volume},3)))

Combination of VWAP-close deviation extremes and volume change.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_011(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_011: VWAP-close deviation with volume change.

    Combines extreme VWAP-close deviations with volume delta ranking.
    """

    default_params = {"window": 3}

    @property
    def name(self) -> str:
        return "alpha_101_011"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_011."""
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        window = self.params["window"]

        # vwap - close
        vwap_close_diff = operator.sub(vwap, close)

        # ts_max(vwap - close, 3)
        max_diff = operator.ts_max(vwap_close_diff, window)

        # ts_min(vwap - close, 3)
        min_diff = operator.ts_min(vwap_close_diff, window)

        # rank(ts_max(...)) + rank(ts_min(...))
        rank_max = operator.rank(max_diff)
        rank_min = operator.rank(min_diff)
        first_part = operator.add(rank_max, rank_min)

        # rank(ts_delta(volume, 3))
        volume_delta = operator.ts_delta(volume, window)
        second_part = operator.rank(volume_delta)

        # mul(first_part, second_part)
        alpha = operator.mul(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
