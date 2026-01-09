"""Alpha 101_005: VWAP deviation signal.

Formula: mul(rank(sub({disk:open},div(ts_sum({disk:vwap},10),10))),mul(-1,abs(rank(sub({disk:close},{disk:vwap})))))

Combination of open-VWAP deviation and close-VWAP deviation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_005(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_005: VWAP deviation combination.

    Combines open price deviation from average VWAP with close-VWAP deviation.
    """

    default_params = {"vwap_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_005"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_005."""
        open_ = data["open"]
        close = data["close"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        window = self.params["vwap_window"]

        # ts_sum(vwap, 10) / 10 - average VWAP
        vwap_avg = operator.div(operator.ts_sum(vwap, window), window)

        # open - vwap_avg
        open_vwap_diff = operator.sub(open_, vwap_avg)

        # rank(open_vwap_diff)
        first_part = operator.rank(open_vwap_diff)

        # close - vwap
        close_vwap_diff = operator.sub(close, vwap)

        # rank(close_vwap_diff)
        close_vwap_rank = operator.rank(close_vwap_diff)

        # abs(close_vwap_rank)
        abs_close_rank = operator.abs(close_vwap_rank)

        # -1 * abs_close_rank
        second_part = operator.mul(abs_close_rank, -1)

        # first_part * second_part
        alpha = operator.mul(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
