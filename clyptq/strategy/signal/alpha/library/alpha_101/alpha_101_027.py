"""Alpha 101_027: Volume-VWAP correlation ranking signal.

Formula: condition(lt(0.5,rank(div(ts_sum(ts_corr(rank({disk:volume}),rank({disk:vwap}),6),2),2.0))),mul(-1,1),1)

Conditional factor based on ranked volume-VWAP correlation sum.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_027(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_027: Volume-VWAP correlation ranking.

    Returns -1 when ranked correlation sum exceeds 0.5, else 1.
    """

    default_params = {"corr_window": 6, "sum_window": 2}

    @property
    def name(self) -> str:
        return "alpha_101_027"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_027."""
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # rank(volume)
        volume_rank = operator.rank(volume)

        # rank(vwap)
        vwap_rank = operator.rank(vwap)

        # ts_corr(volume_rank, vwap_rank, 6)
        corr = operator.ts_corr(volume_rank, vwap_rank, self.params["corr_window"])

        # ts_sum(corr, 2) / 2.0
        sum_corr = operator.ts_sum(corr, self.params["sum_window"])
        div_result = operator.div(sum_corr, 2.0)

        # rank(div_result)
        rank_result = operator.rank(div_result)

        # condition: 0.5 < rank_result
        condition = operator.lt(0.5, rank_result)

        # condition(condition, -1, 1)
        alpha = operator.condition(condition, -1, 1)

        return operator.ts_fillna(alpha, 0)
