"""Alpha 101_050: Volume-VWAP correlation max signal.

Formula: mul(-1,ts_max(rank(ts_corr(rank({disk:volume}),rank({disk:vwap}),5)),5))

Negative of 5-period max of ranked volume-VWAP correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_050(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_050: Volume-VWAP correlation max.

    Negates the 5-period maximum of ranked correlation between volume and VWAP rankings.
    """

    default_params = {"corr_window": 5, "max_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_050"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_050."""
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

        # ts_corr(volume_rank, vwap_rank, 5)
        corr = operator.ts_corr(volume_rank, vwap_rank, self.params["corr_window"])

        # rank(corr)
        corr_rank = operator.rank(corr)

        # ts_max(corr_rank, 5)
        max_corr = operator.ts_max(corr_rank, self.params["max_window"])

        # mul(-1, max_corr)
        alpha = operator.mul(max_corr, -1)

        return operator.ts_fillna(alpha, 0)
