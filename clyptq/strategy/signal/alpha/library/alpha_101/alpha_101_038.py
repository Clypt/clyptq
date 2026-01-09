"""Alpha 101_038: Close time-series rank with close/open ratio signal.

Formula: mul(mul(-1,rank(ts_rank({disk:close},10))),rank(div({disk:close},{disk:open})))

Negative product of close time-series rank and close/open ratio rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_038(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_038: Close time-series with close/open ratio.

    Multiplies negative close time-series rank with close/open ratio rank.
    """

    default_params = {"ts_rank_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_038"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_038."""
        close = data["close"]
        open_ = data["open"]

        # ts_rank(close, 10)
        close_tsrank = operator.ts_rank(close, self.params["ts_rank_window"])

        # rank(close_tsrank) * -1
        close_rank = operator.rank(close_tsrank)
        neg_close_rank = operator.mul(close_rank, -1)

        # close / open
        close_open_ratio = operator.div(close, open_)

        # rank(close_open_ratio)
        ratio_rank = operator.rank(close_open_ratio)

        # neg_close_rank * ratio_rank
        alpha = operator.mul(neg_close_rank, ratio_rank)

        return operator.ts_fillna(alpha, 0)
