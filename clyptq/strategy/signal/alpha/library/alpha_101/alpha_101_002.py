"""Alpha 101_002: Volume-price correlation signal.

Formula: mul(-1,ts_corr(rank(ts_delta(log(volume),2)),rank(div(sub(close,open),open)),6))

Exploiting negative correlation between volume change and price momentum
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_002(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_002: Volume-price correlation.

    Negative correlation between volume change rank and price momentum rank.
    """

    default_params = {"delta_window": 2, "corr_window": 6}

    @property
    def name(self) -> str:
        return "alpha_101_002"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_002."""
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        # 2-period difference of log(volume)
        log_volume = operator.log(volume)
        volume_delta = operator.ts_delta(log_volume, self.params["delta_window"])
        volume_rank = operator.rank(volume_delta)

        # Ranking of (close - open) / open
        price_change = operator.div(operator.sub(close, open_), open_)
        price_rank = operator.rank(price_change)

        # 6-period correlation * -1
        corr = operator.ts_corr(volume_rank, price_rank, self.params["corr_window"])
        alpha = operator.mul(corr, -1)

        return operator.ts_fillna(alpha, 0)
