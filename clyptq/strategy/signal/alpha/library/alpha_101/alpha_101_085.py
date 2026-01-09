"""Alpha 101_085: Weighted high-close amount correlation rank power by mid-volume ts_rank correlation rank signal.

Formula: pow(rank(ts_corr(add(mul({disk:high},0.876703),mul({disk:close},sub(1,0.876703))),ts_mean({disk:amount},30),9.61331)),rank(ts_corr(ts_rank(div(add({disk:high},{disk:low}),2),3.70596),ts_rank({disk:volume},10.1595),7.11408)))

Weighted high-close amount correlation rank raised to mid price-volume ts_rank correlation rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_085(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_085: Weighted high-close amount correlation rank power by mid-volume ts_rank correlation rank.

    Raises weighted price-amount correlation rank to mid-volume ts_rank correlation rank power.
    """

    default_params = {
        "weight": 0.876703,
        "amount_window": 30,
        "corr_window1": 10,
        "mid_rank_window": 4,
        "vol_rank_window": 10,
        "corr_window2": 7,
    }

    @property
    def name(self) -> str:
        return "alpha_101_085"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_085."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        weight = self.params["weight"]

        # Base: Weighted high-close amount correlation rank
        # Weighted high-close: high * weight + close * (1-weight)
        high_part = operator.mul(high, weight)
        close_part = operator.mul(close, 1 - weight)
        weighted_high_close = operator.add(high_part, close_part)

        # ts_mean(amount, 30)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(weighted_high_close, amount_mean, 10)
        first_corr = operator.ts_corr(weighted_high_close, amount_mean, self.params["corr_window1"])

        # rank(first_corr)
        base = operator.rank(first_corr)

        # Exponent: Mid price-volume ts_rank correlation rank
        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # ts_rank(mid_price, 4)
        mid_tsrank = operator.ts_rank(mid_price, self.params["mid_rank_window"])

        # ts_rank(volume, 10)
        vol_tsrank = operator.ts_rank(volume, self.params["vol_rank_window"])

        # ts_corr(mid_tsrank, vol_tsrank, 7)
        second_corr = operator.ts_corr(mid_tsrank, vol_tsrank, self.params["corr_window2"])

        # rank(second_corr)
        power = operator.rank(second_corr)

        # pow(base, power)
        alpha = operator.pow(base, power)

        return operator.ts_fillna(alpha, 0)
