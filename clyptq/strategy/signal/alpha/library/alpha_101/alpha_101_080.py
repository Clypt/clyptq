"""Alpha 101_080: Demeaned weighted open-high delta sign rank power by high-amount correlation ts_rank signal.

Formula: mul(pow(rank(sign(ts_delta(grouped_demean(add(mul({disk:open},0.868128),mul({disk:high},sub(1,0.868128))),{disk:industry_group_lv2}),4.04545))),ts_rank(ts_corr({disk:high},ts_mean({disk:amount},10),5.11456),5.53756)),-1)

Negative of demeaned weighted open-high delta sign rank raised to high-amount correlation ts_rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_080(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_080: Demeaned weighted open-high delta sign rank power by high-amount correlation ts_rank.

    Negates the power of demeaned price delta sign rank by high-amount correlation ts_rank.
    """

    default_params = {
        "weight": 0.868128,
        "delta_window": 4,
        "amount_window": 10,
        "corr_window": 5,
        "rank_window": 6,
    }

    @property
    def name(self) -> str:
        return "alpha_101_080"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_080."""
        open_ = data["open"]
        high = data["high"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        weight = self.params["weight"]

        # Part 1: Demeaned weighted open-high delta sign
        # Weighted open-high: open * weight + high * (1-weight)
        open_part = operator.mul(open_, weight)
        high_part = operator.mul(high, 1 - weight)
        weighted_open_high = operator.add(open_part, high_part)

        # Demean (cross-sectional mean removal)
        weighted_demeaned = operator.demean(weighted_open_high)

        # ts_delta(weighted_demeaned, 4)
        weighted_delta = operator.ts_delta(weighted_demeaned, self.params["delta_window"])

        # sign(weighted_delta)
        delta_sign = operator.sign(weighted_delta)

        # rank(delta_sign)
        base_rank = operator.rank(delta_sign)

        # Part 2: High-amount correlation ts_rank
        # ts_mean(amount, 10)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(high, amount_mean, 5)
        corr = operator.ts_corr(high, amount_mean, self.params["corr_window"])

        # ts_rank(corr, 6)
        power_rank = operator.ts_rank(corr, self.params["rank_window"])

        # pow(base_rank, power_rank)
        powered = operator.pow(base_rank, power_rank)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
