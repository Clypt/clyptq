"""Alpha 101_069: Demeaned VWAP delta max rank power by weighted price-amount correlation ts_rank signal.

Formula: mul(pow(rank(ts_max(ts_delta(grouped_demean({disk:vwap},{disk:industry_group_lv2}),2.72412),4.79344)),ts_rank(ts_corr(add(mul({disk:close},0.490655),mul({disk:vwap},sub(1,0.490655))),ts_mean({disk:amount},20),4.92416),9.0615)),-1)

Negative of demeaned VWAP delta max rank raised to weighted price-amount correlation ts_rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_069(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_069: Demeaned VWAP delta max rank power by weighted price-amount correlation ts_rank.

    Negates the power of demeaned VWAP delta max rank by price-amount correlation ts_rank.
    """

    default_params = {
        "delta_window": 3,
        "max_window": 5,
        "weight": 0.490655,
        "amount_window": 20,
        "corr_window": 5,
        "rank_window": 9,
    }

    @property
    def name(self) -> str:
        return "alpha_101_069"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_069."""
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: Demeaned VWAP delta max
        # Demean VWAP (cross-sectional mean removal)
        vwap_demeaned = operator.demean(vwap)

        # ts_delta(vwap_demeaned, 3)
        vwap_delta = operator.ts_delta(vwap_demeaned, self.params["delta_window"])

        # ts_max(vwap_delta, 5)
        vwap_max = operator.ts_max(vwap_delta, self.params["max_window"])

        # rank(vwap_max)
        base_rank = operator.rank(vwap_max)

        # Part 2: Weighted price-amount correlation ts_rank
        weight = self.params["weight"]

        # Weighted price: close * weight + vwap * (1-weight)
        close_part = operator.mul(close, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_price = operator.add(close_part, vwap_part)

        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(weighted_price, amount_mean, 5)
        corr = operator.ts_corr(weighted_price, amount_mean, self.params["corr_window"])

        # ts_rank(corr, 9)
        power_rank = operator.ts_rank(corr, self.params["rank_window"])

        # pow(base_rank, power_rank)
        powered = operator.pow(base_rank, power_rank)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
