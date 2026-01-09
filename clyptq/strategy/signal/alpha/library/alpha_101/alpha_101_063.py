"""Alpha 101_063: Demeaned close delta vs weighted price-amount correlation signal.

Formula: mul(sub(rank(ts_decayed_linear(ts_delta(grouped_demean({disk:close},{disk:industry_group_lv2}),2.25164),8.22237)),rank(ts_decayed_linear(ts_corr(add(mul({disk:vwap},0.318108),mul({disk:open},sub(1,0.318108))),ts_sum(ts_mean({disk:amount},180),37.2467),13.557),12.2883))),-1)

Negative of difference between demeaned close delta decay rank and weighted price-amount correlation decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_063(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_063: Demeaned close delta vs weighted price-amount correlation.

    Negates the difference between two decayed linear rank values.
    """

    default_params = {
        "delta_window": 2,
        "decay_window1": 8,
        "vwap_weight": 0.318108,
        "amount_window": 180,
        "sum_window": 37,
        "corr_window": 14,
        "decay_window2": 12,
    }

    @property
    def name(self) -> str:
        return "alpha_101_063"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_063."""
        open_ = data["open"]
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

        # Part 1: Demeaned close delta
        # Demean close (cross-sectional mean removal)
        close_demeaned = operator.demean(close)

        # ts_delta(close_demeaned, 2)
        close_delta = operator.ts_delta(close_demeaned, self.params["delta_window"])

        # ts_decayed_linear(close_delta, 8)
        first_decayed = operator.ts_decayed_linear(close_delta, self.params["decay_window1"])

        # rank(first_decayed)
        first_rank = operator.rank(first_decayed)

        # Part 2: Weighted price-amount correlation
        weight = self.params["vwap_weight"]

        # Weighted price: vwap * weight + open * (1-weight)
        vwap_part = operator.mul(vwap, weight)
        open_part = operator.mul(open_, 1 - weight)
        weighted_price = operator.add(vwap_part, open_part)

        # ts_mean(amount, 180)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 37)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(weighted_price, amount_sum, 14)
        price_corr = operator.ts_corr(weighted_price, amount_sum, self.params["corr_window"])

        # ts_decayed_linear(price_corr, 12)
        second_decayed = operator.ts_decayed_linear(price_corr, self.params["decay_window2"])

        # rank(second_decayed)
        second_rank = operator.rank(second_decayed)

        # sub(first_rank, second_rank)
        rank_diff = operator.sub(first_rank, second_rank)

        # mul(rank_diff, -1)
        alpha = operator.mul(rank_diff, -1)

        return operator.ts_fillna(alpha, 0)
