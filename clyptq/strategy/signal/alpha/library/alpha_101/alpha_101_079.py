"""Alpha 101_079: Demeaned weighted close-open delta rank vs VWAP-amount ts_rank correlation rank signal.

Formula: lt(rank(ts_delta(grouped_demean(add(mul({disk:close},0.60733),mul({disk:open},sub(1,0.60733))),{disk:industry_group_lv1}),1.23438)),rank(ts_corr(ts_rank({disk:vwap},3.60973),ts_rank(ts_mean({disk:amount},150),9.18637),14.6644)))

Compares demeaned weighted close-open delta rank with VWAP-amount ts_rank correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_079(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_079: Demeaned weighted close-open delta rank vs VWAP-amount ts_rank correlation rank.

    Returns 1 when demeaned price delta rank is less than VWAP-amount correlation rank.
    """

    default_params = {
        "weight": 0.60733,
        "delta_window": 1,
        "vwap_rank_window": 4,
        "amount_window": 150,
        "amount_rank_window": 9,
        "corr_window": 15,
    }

    @property
    def name(self) -> str:
        return "alpha_101_079"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_079."""
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

        weight = self.params["weight"]

        # Part 1: Demeaned weighted close-open delta
        # Weighted close-open: close * weight + open * (1-weight)
        close_part = operator.mul(close, weight)
        open_part = operator.mul(open_, 1 - weight)
        weighted_close_open = operator.add(close_part, open_part)

        # Demean (cross-sectional mean removal)
        weighted_demeaned = operator.demean(weighted_close_open)

        # ts_delta(weighted_demeaned, 1)
        weighted_delta = operator.ts_delta(weighted_demeaned, self.params["delta_window"])

        # rank(weighted_delta)
        first_rank = operator.rank(weighted_delta)

        # Part 2: VWAP-amount ts_rank correlation
        # ts_rank(vwap, 4)
        vwap_tsrank = operator.ts_rank(vwap, self.params["vwap_rank_window"])

        # ts_mean(amount, 150)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_rank(amount_mean, 9)
        amount_tsrank = operator.ts_rank(amount_mean, self.params["amount_rank_window"])

        # ts_corr(vwap_tsrank, amount_tsrank, 15)
        corr = operator.ts_corr(vwap_tsrank, amount_tsrank, self.params["corr_window"])

        # rank(corr)
        second_rank = operator.rank(corr)

        # lt(first_rank, second_rank)
        alpha = operator.lt(first_rank, second_rank)

        # Convert boolean to float
        alpha = alpha.astype(float)

        return operator.ts_fillna(alpha, 0)
