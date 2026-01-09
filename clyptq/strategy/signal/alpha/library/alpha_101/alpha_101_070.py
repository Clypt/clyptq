"""Alpha 101_070: VWAP delta rank power by demeaned close-amount correlation ts_rank signal.

Formula: mul(pow(rank(ts_delta({disk:vwap},1.29456)),ts_rank(ts_corr(grouped_demean({disk:close},{disk:industry_group_lv2}),ts_mean({disk:amount},50),17.8256),17.9171)),-1)

Negative of VWAP delta rank raised to demeaned close-amount correlation ts_rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_070(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_070: VWAP delta rank power by demeaned close-amount correlation ts_rank.

    Negates the power of VWAP delta rank by demeaned close-amount correlation ts_rank.
    """

    default_params = {
        "delta_window": 1,
        "amount_window": 50,
        "corr_window": 18,
        "rank_window": 18,
    }

    @property
    def name(self) -> str:
        return "alpha_101_070"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_070."""
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

        # Part 1: VWAP delta rank
        # ts_delta(vwap, 1)
        vwap_delta = operator.ts_delta(vwap, self.params["delta_window"])

        # rank(vwap_delta)
        base_rank = operator.rank(vwap_delta)

        # Part 2: Demeaned close-amount correlation ts_rank
        # Demean close (cross-sectional mean removal)
        close_demeaned = operator.demean(close)

        # ts_mean(amount, 50)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(close_demeaned, amount_mean, 18)
        corr = operator.ts_corr(close_demeaned, amount_mean, self.params["corr_window"])

        # ts_rank(corr, 18)
        power_rank = operator.ts_rank(corr, self.params["rank_window"])

        # pow(base_rank, power_rank)
        powered = operator.pow(base_rank, power_rank)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
