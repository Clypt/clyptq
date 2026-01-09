"""Alpha 101_090: Close max difference rank power by demeaned amount-low correlation ts_rank signal.

Formula: mul(pow(rank(sub({disk:close},ts_max({disk:close},4.66719))),ts_rank(ts_corr(grouped_demean(ts_mean({disk:amount},40),{disk:industry_group_lv3}),{disk:low},5.38375),3.21856)),-1)

Negative of close-max close difference rank raised to demeaned amount-low correlation ts_rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_090(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_090: Close max difference rank power by demeaned amount-low correlation ts_rank.

    Negates the power of close-max close difference rank by demeaned amount-low correlation ts_rank.
    """

    default_params = {
        "max_window": 5,
        "amount_window": 40,
        "corr_window": 5,
        "rank_window": 3,
    }

    @property
    def name(self) -> str:
        return "alpha_101_090"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_090."""
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Base: Close - max close rank
        # ts_max(close, 5)
        close_max = operator.ts_max(close, self.params["max_window"])

        # sub(close, close_max)
        close_diff = operator.sub(close, close_max)

        # rank(close_diff)
        base = operator.rank(close_diff)

        # Exponent: Demeaned amount-low correlation ts_rank
        # ts_mean(amount, 40)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # Demean amount_mean (cross-sectional mean removal)
        amount_demeaned = operator.demean(amount_mean)

        # ts_corr(amount_demeaned, low, 5)
        amount_low_corr = operator.ts_corr(amount_demeaned, low, self.params["corr_window"])

        # ts_rank(amount_low_corr, 3)
        power = operator.ts_rank(amount_low_corr, self.params["rank_window"])

        # pow(base, power)
        powered = operator.pow(base, power)

        # mul(powered, -1)
        alpha = operator.mul(powered, -1)

        return operator.ts_fillna(alpha, 0)
