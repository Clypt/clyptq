"""Alpha 101_086: Close-amount correlation ts_rank vs price sum difference rank signal.

Formula: mul(lt(ts_rank(ts_corr({disk:close},ts_sum(ts_mean({disk:amount},20),14.7444),6.00049),20.4195),rank(sub(add({disk:open},{disk:close}),add({disk:vwap},{disk:open})))),-1)

Negative comparison between close-amount correlation ts_rank and price sum difference rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_086(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_086: Close-amount correlation ts_rank vs price sum difference rank.

    Negates the comparison between close-amount correlation ts_rank and price difference rank.
    """

    default_params = {
        "amount_window": 20,
        "sum_window": 15,
        "corr_window": 6,
        "rank_window": 20,
    }

    @property
    def name(self) -> str:
        return "alpha_101_086"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_086."""
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

        # Part 1: Close-amount correlation ts_rank
        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 15)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(close, amount_sum, 6)
        close_corr = operator.ts_corr(close, amount_sum, self.params["corr_window"])

        # ts_rank(close_corr, 20)
        first_part = operator.ts_rank(close_corr, self.params["rank_window"])

        # Part 2: Price sum difference rank
        # open + close
        open_close_sum = operator.add(open_, close)

        # vwap + open
        vwap_open_sum = operator.add(vwap, open_)

        # (open + close) - (vwap + open)
        price_diff = operator.sub(open_close_sum, vwap_open_sum)

        # rank(price_diff)
        second_part = operator.rank(price_diff)

        # lt(first_part, second_part)
        condition = operator.lt(first_part, second_part)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
