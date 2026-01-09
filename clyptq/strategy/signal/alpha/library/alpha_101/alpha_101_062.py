"""Alpha 101_062: VWAP-amount correlation vs price rank signal.

Formula: mul(lt(rank(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},20),22.4101),9.91009)),rank(lt(add(rank({disk:open}),rank({disk:open})),add(rank(div(add({disk:high},{disk:low}),2)),rank({disk:high}))))),-1)

Negative conditional comparing VWAP-amount correlation rank with price ranks.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_062(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_062: VWAP-amount correlation vs price rank.

    Negates the comparison between VWAP-amount correlation rank and price rank conditions.
    """

    default_params = {"amount_window": 20, "sum_window": 22, "corr_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_062"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_062."""
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(high, low, close, volume)

        # Calculate amount if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # Part 1: VWAP-amount correlation
        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 22)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(vwap, amount_sum, 10)
        vwap_corr = operator.ts_corr(vwap, amount_sum, self.params["corr_window"])

        # rank(vwap_corr)
        first_rank = operator.rank(vwap_corr)

        # Part 2: Price rank comparison
        # rank(open)
        open_rank = operator.rank(open_)

        # 2 * open_rank
        open_double = operator.add(open_rank, open_rank)

        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # rank(mid_price)
        mid_rank = operator.rank(mid_price)

        # rank(high)
        high_rank = operator.rank(high)

        # mid_rank + high_rank
        price_sum = operator.add(mid_rank, high_rank)

        # lt(open_double, price_sum)
        price_condition = operator.lt(open_double, price_sum)

        # rank(price_condition)
        second_rank = operator.rank(price_condition.astype(float))

        # lt(first_rank, second_rank)
        main_condition = operator.lt(first_rank, second_rank)

        # mul(main_condition, -1)
        alpha = operator.mul(main_condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
