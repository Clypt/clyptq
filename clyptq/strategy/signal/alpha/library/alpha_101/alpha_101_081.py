"""Alpha 101_081: VWAP-amount correlation product log rank vs VWAP-volume rank correlation rank signal.

Formula: mul(lt(rank(log(ts_product(rank(pow(rank(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},10),49.6054),8.47743)),4)),14.9655))),rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),5.07914))),-1)

Negative comparison between VWAP-amount correlation product log rank and VWAP-volume rank correlation rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_081(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_081: VWAP-amount correlation product log rank vs VWAP-volume rank correlation rank.

    Negates the comparison between two complex correlation rank values.
    """

    default_params = {
        "amount_window": 10,
        "sum_window": 50,
        "corr_window1": 8,
        "product_window": 15,
        "corr_window2": 5,
    }

    @property
    def name(self) -> str:
        return "alpha_101_081"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_081."""
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

        # Part 1: VWAP-amount correlation product log rank
        # ts_mean(amount, 10)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 50)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(vwap, amount_sum, 8)
        vwap_corr = operator.ts_corr(vwap, amount_sum, self.params["corr_window1"])

        # rank(vwap_corr)
        corr_rank = operator.rank(vwap_corr)

        # pow(corr_rank, 4)
        corr_powered = operator.pow(corr_rank, 4)

        # rank(corr_powered)
        powered_rank = operator.rank(corr_powered)

        # ts_product(powered_rank, 15)
        product_result = operator.ts_product(powered_rank, self.params["product_window"])

        # log(product_result)
        log_result = operator.log(product_result)

        # rank(log_result)
        first_rank = operator.rank(log_result)

        # Part 2: VWAP-volume rank correlation
        # rank(vwap)
        vwap_rank = operator.rank(vwap)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(vwap_rank, volume_rank, 5)
        second_corr = operator.ts_corr(vwap_rank, volume_rank, self.params["corr_window2"])

        # rank(second_corr)
        second_rank = operator.rank(second_corr)

        # lt(first_rank, second_rank)
        condition = operator.lt(first_rank, second_rank)

        # mul(condition, -1)
        alpha = operator.mul(condition.astype(float), -1)

        return operator.ts_fillna(alpha, 0)
