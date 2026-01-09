"""Alpha 101_078: Weighted low-VWAP amount correlation rank power by VWAP-volume rank correlation rank signal.

Formula: pow(rank(ts_corr(ts_sum(add(mul({disk:low},0.352233),mul({disk:vwap},sub(1,0.352233))),19.7428),ts_sum(ts_mean({disk:amount},40),19.7428),6.83313)),rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),5.77492)))

Weighted low-VWAP amount sum correlation rank raised to VWAP-volume rank correlation rank power.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_078(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_078: Weighted low-VWAP amount correlation rank power by VWAP-volume rank correlation rank.

    Raises weighted price-amount correlation rank to VWAP-volume correlation rank power.
    """

    default_params = {
        "weight": 0.352233,
        "sum_window": 20,
        "amount_window": 40,
        "corr_window1": 7,
        "corr_window2": 6,
    }

    @property
    def name(self) -> str:
        return "alpha_101_078"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_078."""
        low = data["low"]
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

        # Part 1: Weighted low-VWAP amount sum correlation
        # Weighted low-VWAP: low * weight + vwap * (1-weight)
        low_part = operator.mul(low, weight)
        vwap_part = operator.mul(vwap, 1 - weight)
        weighted_low_vwap = operator.add(low_part, vwap_part)

        # ts_sum(weighted_low_vwap, 20)
        weighted_sum = operator.ts_sum(weighted_low_vwap, self.params["sum_window"])

        # ts_mean(amount, 40)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_sum(amount_mean, 20)
        amount_sum = operator.ts_sum(amount_mean, self.params["sum_window"])

        # ts_corr(weighted_sum, amount_sum, 7)
        first_corr = operator.ts_corr(weighted_sum, amount_sum, self.params["corr_window1"])

        # rank(first_corr)
        base_rank = operator.rank(first_corr)

        # Part 2: VWAP-volume rank correlation
        # rank(vwap)
        vwap_rank = operator.rank(vwap)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # ts_corr(vwap_rank, volume_rank, 6)
        second_corr = operator.ts_corr(vwap_rank, volume_rank, self.params["corr_window2"])

        # rank(second_corr)
        power_rank = operator.rank(second_corr)

        # pow(base_rank, power_rank)
        alpha = operator.pow(base_rank, power_rank)

        return operator.ts_fillna(alpha, 0)
