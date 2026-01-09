"""Alpha 101_072: Mid price-amount correlation decay rank ratio signal.

Formula: div(rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:amount},40),8.93345),10.1519)),rank(ts_decayed_linear(ts_corr(ts_rank({disk:vwap},3.72469),ts_rank({disk:volume},18.5188),6.86671),2.95011)))

Ratio of mid price-amount correlation decay rank to VWAP-volume ts_rank correlation decay rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_072(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_072: Mid price-amount correlation decay rank ratio.

    Divides mid price-amount correlation decay rank by VWAP-volume correlation decay rank.
    """

    default_params = {
        "amount_window": 40,
        "corr_window1": 9,
        "decay_window1": 10,
        "vwap_rank_window": 4,
        "volume_rank_window": 19,
        "corr_window2": 7,
        "decay_window2": 3,
    }

    @property
    def name(self) -> str:
        return "alpha_101_072"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_072."""
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

        # Part 1 (Numerator): Mid price-amount correlation decay rank
        # Mid price: (high + low) / 2
        mid_price = operator.div(operator.add(high, low), 2)

        # ts_mean(amount, 40)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # ts_corr(mid_price, amount_mean, 9)
        corr1 = operator.ts_corr(mid_price, amount_mean, self.params["corr_window1"])

        # ts_decayed_linear(corr1, 10)
        decayed1 = operator.ts_decayed_linear(corr1, self.params["decay_window1"])

        # rank(decayed1)
        numerator = operator.rank(decayed1)

        # Part 2 (Denominator): VWAP-volume ts_rank correlation decay rank
        # ts_rank(vwap, 4)
        vwap_tsrank = operator.ts_rank(vwap, self.params["vwap_rank_window"])

        # ts_rank(volume, 19)
        volume_tsrank = operator.ts_rank(volume, self.params["volume_rank_window"])

        # ts_corr(vwap_tsrank, volume_tsrank, 7)
        corr2 = operator.ts_corr(vwap_tsrank, volume_tsrank, self.params["corr_window2"])

        # ts_decayed_linear(corr2, 3)
        decayed2 = operator.ts_decayed_linear(corr2, self.params["decay_window2"])

        # rank(decayed2)
        denominator = operator.rank(decayed2)

        # div(numerator, denominator)
        alpha = operator.div(numerator, denominator)

        return operator.ts_fillna(alpha, 0)
