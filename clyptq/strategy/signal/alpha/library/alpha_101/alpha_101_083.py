"""Alpha 101_083: Delayed range ratio rank times double volume rank ratio signal.

Formula: div(mul(rank(delay(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),2)),rank(rank({disk:volume}))),div(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),sub({disk:vwap},{disk:close})))

Ratio of delayed range ratio rank times double volume rank to current range ratio divided by VWAP-close difference.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_083(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_083: Delayed range ratio rank times double volume rank ratio.

    Complex ratio involving price range, average close, and VWAP-close difference.
    """

    default_params = {"mean_window": 5, "delay_period": 2}

    @property
    def name(self) -> str:
        return "alpha_101_083"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_083."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(high, low, close, volume)

        # Common calculation: range ratio
        # high - low
        high_low_range = operator.sub(high, low)

        # ts_sum(close, 5) / 5 - 5-day mean
        close_sum = operator.ts_sum(close, self.params["mean_window"])
        close_mean = operator.div(close_sum, self.params["mean_window"])

        # (high - low) / close_mean
        range_ratio = operator.div(high_low_range, close_mean)

        # Numerator
        # delay(range_ratio, 2)
        delayed_ratio = operator.delay(range_ratio, self.params["delay_period"])

        # rank(delayed_ratio)
        delayed_rank = operator.rank(delayed_ratio)

        # rank(volume)
        volume_rank = operator.rank(volume)

        # rank(volume_rank) - double rank
        double_volume_rank = operator.rank(volume_rank)

        # delayed_rank * double_volume_rank
        numerator = operator.mul(delayed_rank, double_volume_rank)

        # Denominator
        # vwap - close
        vwap_close_diff = operator.sub(vwap, close)

        # range_ratio / vwap_close_diff
        denominator = operator.div(range_ratio, vwap_close_diff)

        # numerator / denominator
        alpha = operator.div(numerator, denominator)

        return operator.ts_fillna(alpha, 0)
