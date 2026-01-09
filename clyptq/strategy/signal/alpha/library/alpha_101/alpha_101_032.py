"""Alpha 101_032: Mean reversion with VWAP correlation signal.

Formula: add(twise_a_scale(sub(div(ts_sum({disk:close},7),7),{disk:close})),mul(20,twise_a_scale(ts_corr({disk:vwap},delay({disk:close},5),230))))

Combines 7-day mean reversion with long-term VWAP-close correlation.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_032(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_032: Mean reversion with VWAP correlation.

    Combines scaled 7-day mean reversion with weighted VWAP-delayed close correlation.
    """

    default_params = {"mean_window": 7, "delay_window": 5, "corr_window": 230}

    @property
    def name(self) -> str:
        return "alpha_101_032"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_032."""
        close = data["close"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Part 1: 7-day mean reversion
        close_mean = operator.div(
            operator.ts_sum(close, self.params["mean_window"]),
            self.params["mean_window"],
        )
        diff = operator.sub(close_mean, close)
        first_part = operator.twise_a_scale(diff, 1)

        # Part 2: VWAP-delayed close correlation
        close_lag = operator.delay(close, self.params["delay_window"])
        corr = operator.ts_corr(vwap, close_lag, self.params["corr_window"])
        scaled_corr = operator.twise_a_scale(corr, 1)
        second_part = operator.mul(20, scaled_corr)

        # Sum parts
        alpha = operator.add(first_part, second_part)

        return operator.ts_fillna(alpha, 0)
