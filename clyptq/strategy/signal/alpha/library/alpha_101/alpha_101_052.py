"""Alpha 101_052: Low minimum change with returns and volume signal.

Formula: mul(mul(add(mul(-1,ts_min({disk:low},5)),delay(ts_min({disk:low},5),5)),rank(div(sub(ts_sum({disk:returns},240),ts_sum({disk:returns},20)),220))),ts_rank({disk:volume},5))

Combines 5-period low minimum change, long-short returns difference, and volume rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_052(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_052: Low minimum change with returns and volume.

    Multiplies low minimum change by returns difference rank and volume time-series rank.
    """

    default_params = {
        "low_window": 5,
        "returns_long": 240,
        "returns_short": 20,
        "volume_window": 5,
    }

    @property
    def name(self) -> str:
        return "alpha_101_052"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_052."""
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.returns(close)

        # Part 1: Low minimum change
        # ts_min(low, 5)
        low_min = operator.ts_min(low, self.params["low_window"])

        # -ts_min(low, 5)
        neg_low_min = operator.mul(low_min, -1)

        # delay(ts_min(low, 5), 5)
        low_min_delayed = operator.delay(low_min, self.params["low_window"])

        # add(-low_min, low_min_delayed)
        low_change = operator.add(neg_low_min, low_min_delayed)

        # Part 2: Long-short returns difference
        # ts_sum(returns, 240)
        returns_long = operator.ts_sum(returns, self.params["returns_long"])

        # ts_sum(returns, 20)
        returns_short = operator.ts_sum(returns, self.params["returns_short"])

        # sub(returns_long, returns_short)
        returns_diff = operator.sub(returns_long, returns_short)

        # div(returns_diff, 220)
        returns_avg_diff = operator.div(
            returns_diff, self.params["returns_long"] - self.params["returns_short"]
        )

        # rank(returns_avg_diff)
        returns_rank = operator.rank(returns_avg_diff)

        # Part 3: Volume rank
        # ts_rank(volume, 5)
        volume_rank = operator.ts_rank(volume, self.params["volume_window"])

        # Multiply all parts
        product = operator.mul(low_change, returns_rank)
        alpha = operator.mul(product, volume_rank)

        return operator.ts_fillna(alpha, 0)
