"""Alpha 101_035: Volume-price-returns time-series ranking signal.

Formula: mul(mul(ts_rank({disk:volume},32),sub(1,ts_rank(sub(add({disk:close},{disk:high}),{disk:low}),16))),sub(1,ts_rank({disk:returns},32)))

Combines time-series rankings of volume, price position, and returns.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_035(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_035: Volume-price-returns time-series ranking.

    Multiplies volume rank with inverse price position and inverse returns ranks.
    """

    default_params = {"volume_window": 32, "price_window": 16, "returns_window": 32}

    @property
    def name(self) -> str:
        return "alpha_101_035"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_035."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # ts_rank(volume, 32)
        volume_rank = operator.ts_rank(volume, self.params["volume_window"])

        # (close + high) - low
        price_range = operator.sub(operator.add(close, high), low)

        # 1 - ts_rank(price_range, 16)
        price_rank = operator.ts_rank(price_range, self.params["price_window"])
        one_minus_price = operator.sub(1, price_rank)

        # 1 - ts_rank(returns, 32)
        returns_rank = operator.ts_rank(returns, self.params["returns_window"])
        one_minus_returns = operator.sub(1, returns_rank)

        # Multiply all components
        alpha = operator.mul(
            operator.mul(volume_rank, one_minus_price), one_minus_returns
        )

        return operator.ts_fillna(alpha, 0)
