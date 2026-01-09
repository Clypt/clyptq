"""Alpha 101_025: Returns-amount-VWAP composite signal.

Formula: rank(mul(mul(mul(mul(-1,{disk:returns}),ts_mean({disk:amount},20)),{disk:vwap}),sub({disk:high},{disk:close})))

Ranking of composite factor: returns, average amount, VWAP, and high-close spread.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_025(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_025: Returns-amount-VWAP composite.

    Ranks product of negative returns, average amount, VWAP, and high-close spread.
    """

    default_params = {"amount_window": 20}

    @property
    def name(self) -> str:
        return "alpha_101_025"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_025."""
        close = data["close"]
        high = data["high"]
        volume = data["volume"]
        vwap = data.get("vwap")

        # Calculate VWAP if not available
        if vwap is None:
            vwap = operator.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # -1 * returns
        neg_returns = operator.mul(returns, -1)

        # ts_mean(amount, 20)
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])

        # high - close
        high_close_diff = operator.sub(high, close)

        # Multiply all components
        product = operator.mul(
            operator.mul(operator.mul(neg_returns, amount_mean), vwap),
            high_close_diff,
        )

        # rank(product)
        alpha = operator.rank(product)

        return operator.ts_fillna(alpha, 0)
