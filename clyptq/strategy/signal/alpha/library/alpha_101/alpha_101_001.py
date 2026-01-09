"""Alpha 101_001: Volatility-based ranking signal.

Formula: sub(rank(ts_argmax(pow(condition(lt(returns,0),ts_std(returns,20),close),2.),5)),0.5)

Argmax ranking of conditional volatility vs price based on returns
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_001(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_001: Volatility-based conditional ranking.

    When returns are negative, uses volatility; otherwise uses close price.
    Then ranks the argmax of squared values.
    """

    default_params = {"std_window": 20, "argmax_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_001"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_001."""
        close = data["close"]

        # Calculate returns using operator
        returns = operator.ts_returns(close)

        # condition(lt(returns, 0), ts_std(returns, 20), close)
        returns_std = operator.ts_std(returns, self.params["std_window"])
        returns_negative = operator.lt(returns, 0)
        condition_result = operator.condition(returns_negative, returns_std, close)

        # pow(..., 2)
        powered = operator.pow(condition_result, 2)

        # ts_argmax(..., 5)
        argmax = operator.ts_argmax(powered, self.params["argmax_window"])

        # rank(...)
        ranked = operator.rank(argmax)

        # sub(..., 0.5)
        alpha = operator.sub(ranked, 0.5)

        return operator.ts_fillna(alpha, 0)
