"""Alpha 101_048: Price change correlation with volatility signal.

Formula: div(grouped_demean(div(mul(ts_corr(ts_delta({disk:close},1),ts_delta(delay({disk:close},1),1),250),ts_delta({disk:close},1)),{disk:close}),{disk:industry_group_lv3}),ts_sum(pow(div(ts_delta({disk:close},1),delay({disk:close},1)),2),250))

Industry-demeaned price change correlation divided by cumulative volatility.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_048(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_048: Price change correlation with volatility.

    Divides demeaned price change correlation by cumulative squared returns.
    """

    default_params = {"corr_window": 250, "vol_window": 250}

    @property
    def name(self) -> str:
        return "alpha_101_048"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_048."""
        close = data["close"]

        # ts_delta(close, 1)
        close_delta = operator.ts_delta(close, 1)

        # delay(close, 1)
        close_lag = operator.delay(close, 1)

        # ts_delta(delay(close, 1), 1)
        lag_delta = operator.ts_delta(close_lag, 1)

        # ts_corr(close_delta, lag_delta, 250)
        corr = operator.ts_corr(close_delta, lag_delta, self.params["corr_window"])

        # corr * close_delta / close
        corr_product = operator.mul(corr, close_delta)
        ratio = operator.div(corr_product, close)

        # Demean (cross-sectional mean removal as proxy for industry demean)
        demeaned = operator.demean(ratio)

        # Returns = close_delta / close_lag
        returns = operator.div(close_delta, close_lag)

        # Squared returns
        returns_squared = operator.pow(returns, 2)

        # ts_sum(returns_squared, 250)
        volatility = operator.ts_sum(returns_squared, self.params["vol_window"])

        # demeaned / volatility
        alpha = operator.div(demeaned, volatility)

        return operator.ts_fillna(alpha, 0)
