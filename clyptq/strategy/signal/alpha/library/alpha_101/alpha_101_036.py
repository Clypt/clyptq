"""Alpha 101_036: Weighted multi-factor composite signal.

Formula: add(add(add(add(mul(2.21,rank(ts_corr(sub({disk:close},{disk:open}),delay({disk:volume},1),15))),mul(0.7,rank(sub({disk:open},{disk:close})))),mul(0.73,rank(ts_rank(delay(mul(-1,{disk:returns}),6),5)))),rank(abs(ts_corr({disk:vwap},ts_mean({disk:amount},20),6)))),mul(0.6,rank(mul(sub(div(ts_sum({disk:close},200),200),{disk:open}),sub({disk:close},{disk:open})))))

Weighted combination of price-volume correlations, returns, and momentum factors.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_036(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_036: Weighted multi-factor composite.

    Combines multiple weighted factors: price-volume correlation, open-close diff,
    delayed returns, VWAP-amount correlation, and momentum.
    """

    default_params = {}

    @property
    def name(self) -> str:
        return "alpha_101_036"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_036."""
        close = data["close"]
        open_ = data["open"]
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

        # Part 1: 2.21 * rank(ts_corr(close-open, delay(volume,1), 15))
        close_open_diff = operator.sub(close, open_)
        volume_lag = operator.delay(volume, 1)
        corr_15 = operator.ts_corr(close_open_diff, volume_lag, 15)
        part1 = operator.mul(2.21, operator.rank(corr_15))

        # Part 2: 0.7 * rank(open - close)
        open_close_diff = operator.sub(open_, close)
        part2 = operator.mul(0.7, operator.rank(open_close_diff))

        # Part 3: 0.73 * rank(ts_rank(delay(-returns, 6), 5))
        neg_returns = operator.mul(returns, -1)
        returns_lag = operator.delay(neg_returns, 6)
        returns_tsrank = operator.ts_rank(returns_lag, 5)
        part3 = operator.mul(0.73, operator.rank(returns_tsrank))

        # Part 4: rank(abs(ts_corr(vwap, ts_mean(amount, 20), 6)))
        amount_mean = operator.ts_mean(amount, 20)
        vwap_amount_corr = operator.ts_corr(vwap, amount_mean, 6)
        part4 = operator.rank(operator.abs(vwap_amount_corr))

        # Part 5: 0.6 * rank((close_mean_200 - open) * (close - open))
        close_mean_200 = operator.div(operator.ts_sum(close, 200), 200)
        mean_open_diff = operator.sub(close_mean_200, open_)
        close_open_product = operator.mul(mean_open_diff, close_open_diff)
        part5 = operator.mul(0.6, operator.rank(close_open_product))

        # Sum all parts
        alpha = operator.add(
            operator.add(operator.add(operator.add(part1, part2), part3), part4), part5
        )

        return operator.ts_fillna(alpha, 0)
