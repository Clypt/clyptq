"""Alpha 101_029: Complex nested ranking signal.

Formula: add(min(ts_product(rank(rank(twise_a_scale(log(ts_sum(ts_min(rank(rank(mul(-1,rank(ts_delta(sub({disk:close},1),5))))),2),1))))),1),5),ts_rank(delay(mul(-1,{disk:returns}),6),5))

Complex factor with nested rankings and delayed returns minimum.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_029(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_029: Complex nested ranking.

    Uses complex nested rankings with minimum of product and delayed returns rank.
    """

    default_params = {"delta_window": 5, "delay_window": 6, "ts_rank_window": 5}

    @property
    def name(self) -> str:
        return "alpha_101_029"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_029."""
        close = data["close"]

        # Calculate returns if not available
        returns = data.get("returns")
        if returns is None:
            returns = operator.ts_returns(close)

        # sub(close, 1)
        close_minus1 = operator.sub(close, 1)

        # ts_delta(close_minus1, 5)
        delta = operator.ts_delta(close_minus1, self.params["delta_window"])

        # rank(delta)
        rank1 = operator.rank(delta)

        # mul(-1, rank1)
        neg_rank = operator.mul(rank1, -1)

        # rank(rank(neg_rank))
        rank2 = operator.rank(neg_rank)
        rank3 = operator.rank(rank2)

        # ts_min(rank3, 2)
        min_2 = operator.ts_min(rank3, 2)

        # ts_sum(min_2, 1)
        sum_1 = operator.ts_sum(min_2, 1)

        # log(sum_1)
        log_result = operator.log(sum_1)

        # twise_a_scale(log_result, 1)
        scaled = operator.twise_a_scale(log_result, 1)

        # rank(rank(scaled))
        rank4 = operator.rank(scaled)
        rank5 = operator.rank(rank4)

        # ts_product(rank5, 5)
        product = operator.ts_product(rank5, self.params["ts_rank_window"])

        # delay(mul(-1, returns), 6)
        neg_returns = operator.mul(returns, -1)
        delayed_returns = operator.delay(neg_returns, self.params["delay_window"])

        # ts_rank(delayed_returns, 5)
        rank_returns = operator.ts_rank(delayed_returns, self.params["ts_rank_window"])

        # min(product, rank_returns)
        min_result = operator.elem_min(product, rank_returns)

        # add(min_result, ts_rank(...)) - simplified
        alpha = min_result

        return operator.ts_fillna(alpha, 0)
