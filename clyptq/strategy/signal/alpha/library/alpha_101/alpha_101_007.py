"""Alpha 101_007: Volume-conditional price change signal.

Formula: condition(lt(ts_mean({disk:amount},20),{disk:volume}),mul(mul(-1,ts_rank(abs(ts_delta({disk:close},7)),60)),sign(ts_delta({disk:close},7))),mul(-1,1))

Conditional price change signal based on volume threshold.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_007(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_007: Volume-conditional price change.

    Returns momentum signal when volume exceeds average amount, otherwise -1.
    """

    default_params = {"amount_window": 20, "delta_window": 7, "rank_window": 60}

    @property
    def name(self) -> str:
        return "alpha_101_007"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_007."""
        close = data["close"]
        volume = data["volume"]

        # amount = volume * close if not available
        amount = data.get("amount")
        if amount is None:
            amount = operator.mul(volume, close)

        # ts_mean(amount, 20) < volume condition
        amount_mean = operator.ts_mean(amount, self.params["amount_window"])
        condition_check = operator.lt(amount_mean, volume)

        # ts_delta(close, 7)
        close_delta = operator.ts_delta(close, self.params["delta_window"])

        # abs(ts_delta(close, 7))
        abs_close_delta = operator.abs(close_delta)

        # ts_rank(abs_close_delta, 60)
        ts_ranked = operator.ts_rank(abs_close_delta, self.params["rank_window"])

        # -1 * ts_rank
        neg_ts_rank = operator.mul(ts_ranked, -1)

        # sign(ts_delta(close, 7))
        sign_delta = operator.sign(close_delta)

        # (-1 * ts_rank) * sign_delta
        true_value = operator.mul(neg_ts_rank, sign_delta)

        # Condition: if volume > amount_mean, use true_value, else -1
        alpha = operator.condition(condition_check, true_value, -1)

        return operator.ts_fillna(alpha, 0)
