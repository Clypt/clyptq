"""Alpha 101_060: Price position volume vs argmax signal.

Formula: sub(0,mul(1,sub(mul(2,twise_a_scale(rank(mul(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low})),{disk:volume})))),twise_a_scale(rank(ts_argmax({disk:close},10))))))

Negative of difference between scaled price position-volume rank and scaled argmax rank.
"""

from typing import Optional


from clyptq.strategy.signal.base import BaseSignal, SignalRole
from clyptq import operator


class alpha_101_060(BaseSignal):
    role = SignalRole.ALPHA
    """Alpha 101_060: Price position volume vs argmax.

    Negates the difference between scaled price position-volume product rank and scaled argmax rank.
    """

    default_params = {"argmax_window": 10}

    @property
    def name(self) -> str:
        return "alpha_101_060"

    def calculate(
        self,
        data,
        pair: Optional[str] = None,
    ):
        """Calculate Alpha 101_060."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Part 1: Price position indicator
        # close - low
        close_low = operator.sub(close, low)

        # high - close
        high_close = operator.sub(high, close)

        # (close - low) - (high - close)
        price_position_num = operator.sub(close_low, high_close)

        # high - low
        high_low = operator.sub(high, low)

        # Price position ratio
        price_position = operator.div(price_position_num, high_low)

        # mul(price_position, volume)
        position_volume = operator.mul(price_position, volume)

        # rank(position_volume)
        position_rank = operator.rank(position_volume)

        # twise_a_scale(position_rank)
        scaled_position = operator.twise_a_scale(position_rank, 1)

        # mul(2, scaled_position)
        first_part = operator.mul(2, scaled_position)

        # Part 2: Argmax position
        # ts_argmax(close, 10)
        argmax = operator.ts_argmax(close, self.params["argmax_window"])

        # rank(argmax)
        argmax_rank = operator.rank(argmax)

        # twise_a_scale(argmax_rank)
        second_part = operator.twise_a_scale(argmax_rank, 1)

        # sub(first_part, second_part)
        diff = operator.sub(first_part, second_part)

        # Negate
        alpha = operator.mul(diff, -1)

        return operator.ts_fillna(alpha, 0)
