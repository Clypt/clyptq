"""Test strategy."""

from clypt import Constraints
from clypt.factors.library.momentum import MomentumFactor
from clypt.factors.library.volatility import VolatilityFactor
from clypt.portfolio.construction import TopNConstructor
from clypt.strategy.base import SimpleStrategy


class TestStrategy(SimpleStrategy):
    """Simple test strategy."""

    def __init__(self):
        factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
        ]

        constraints = Constraints(
            max_position_size=0.6,
            max_gross_exposure=1.0,
            min_position_size=0.1,
            max_num_positions=2,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=2),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="TestStrategy",
        )
