from clyptq import Constraints
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.factors.library.volatility import VolatilityFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class MomentumStrategy(SimpleStrategy):
    """
    Simple momentum strategy with volatility filter.

    Logic:
    - Rank assets by 20-day momentum
    - Filter by volatility (lower is better)
    - Hold top 3 positions
    - Rebalance daily
    """

    def __init__(self):
        factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
        ]

        constraints = Constraints(
            max_position_size=0.4,
            max_gross_exposure=1.0,
            min_position_size=0.1,
            max_num_positions=3,
            allow_short=False,
        )

        super().__init__(
            factors_list=factors,
            constructor=TopNConstructor(top_n=3),
            constraints_obj=constraints,
            schedule_str="daily",
            warmup=25,
            name="Momentum",
        )
