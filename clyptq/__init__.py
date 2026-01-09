"""
clyptq - Quantitative Trading Framework

Usage:
    ```python
    from clyptq import Strategy, Engine, operator
    from clyptq.universe import DynamicUniverse
    from clyptq.data.spec import OHLCVSpec

    class MyStrategy(Strategy):
        universe = DynamicUniverse(symbols=["BTC", "ETH"])
        data = {"ohlcv": OHLCVSpec(timeframe="1h")}
        rebalance_freq = "1d"

        def compute_signal(self):
            return operator.rank(self.provider["close"])

        def warmup_periods(self) -> int:
            return 50

    # Run backtest
    engine = Engine()
    result = engine.run(
        MyStrategy(),
        mode="backtest",
        data_path="data/crypto/",
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1),
    )
    ```
"""

__version__ = "1.0.0"

from clyptq.core.types import (
    Constraints,
    CostModel,
    EngineMode,
    BacktestResult,
    Position,
    Fill,
    Order,
)

# Engine (main interface)
from clyptq.trading.engine import Engine

# Strategy classes
from clyptq.strategy.base import Strategy

# Transforms
from clyptq.strategy.transform import BaseTransform

# Data specs
from clyptq.data.spec import OHLCVSpec, OrderBookSpec, FundingSpec

# Global operators
from clyptq import operator

# Universe
from clyptq import universe
from clyptq.universe import (
    Universe,
    BaseFilter,
    StaticUniverse,
    DynamicUniverse,
)
# Legacy alias
BaseUniverse = Universe

__all__ = [
    "__version__",
    "EngineMode",
    "Constraints",
    "CostModel",
    "BacktestResult",
    "Position",
    "Fill",
    "Order",
    "Engine",
    "Strategy",
    "BaseTransform",
    "OHLCVSpec",
    "OrderBookSpec",
    "FundingSpec",
    "operator",
    "universe",
    "Universe",
    "BaseFilter",
    "BaseUniverse",  # Legacy alias
    "StaticUniverse",
    "DynamicUniverse",
]
