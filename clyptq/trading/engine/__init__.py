"""Trading engines.

Usage:
    ```python
    from clyptq import Engine

    # Backtest
    engine = Engine()
    result = engine.run(strategy, mode="backtest", data_path="data/", ...)

    # Paper trading
    result = engine.run(strategy, mode="paper")

    # Live trading
    result = engine.run(strategy, mode="live")
    ```
"""

# Unified Engine (main interface)
from clyptq.trading.engine.engine import Engine

# Internal engines (for advanced use)
from clyptq.trading.engine.backtest import BacktestEngine
from clyptq.trading.engine.live import LiveEngine

__all__ = [
    # Main interface
    "Engine",
    # Internal (for advanced use)
    "BacktestEngine",
    "LiveEngine",
]
