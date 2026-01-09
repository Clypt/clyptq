"""Live trading executors by asset class.

Structure:
    live/
    ├── base.py          # LiveExecutor ABC
    ├── crypto/          # Cryptocurrency executors
    │   └── ccxt.py      # CCXTExecutor (Binance, Coinbase, etc.)
    └── stock/           # Stock broker executors (planned)
"""

from clyptq.trading.execution.live.base import LiveExecutor

__all__ = [
    "LiveExecutor",
]

# Optional: ccxt executor
try:
    from clyptq.trading.execution.live.crypto import CCXTExecutor
    __all__.append("CCXTExecutor")
except ImportError:
    pass
