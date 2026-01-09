"""Base class for live executors.

LiveExecutor extends Executor with live-specific methods:
- get_balance: Fetch account balance
- fetch_positions: Get current positions from exchange
- close: Cleanup connections

Each asset class (crypto, stocks, forex) implements its own LiveExecutor.
"""

from abc import abstractmethod
from typing import Dict, List, Optional

from clyptq.trading.execution.base import Executor


class LiveExecutor(Executor):
    """Abstract base for live trading executors.

    Extends Executor with methods needed for live/paper trading:
    - Account balance queries
    - Position fetching
    - Connection management

    Implementations:
    - CCXTExecutor: Crypto exchanges via CCXT
    - IBKRExecutor: Interactive Brokers (stocks, options)
    - AlpacaExecutor: Alpaca (US stocks)
    """

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Return asset class identifier (e.g., 'crypto', 'stock', 'forex')."""
        pass

    @property
    @abstractmethod
    def exchange_id(self) -> str:
        """Return exchange identifier (e.g., 'binance', 'ibkr', 'alpaca')."""
        pass

    @abstractmethod
    def get_balance(self, currency: str = "USD") -> float:
        """Get available balance in specified currency.

        Args:
            currency: Currency to query (USD, USDT, etc.)

        Returns:
            Available balance
        """
        pass

    @abstractmethod
    def fetch_positions(self) -> Dict[str, Dict]:
        """Fetch current positions from exchange.

        Returns:
            {symbol: {"amount": float, "avg_price": float}}
        """
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols this executor can trade.

        Returns:
            List of tradeable symbols
        """
        pass

    def can_trade(self, symbol: str) -> bool:
        """Check if executor can trade this symbol.

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol is tradeable by this executor
        """
        return symbol in self.get_supported_symbols()

    @abstractmethod
    def close(self) -> None:
        """Close connections and cleanup resources."""
        pass
