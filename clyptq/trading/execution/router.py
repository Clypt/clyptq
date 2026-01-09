"""
Multi-asset order router.

Routes orders to appropriate executors based on symbol/asset class.
Supports trading across multiple asset classes (crypto, stocks, forex)
with different brokers simultaneously.
"""

from datetime import datetime
from typing import Dict, List, Optional

from clyptq.trading.execution.base import Executor
from clyptq.trading.execution.live.base import LiveExecutor
from clyptq.core.types import Fill, Order
from clyptq.infra.utils import get_logger


class MultiAssetRouter(Executor):
    """Routes orders to appropriate executors based on symbol.

    Enables trading across multiple asset classes with different brokers.
    For example: crypto on Binance, stocks on IBKR.

    Example:
        ```python
        # Create executors for different asset classes
        crypto_executor = CCXTExecutor(exchange_id="binance", ...)
        stock_executor = IBKRExecutor(...)

        # Create router with symbol mappings
        router = MultiAssetRouter()
        router.register_executor(crypto_executor, ["BTC/USDT", "ETH/USDT"])
        router.register_executor(stock_executor, ["AAPL", "GOOGL"])

        # Or auto-register all symbols from executor
        router.register_executor_auto(crypto_executor)

        # Execute orders - router distributes to correct executor
        fills = router.execute(orders, timestamp, prices)
        ```

    Attributes:
        executors: Dict mapping asset_class to executor
        symbol_to_executor: Dict mapping symbol to executor
    """

    def __init__(self):
        self.executors: Dict[str, LiveExecutor] = {}
        self.symbol_to_executor: Dict[str, LiveExecutor] = {}
        self.logger = get_logger(__name__)

    def register_executor(
        self,
        executor: LiveExecutor,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """Register an executor for specific symbols.

        Args:
            executor: LiveExecutor to register
            symbols: List of symbols this executor handles.
                     If None, uses executor.get_supported_symbols()
        """
        asset_class = executor.asset_class
        exchange_id = executor.exchange_id

        # Store by asset class (may have multiple per class)
        key = f"{asset_class}:{exchange_id}"
        self.executors[key] = executor

        # Map symbols to executor
        if symbols is None:
            symbols = executor.get_supported_symbols()

        for symbol in symbols:
            if symbol in self.symbol_to_executor:
                existing = self.symbol_to_executor[symbol]
                self.logger.warning(
                    f"Symbol {symbol} already registered to {existing.exchange_id}, "
                    f"overwriting with {exchange_id}"
                )
            self.symbol_to_executor[symbol] = executor

        self.logger.info(
            "Executor registered",
            extra={
                "asset_class": asset_class,
                "exchange": exchange_id,
                "symbols_count": len(symbols),
            },
        )

    def register_executor_auto(self, executor: LiveExecutor) -> None:
        """Auto-register executor with all its supported symbols."""
        self.register_executor(executor, symbols=None)

    def get_executor_for_symbol(self, symbol: str) -> Optional[LiveExecutor]:
        """Get executor that can trade this symbol."""
        return self.symbol_to_executor.get(symbol)

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """Execute orders, routing each to appropriate executor.

        Orders are grouped by executor for efficiency, then executed.
        Returns combined fills from all executors.

        Args:
            orders: Orders to execute
            timestamp: Execution timestamp
            prices: Current prices {symbol: price}

        Returns:
            Combined fills from all executors
        """
        if not orders:
            return []

        # Group orders by executor
        orders_by_executor: Dict[str, List[Order]] = {}
        unroutable_symbols: List[str] = []

        for order in orders:
            executor = self.get_executor_for_symbol(order.symbol)
            if executor is None:
                unroutable_symbols.append(order.symbol)
                continue

            key = f"{executor.asset_class}:{executor.exchange_id}"
            if key not in orders_by_executor:
                orders_by_executor[key] = []
            orders_by_executor[key].append(order)

        if unroutable_symbols:
            self.logger.warning(
                "Orders skipped - no executor for symbols",
                extra={"symbols": unroutable_symbols},
            )

        # Execute orders on each executor
        all_fills: List[Fill] = []

        for key, executor_orders in orders_by_executor.items():
            executor = self.executors[key]

            # Filter prices to only include symbols for this executor
            executor_prices = {
                symbol: price
                for symbol, price in prices.items()
                if symbol in [o.symbol for o in executor_orders]
            }

            try:
                fills = executor.execute(executor_orders, timestamp, executor_prices)
                all_fills.extend(fills)

                self.logger.info(
                    "Orders executed",
                    extra={
                        "executor": key,
                        "orders": len(executor_orders),
                        "fills": len(fills),
                    },
                )

            except Exception as e:
                self.logger.error(
                    "Executor failed",
                    extra={"executor": key, "error": str(e)},
                )

        return all_fills

    def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        """Get balances from all executors.

        Returns:
            {executor_key: {currency: balance}}
        """
        balances = {}
        for key, executor in self.executors.items():
            try:
                # Default currencies by asset class
                if executor.asset_class == "crypto":
                    currencies = ["USDT", "BTC", "ETH"]
                else:
                    currencies = ["USD"]

                balances[key] = {
                    currency: executor.get_balance(currency) for currency in currencies
                }
            except Exception as e:
                self.logger.error(f"Failed to get balance from {key}: {e}")
                balances[key] = {}

        return balances

    def get_all_positions(self) -> Dict[str, Dict[str, Dict]]:
        """Get positions from all executors.

        Returns:
            {executor_key: {symbol: position_info}}
        """
        positions = {}
        for key, executor in self.executors.items():
            try:
                positions[key] = executor.fetch_positions()
            except Exception as e:
                self.logger.error(f"Failed to get positions from {key}: {e}")
                positions[key] = {}

        return positions

    def close_all(self) -> None:
        """Close all executor connections."""
        for key, executor in self.executors.items():
            try:
                executor.close()
                self.logger.info(f"Closed executor: {key}")
            except Exception as e:
                self.logger.error(f"Error closing executor {key}: {e}")

    @property
    def registered_symbols(self) -> List[str]:
        """Get all registered symbols."""
        return list(self.symbol_to_executor.keys())

    @property
    def registered_executors(self) -> List[str]:
        """Get all registered executor keys."""
        return list(self.executors.keys())
