"""Live and paper trading engine.

Supports:
- Paper mode: Simulated execution with live prices
- Live mode: Real execution on exchange

Uses DataProvider for unified data access (same interface as backtest/research).
Strategy accesses data via self.provider.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from clyptq.strategy.base import Strategy
from clyptq.trading.execution.base import Executor
from clyptq.core.types import EngineMode, ExecutionResult, Fill, Order, OrderSide, Snapshot
from clyptq.trading.portfolio.state import PortfolioState
from clyptq.infra.utils import get_logger, GracefulShutdown


class LiveEngine:
    """Live and paper trading engine.

    Uses DataProvider for unified data access (same interface as backtest/research).
    Strategy accesses data via self.provider.

    Example:
        ```python
        # Create strategy with provider
        strategy = MyStrategy(provider=provider)

        engine = LiveEngine(
            strategy=strategy,
            executor=CCXTExecutor(...),
            mode=EngineMode.PAPER,
        )

        engine.run_live(interval_seconds=60)
        ```
    """

    def __init__(
        self,
        strategy: Strategy,
        executor: Executor,
        initial_capital: float = 10000.0,
        mode: EngineMode = EngineMode.PAPER,
        shutdown_handler: Optional[GracefulShutdown] = None,
    ):
        """Initialize LiveEngine.

        Args:
            strategy: Strategy to run (must have loaded provider)
            executor: Order executor (CCXTExecutor, etc.)
            initial_capital: Starting capital
            mode: PAPER or LIVE
            shutdown_handler: For graceful shutdown
        """
        if mode not in [EngineMode.LIVE, EngineMode.PAPER]:
            raise ValueError(f"LiveEngine only supports LIVE or PAPER modes, got {mode}")

        self.strategy = strategy
        self.provider = strategy.provider  # Get from strategy
        self.executor = executor
        self.mode = mode
        self.portfolio = PortfolioState(initial_capital)
        self.shutdown_handler = shutdown_handler or GracefulShutdown()

        self.logger = get_logger(
            __name__,
            context={
                "mode": mode.value,
                "strategy": strategy.name,
            },
        )

        self.snapshots: List[Snapshot] = []
        self.trades: List[Fill] = []

    def run_live(self, interval_seconds: int = 60, verbose: bool = True) -> None:
        """Real-time trading loop.

        Args:
            interval_seconds: Seconds between each tick
            verbose: Log progress
        """
        symbols = self.provider.symbols
        if not symbols:
            raise ValueError("DataProvider has no symbols loaded")

        self.logger.info(
            "Live trading started",
            extra={
                "mode": self.mode.value,
                "symbols": symbols,
                "interval_seconds": interval_seconds,
                "warmup_periods": self.strategy.warmup_periods(),
            },
        )

        iteration = 0

        try:
            while not self.shutdown_handler.is_shutdown_requested():
                timestamp = datetime.utcnow()

                try:
                    # Get prices from DataProvider
                    prices = self.provider.current_prices()

                    if not prices:
                        self.logger.warning(
                            "No prices available",
                            extra={"timestamp": timestamp.isoformat()},
                        )
                        time.sleep(interval_seconds)
                        continue

                    # Process bar
                    result = self._process_bar(timestamp, prices)

                    # Log result
                    if verbose:
                        self._log_result(result, iteration)

                except Exception as e:
                    self.logger.error(
                        "Error processing bar",
                        extra={
                            "timestamp": timestamp.isoformat(),
                            "error": str(e),
                        },
                    )

                iteration += 1
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Live trading stopped by user")
        finally:
            self.logger.info(
                "Live trading ended",
                extra={
                    "total_trades": len(self.trades),
                    "final_equity": self.snapshots[-1].equity if self.snapshots else 0,
                },
            )

    def _process_bar(self, timestamp: datetime, prices: Dict[str, float]) -> ExecutionResult:
        """Process a single bar."""
        # Record snapshot
        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

        # Handle delisted positions
        universe_symbols = self.provider.universe_symbols
        self._liquidate_delisted(timestamp, universe_symbols, prices)

        # Check rebalance schedule
        if not self.provider.should_rebalance():
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="schedule",
            )

        # Mark rebalanced
        self.provider.mark_rebalanced()

        # Call strategy.on_bar()
        weights = self.strategy.on_bar(timestamp=timestamp)

        if not weights:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_weights",
            )

        # Convert weights to orders
        orders = self._weights_to_orders(weights, prices)

        if not orders:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_orders",
            )

        # Execute orders
        fills = self.executor.execute(orders, timestamp, prices)

        # Apply fills
        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                self.logger.warning(f"Fill rejected: {e}")

        final_snapshot = self.portfolio.get_snapshot(timestamp, prices)

        return ExecutionResult(
            timestamp=timestamp,
            action="rebalance",
            fills=fills,
            orders=orders,
            snapshot=final_snapshot,
            rebalance_reason="scheduled",
        )

    def _weights_to_orders(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Order]:
        """Convert target weights to orders."""
        orders = []
        equity = self.portfolio.equity(prices)

        # Reserve for fees
        fee_reserve = 0.998

        for symbol, target_weight in weights.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            target_value = equity * target_weight * fee_reserve
            target_amount = target_value / price

            current_amount = 0.0
            if symbol in self.portfolio.positions:
                current_amount = self.portfolio.positions[symbol].amount

            delta = target_amount - current_amount

            if abs(delta) < 1e-8:
                continue

            if delta > 0:
                orders.append(Order(symbol=symbol, side=OrderSide.BUY, amount=delta))
            else:
                orders.append(Order(symbol=symbol, side=OrderSide.SELL, amount=abs(delta)))

        # Sells first, then buys
        sells = [o for o in orders if o.side == OrderSide.SELL]
        buys = [o for o in orders if o.side == OrderSide.BUY]

        return sells + buys

    def _liquidate_delisted(
        self,
        timestamp: datetime,
        available: List[str],
        prices: Dict[str, float],
    ) -> None:
        """Liquidate positions in delisted symbols."""
        if not self.portfolio.positions:
            return

        available_set = set(available)
        delisted = [
            symbol
            for symbol in self.portfolio.positions.keys()
            if symbol not in available_set
        ]

        if not delisted:
            return

        orders = [
            Order(
                symbol=symbol,
                side=OrderSide.SELL,
                amount=abs(self.portfolio.positions[symbol].amount),
            )
            for symbol in delisted
            if symbol in prices
        ]

        if not orders:
            return

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
                self.logger.info(f"Liquidated delisted: {fill.symbol}")
            except ValueError as e:
                self.logger.error(f"Failed to liquidate {fill.symbol}: {e}")

    def _log_result(self, result: ExecutionResult, iteration: int) -> None:
        """Log execution result."""
        ts_str = result.timestamp.strftime('%H:%M:%S')

        if result.action == "rebalance":
            self.logger.info(
                f"[{ts_str}] REBALANCE",
                extra={
                    "fills": len(result.fills),
                    "equity": f"{result.snapshot.equity:.2f}",
                    "positions": result.snapshot.num_positions,
                },
            )

            for fill in result.fills:
                side = "BUY" if fill.side == OrderSide.BUY else "SELL"
                self.logger.info(
                    f"  {side} {fill.symbol}: {fill.amount:.6f} @ ${fill.price:,.2f}",
                )

        elif iteration % 10 == 0:
            self.logger.info(
                f"[{ts_str}] skip ({result.rebalance_reason})",
                extra={"equity": f"{result.snapshot.equity:.2f}"},
            )

    def reset(self) -> None:
        """Reset engine state."""
        self.portfolio.reset()
        self.snapshots.clear()
        self.trades.clear()

    def __repr__(self) -> str:
        return (
            f"LiveEngine("
            f"strategy={self.strategy.name}, "
            f"mode={self.mode.value})"
        )
