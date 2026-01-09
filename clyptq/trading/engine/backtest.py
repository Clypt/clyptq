"""
Backtest engine for historical simulation.

BacktestEngine processes bar events using DataProvider for unified data access.
Uses Strategy.on_bar() with DataProvider (same interface as Research/Live).
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clyptq.analytics.performance.metrics import compute_metrics
from clyptq.trading.execution.base import Executor
from clyptq.core.types import BacktestResult, EngineMode, Fill, Order, OrderSide, Snapshot
from clyptq.infra.utils import get_logger
from clyptq.trading.portfolio.state import PortfolioState, FuturesPortfolioState
from clyptq.strategy.base import Strategy

# Market type literal
MarketType = str  # "spot", "futures", "margin"


class BacktestEngine:
    """Event-driven backtesting engine.

    This engine processes timestamps sequentially, calling strategy.on_bar()
    at each bar. Uses DataProvider for unified data access (same as Research/Live).

    Supports:
    - Spot trading (leverage=1, long only)
    - Futures trading (leverage 1-125x, long/short)

    Example:
        ```python
        # Spot backtest
        engine = BacktestEngine(
            strategy=strategy,
            executor=BacktestExecutor(cost_model),
            initial_capital=100000.0,
        )

        # Futures backtest with 10x leverage
        engine = BacktestEngine(
            strategy=strategy,
            executor=BacktestExecutor(cost_model),
            initial_capital=10000.0,
            market_type="futures",
            leverage=10.0,
        )

        result = engine.run(verbose=True)
        print(f"Total Return: {result.metrics.total_return:.2%}")
        ```

    Attributes:
        strategy: Trading strategy to backtest
        provider: DataProvider (from strategy.provider)
        executor: Order executor
        portfolio: Portfolio state (PortfolioState or FuturesPortfolioState)
        market_type: "spot" or "futures"
        leverage: Leverage multiplier (1.0 for spot)
        snapshots: List of portfolio snapshots
        trades: List of executed trades
    """

    def __init__(
        self,
        strategy: Strategy,
        executor: Executor,
        initial_capital: float = 10000.0,
        market_type: MarketType = "spot",
        leverage: float = 1.0,
    ):
        """Initialize BacktestEngine.

        Args:
            strategy: Strategy to backtest (must have loaded provider)
            executor: Order executor
            initial_capital: Starting capital
            market_type: "spot" or "futures"
            leverage: Leverage for futures (1.0 for spot)
        """
        self.strategy = strategy
        self.provider = strategy.provider  # Get from strategy
        self.executor = executor
        self.market_type = market_type
        self.leverage = leverage

        # Validate
        if market_type == "spot" and leverage != 1.0:
            raise ValueError("Spot market does not support leverage (must be 1.0)")
        if leverage < 1.0:
            raise ValueError(f"Leverage must be >= 1.0, got {leverage}")

        # Create appropriate portfolio
        if market_type == "futures":
            self.portfolio = FuturesPortfolioState(initial_capital, leverage)
        else:
            self.portfolio = PortfolioState(initial_capital)

        self.logger = get_logger(
            __name__,
            context={
                "strategy": strategy.name,
                "mode": "backtest",
                "market_type": market_type,
                "leverage": leverage,
            },
        )

        self.snapshots: List[Snapshot] = []
        self.trades: List[Fill] = []

    def run(self, verbose: bool = False) -> BacktestResult:
        """Run backtest.

        Provider must already be loaded with data.

        Args:
            verbose: Print progress updates

        Returns:
            BacktestResult with performance metrics
        """
        if not self.provider._loaded:
            raise RuntimeError(
                "DataProvider not loaded. Call provider.load() before engine.run()"
            )

        warmup = self.strategy.warmup_periods()

        self.logger.info(
            "Backtest started",
            extra={
                "symbols": len(self.provider.symbols),
                "warmup": warmup,
                "system_clock": self.provider.system_clock,
                "rebalance_freq": self.provider.rebalance_freq,
            },
        )

        bar_count = 0
        warmup_done = False

        # Main tick loop using DataProvider
        while self.provider.tick():
            bar_count += 1

            # Skip warmup period
            if not warmup_done:
                if bar_count < warmup:
                    continue
                warmup_done = True

            try:
                self._process_bar()

                if verbose and bar_count % 100 == 0:
                    equity = self.snapshots[-1].equity if self.snapshots else 0
                    self.logger.info(
                        "Progress",
                        extra={
                            "bar": bar_count,
                            "timestamp": self.provider.current_timestamp.isoformat(),
                            "equity": f"{equity:.2f}",
                        },
                    )

            except Exception as e:
                self.logger.error(
                    "Error processing bar",
                    extra={
                        "timestamp": self.provider.current_timestamp.isoformat(),
                        "error": str(e),
                    },
                )
                continue

        # Compute final metrics
        if not self.snapshots:
            self.logger.warning("No snapshots generated during backtest")
            metrics = self._empty_metrics()
        else:
            metrics = compute_metrics(self.snapshots, self.trades)

        self.logger.info(
            "Backtest completed",
            extra={
                "total_bars": bar_count,
                "total_trades": len(self.trades),
                "final_equity": self.snapshots[-1].equity if self.snapshots else 0,
                "total_return": f"{metrics.total_return:.2%}" if metrics else "N/A",
            },
        )

        return BacktestResult(
            snapshots=self.snapshots,
            trades=self.trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
            mode=EngineMode.BACKTEST,
        )

    def _process_bar(self) -> None:
        """Process a single bar."""
        timestamp = self.provider.current_timestamp

        # Get current prices from DataProvider
        prices = self.provider.current_prices()
        if not prices:
            return

        # Record snapshot
        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

        # Handle delisted positions
        universe_symbols = self.provider.universe_symbols
        self._liquidate_delisted(timestamp, universe_symbols, prices)

        # Check rebalance schedule
        if not self.provider.should_rebalance():
            return

        # Mark rebalanced (updates in_universe)
        self.provider.mark_rebalanced()

        # Call strategy.on_bar()
        weights = self.strategy.on_bar(timestamp=timestamp)

        if not weights:
            return

        # Convert weights to orders
        orders = self._weights_to_orders(weights, prices)

        if not orders:
            return

        # Sort orders: SELL first (to free up cash), then BUY
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        sorted_orders = sell_orders + buy_orders

        # Execute orders
        fills = self.executor.execute(sorted_orders, timestamp, prices)

        # Apply fills
        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                self.logger.warning(f"Fill rejected: {e}")

    def _weights_to_orders(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Order]:
        """Convert target weights to orders.

        Args:
            weights: Target weights {symbol: weight}
            prices: Current prices {symbol: price}

        Returns:
            List of orders to reach target weights
        """
        orders = []
        equity = self.portfolio.equity(prices)

        for symbol, target_weight in weights.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            target_value = equity * target_weight
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

        return orders

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

    def run_monte_carlo(
        self,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """Run Monte Carlo simulation on backtest results.

        Must call run() first to generate backtest results.

        Args:
            num_simulations: Number of Monte Carlo paths
            random_seed: Random seed for reproducibility
            verbose: Print results

        Returns:
            MonteCarloResult with simulation statistics
        """
        from clyptq.analytics.risk.monte_carlo import MonteCarloSimulator

        if not self.snapshots:
            raise ValueError("No backtest results. Call run() first.")

        metrics = compute_metrics(self.snapshots, self.trades)
        backtest_result = BacktestResult(
            snapshots=self.snapshots,
            trades=self.trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
            mode=EngineMode.BACKTEST,
        )

        simulator = MonteCarloSimulator(
            num_simulations=num_simulations,
            random_seed=random_seed,
        )

        result = simulator.run(
            backtest_result,
            initial_capital=self.portfolio.initial_cash,
        )

        if verbose:
            from clyptq.analytics.risk.monte_carlo import print_monte_carlo_results

            print_monte_carlo_results(result)

        return result

    def _empty_metrics(self):
        """Create empty metrics when no trades occurred."""
        from clyptq.core.types import PerformanceMetrics

        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_leverage=0.0,
            max_leverage=0.0,
            avg_num_positions=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            duration_days=0,
        )

    def reset(self) -> None:
        """Reset engine state for re-running."""
        self.portfolio.reset()
        self.snapshots.clear()
        self.trades.clear()

    def __repr__(self) -> str:
        return (
            f"BacktestEngine("
            f"strategy={self.strategy.name}, "
            f"capital={self.portfolio.initial_cash})"
        )
