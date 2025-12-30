"""Live and paper trading engine."""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

from clyptq.core.base import Executor, Factor, PortfolioConstructor, Strategy
from clyptq.core.types import EngineMode, ExecutionResult, Fill, Order, OrderSide, Snapshot
from clyptq.data.live.buffer import RollingPriceBuffer
from clyptq.data.live.view import LiveDataView
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.data.stores.store import DataStore
from clyptq.data.streams.base import StreamingDataSource
from clyptq.execution.position_sync import PositionSynchronizer
from clyptq.portfolio.state import PortfolioState
from clyptq.risk.manager import RiskManager


class LiveEngine:
    """Live and paper trading engine."""

    def __init__(
        self,
        strategy: Strategy,
        data_store: DataStore,
        executor: Executor,
        initial_capital: float = 10000.0,
        risk_manager: Optional[RiskManager] = None,
        mode: EngineMode = EngineMode.PAPER,
    ):
        if mode not in [EngineMode.LIVE, EngineMode.PAPER]:
            raise ValueError(f"LiveEngine only supports LIVE or PAPER modes, got {mode}")

        self.strategy = strategy
        self.data_store = data_store
        self.executor = executor
        self.mode = mode
        self.portfolio = PortfolioState(initial_capital)
        self.risk_manager = risk_manager
        self.snapshots: List[Snapshot] = []
        self.trades: List[Fill] = []
        self._last_rebalance: Optional[datetime] = None
        self.factors: List[Factor] = strategy.factors()
        self.constructor: PortfolioConstructor = strategy.portfolio_constructor()
        self.constraints = strategy.constraints()

        max_lookback = max(
            [getattr(f, 'lookback', 100) for f in self.factors] +
            [strategy.warmup_periods(), 100]
        )
        self.price_buffer = RollingPriceBuffer(max_periods=max_lookback + 50)
        self.position_sync = PositionSynchronizer(tolerance=1e-6)

    def step(self, timestamp: datetime, prices: Dict[str, float]) -> ExecutionResult:
        """Execute one trading step."""
        if isinstance(self.data_store, LiveDataStore):
            self.data_store.update(timestamp, prices)

        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

        universe = self.strategy.universe()
        if universe:
            available = [s for s in universe if s in prices]
        else:
            available = list(prices.keys())

        if not available:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_symbols",
            )

        self._check_and_liquidate_delisted(timestamp, available, prices)

        if not self._should_rebalance(timestamp):
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="schedule",
            )

        data = self.data_store.get_view(timestamp)
        all_scores: Dict[str, float] = {}

        for factor in self.factors:
            try:
                scores = factor.compute(data)
                for symbol, score in scores.items():
                    if symbol in all_scores:
                        all_scores[symbol] = (all_scores[symbol] + score) / 2
                    else:
                        all_scores[symbol] = score
            except Exception:
                continue

        if not all_scores:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_scores",
            )

        target_weights = self.constructor.construct(all_scores, self.constraints)

        if not target_weights:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_weights",
            )

        current_weights = self.portfolio.get_weights(prices)
        orders = self._generate_orders(current_weights, target_weights, snapshot.equity, prices)

        if not orders:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="no_orders",
            )

        if self.risk_manager:
            orders = self.risk_manager.apply_position_limits(
                orders, self.portfolio.positions, prices, snapshot.equity
            )

        if not orders:
            return ExecutionResult(
                timestamp=timestamp,
                action="skip",
                fills=[],
                orders=[],
                snapshot=snapshot,
                rebalance_reason="risk_filtered",
            )

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                print(f"Fill rejected: {e}")
                continue

        final_snapshot = self.portfolio.get_snapshot(timestamp, prices)

        return ExecutionResult(
            timestamp=timestamp,
            action="rebalance",
            fills=fills,
            orders=orders,
            snapshot=final_snapshot,
            rebalance_reason="scheduled",
        )

    def run_live(self, interval_seconds: int = 60, verbose: bool = True) -> None:
        """Real-time trading loop."""
        if not hasattr(self.executor, "fetch_prices"):
            raise ValueError("Executor must have fetch_prices() for live trading")

        universe = self.strategy.universe()
        if not universe:
            raise ValueError("Strategy must define universe() for live trading")

        if verbose:
            print(f"Live trading started ({self.mode.value})")
            print(f"Universe: {universe}")
            print(f"Interval: {interval_seconds}s")

        iteration = 0
        position_sync_interval = max(10, 600 // interval_seconds)

        try:
            while True:
                timestamp = datetime.utcnow()

                try:
                    prices = self.executor.fetch_prices(universe)

                    if not prices:
                        if verbose:
                            print(f"No prices at {timestamp}. API sleeping?")
                        time.sleep(interval_seconds)
                        continue

                    self._process_live_timestamp(timestamp, prices)

                    if iteration % position_sync_interval == 0:
                        self._check_position_sync(verbose)

                    if hasattr(self.executor, 'cleanup_old_orders'):
                        if iteration % 100 == 0:
                            self.executor.cleanup_old_orders()

                    if verbose:
                        equity = self.snapshots[-1].equity if self.snapshots else 0
                        print(f"{timestamp.strftime('%H:%M:%S')} | ${equity:.2f}")

                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")

                iteration += 1
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            if verbose:
                print("\nStopped.")

    async def run_stream(
        self, stream: StreamingDataSource, verbose: bool = True
    ) -> None:
        """Real-time trading with streaming data."""
        universe = self.strategy.universe()
        if not universe:
            raise ValueError("Strategy must define universe() for live trading")

        if verbose:
            print(f"Streaming started ({self.mode.value})")
            print(f"Universe: {universe}")

        iteration = 0
        position_sync_interval = 100

        def on_tick(timestamp: datetime, prices: Dict[str, float]) -> None:
            nonlocal iteration

            try:
                self._process_live_timestamp(timestamp, prices)

                if iteration % position_sync_interval == 0:
                    self._check_position_sync(verbose)

                if hasattr(self.executor, 'cleanup_old_orders'):
                    if iteration % 100 == 0:
                        self.executor.cleanup_old_orders()

                if verbose and iteration % 10 == 0:
                    equity = self.snapshots[-1].equity if self.snapshots else 0
                    print(f"{timestamp.strftime('%H:%M:%S')} | ${equity:.2f}")

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")

            iteration += 1

        try:
            await stream.start(universe, on_tick)
            while stream.is_running():
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            if verbose:
                print("\nStopping stream...")
        finally:
            await stream.stop()
            if verbose:
                print("Stopped.")

    def _should_rebalance(self, timestamp: datetime) -> bool:
        schedule = self.strategy.schedule()

        if schedule == "daily":
            if self._last_rebalance is None or timestamp.date() != self._last_rebalance.date():
                self._last_rebalance = timestamp
                return True
            return False

        elif schedule == "weekly":
            if self._last_rebalance is None:
                self._last_rebalance = timestamp
                return True

            last_week = self._last_rebalance.isocalendar()[1]
            current_week = timestamp.isocalendar()[1]

            if current_week != last_week:
                self._last_rebalance = timestamp
                return True

            return False

        elif schedule == "monthly":
            if self._last_rebalance is None:
                self._last_rebalance = timestamp
                return True

            if timestamp.month != self._last_rebalance.month or timestamp.year != self._last_rebalance.year:
                self._last_rebalance = timestamp
                return True

            return False

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _process_live_timestamp(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        self.price_buffer.update(timestamp, prices)

        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

        universe = self.strategy.universe()
        if universe:
            available = [s for s in universe if s in prices]
        else:
            available = list(prices.keys())

        self._check_and_liquidate_delisted(timestamp, available, prices)

        if self.risk_manager:
            if self.risk_manager.check_max_drawdown(snapshot.equity):
                print("MAX DD HIT. dumping everything")
                liquidate_orders = [
                    Order(symbol=symbol, side=OrderSide.SELL, amount=pos.amount)
                    for symbol, pos in self.portfolio.positions.items()
                ]
                fills = self.executor.execute(liquidate_orders, timestamp, prices)
                for fill in fills:
                    try:
                        self.portfolio.apply_fill(fill)
                        self.trades.append(fill)
                    except ValueError as e:
                        print(f"Fill rejected: {e}")
                return

            exit_orders = self.risk_manager.check_position_exits(
                self.portfolio.positions, prices
            )
            if exit_orders:
                fills = self.executor.execute(exit_orders, timestamp, prices)
                for fill in fills:
                    try:
                        self.portfolio.apply_fill(fill)
                        self.trades.append(fill)
                    except ValueError as e:
                        print(f"Fill rejected: {e}")

        if not self._should_rebalance(timestamp):
            return

        warmup = self.strategy.warmup_periods()
        if len(self.price_buffer.timestamps) < warmup:
            return

        data = LiveDataView(self.price_buffer, timestamp)
        all_scores: Dict[str, float] = {}

        for factor in self.factors:
            try:
                scores = factor.compute(data)
                for symbol, score in scores.items():
                    if symbol in all_scores:
                        all_scores[symbol] = (all_scores[symbol] + score) / 2
                    else:
                        all_scores[symbol] = score
            except Exception:
                continue

        if not all_scores:
            return

        target_weights = self.constructor.construct(all_scores, self.constraints)

        if not target_weights:
            return

        current_weights = self.portfolio.get_weights(prices)
        orders = self._generate_orders(current_weights, target_weights, snapshot.equity, prices)

        if not orders:
            return

        if self.risk_manager:
            orders = self.risk_manager.apply_position_limits(
                orders, self.portfolio.positions, prices, snapshot.equity
            )

        if not orders:
            return

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                print(f"Fill rejected: {e}")

    def _generate_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        equity: float,
        prices: Dict[str, float],
    ) -> List[Order]:
        orders = []
        all_symbols = sorted(set(current_weights.keys()) | set(target_weights.keys()))
        sells = []
        buys = []

        fee_reserve_factor = 1.0 - 0.002

        for symbol in all_symbols:
            if symbol not in prices:
                continue

            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 1e-6:
                continue

            if weight_diff > 0:
                target_value = target_weight * equity * fee_reserve_factor
            else:
                target_value = target_weight * equity

            target_amount = target_value / prices[symbol] if prices[symbol] > 0 else 0.0
            current_value = current_weight * equity
            current_amount = current_value / prices[symbol] if prices[symbol] > 0 else 0.0
            amount_diff = target_amount - current_amount

            if abs(amount_diff) < 1e-8:
                continue

            if amount_diff > 0:
                order = Order(symbol=symbol, side=OrderSide.BUY, amount=amount_diff)
                buys.append(order)
            else:
                order = Order(symbol=symbol, side=OrderSide.SELL, amount=abs(amount_diff))
                sells.append(order)

        orders = sells + buys
        return orders

    def _check_and_liquidate_delisted(
        self, timestamp: datetime, available: List[str], prices: Dict[str, float]
    ) -> None:
        if not self.portfolio.positions:
            return

        delisted = [
            symbol for symbol in self.portfolio.positions.keys()
            if symbol not in available
        ]

        if not delisted:
            return

        orders = [
            Order(symbol=symbol, side=OrderSide.SELL, amount=pos.amount)
            for symbol, pos in self.portfolio.positions.items()
            if symbol in delisted
        ]

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                print(f"Delisted liquidation failed: {e}")

    def _check_position_sync(self, verbose: bool = False) -> None:
        if not hasattr(self.executor, 'fetch_positions'):
            return

        try:
            exchange_positions = self.executor.fetch_positions()
            discrepancies = self.position_sync.check_discrepancies(
                self.portfolio.positions, exchange_positions
            )

            critical = [d for d in discrepancies if d.is_critical]
            if critical and verbose:
                print(f"WARNING: {len(critical)} position discrepancies detected")
                for disc in critical:
                    print(
                        f"  {disc.symbol}: internal={disc.internal_amount:.4f} "
                        f"exchange={disc.exchange_amount:.4f} diff={disc.amount_diff:.4f}"
                    )

        except Exception as e:
            if verbose:
                print(f"Position sync check failed: {e}")

    def reset(self) -> None:
        self.portfolio.reset()
        self.snapshots.clear()
        self.trades.clear()
        self._last_rebalance = None
        self.price_buffer.clear()
