import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clypt.analytics.metrics import compute_metrics
from clypt.data.live_view import LiveDataView
from clypt.data.rolling_buffer import RollingPriceBuffer
from clypt.data.store import DataStore
from clypt.engine.cost_model import CostModel
from clypt.engine.executors import Executor
from clypt.engine.portfolio_state import PortfolioState
from clypt.engine.risk_manager import RiskManager
from clypt.factors.base import Factor
from clypt.portfolio.construction import PortfolioConstructor
from clypt.strategy.base import Strategy
from clypt.types import (
    BacktestResult,
    EngineMode,
    Fill,
    Order,
    OrderSide,
    Snapshot,
)


class Engine:

    def __init__(
        self,
        strategy: Strategy,
        data_store: DataStore,
        mode: EngineMode,
        executor: Executor,
        initial_capital: float = 10000.0,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.strategy = strategy
        self.data_store = data_store
        self.mode = mode
        self.executor = executor
        self.portfolio = PortfolioState(initial_capital)
        self.risk_manager = risk_manager
        self.snapshots: List[Snapshot] = []
        self.trades: List[Fill] = []
        self._last_rebalance: Optional[datetime] = None
        self.factors: List[Factor] = strategy.factors()
        self.constructor: PortfolioConstructor = strategy.portfolio_constructor()
        self.constraints = strategy.constraints()

        self.price_buffer: Optional[RollingPriceBuffer] = None
        if mode in [EngineMode.LIVE, EngineMode.PAPER]:
            max_lookback = max(
                [getattr(f, 'lookback', 100) for f in self.factors] +
                [strategy.warmup_periods(), 100]
            )
            self.price_buffer = RollingPriceBuffer(max_periods=max_lookback + 50)

    def run_backtest(
        self, start: datetime, end: datetime, verbose: bool = False
    ) -> BacktestResult:
        if self.mode not in [EngineMode.BACKTEST, EngineMode.PAPER]:
            raise ValueError("run_backtest only works in BACKTEST or PAPER modes")

        timestamps = self._get_timestamps(start, end)

        if verbose:
            print(f"Running backtest: {start} to {end}")
            print(f"Total timestamps: {len(timestamps)}")

        warmup = self.strategy.warmup_periods()

        for i, timestamp in enumerate(timestamps):
            if i < warmup:
                continue

            try:
                self._process_timestamp(timestamp)

                if verbose and (i % 100 == 0 or i == len(timestamps) - 1):
                    pct = (i + 1) / len(timestamps) * 100
                    print(f"Progress: {i+1}/{len(timestamps)} ({pct:.1f}%)")

            except Exception as e:
                if verbose:
                    print(f"Error at {timestamp}: {e}")
                continue

        metrics = compute_metrics(self.snapshots, self.trades)

        return BacktestResult(
            snapshots=self.snapshots,
            trades=self.trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
            mode=self.mode,
        )

    def _get_timestamps(self, start: datetime, end: datetime) -> List[datetime]:
        schedule = self.strategy.schedule()

        if schedule == "daily":
            return pd.date_range(start, end, freq="D").to_pydatetime().tolist()
        elif schedule == "weekly":
            return pd.date_range(start, end, freq="W-MON").to_pydatetime().tolist()
        elif schedule == "monthly":
            return pd.date_range(start, end, freq="MS").to_pydatetime().tolist()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

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

    def _process_timestamp(self, timestamp: datetime) -> None:
        data = self.data_store.get_view(timestamp)

        universe = self.strategy.universe()
        if universe:
            available = [s for s in universe if s in data.symbols]
        else:
            available = self.data_store.available_symbols(timestamp)

        if not available:
            return

        prices = data.current_prices()
        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

        if not self._should_rebalance(timestamp):
            return

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

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                print(f"Fill rejected: {e}")
                continue

    def _generate_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        equity: float,
        prices: Dict[str, float],
    ) -> List[Order]:
        orders = []
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
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

    def run_live(self, interval_seconds: int = 60, verbose: bool = True) -> None:
        """Real-time loop. Don't fuck this up."""
        if self.mode not in [EngineMode.LIVE, EngineMode.PAPER]:
            raise ValueError("run_live only works in LIVE or PAPER modes")

        if not hasattr(self.executor, "fetch_prices"):
            raise ValueError("Executor must have fetch_prices() for live trading")

        universe = self.strategy.universe()
        if not universe:
            raise ValueError("Strategy must define universe() for live trading")

        if verbose:
            print(f"Live trading started ({self.mode.value})")
            print(f"Universe: {universe}")
            print(f"Interval: {interval_seconds}s")

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

                    if verbose:
                        equity = self.snapshots[-1].equity if self.snapshots else 0
                        print(f"{timestamp.strftime('%H:%M:%S')} | ${equity:.2f}")

                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            if verbose:
                print("\nStopped.")

    def _process_live_timestamp(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        if self.price_buffer is not None:
            self.price_buffer.update(timestamp, prices)

        snapshot = self.portfolio.get_snapshot(timestamp, prices)
        self.snapshots.append(snapshot)

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
        if self.price_buffer is None or len(self.price_buffer.timestamps) < warmup:
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

        fills = self.executor.execute(orders, timestamp, prices)

        for fill in fills:
            try:
                self.portfolio.apply_fill(fill)
                self.trades.append(fill)
            except ValueError as e:
                print(f"Fill rejected: {e}")

    def reset(self) -> None:
        self.portfolio.reset()
        self.snapshots.clear()
        self.trades.clear()
        self._last_rebalance = None
        if self.price_buffer is not None:
            self.price_buffer.clear()
