"""Backtest engine for historical simulation."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from clyptq.analytics.performance.metrics import compute_metrics
from clyptq.core.base import Executor, Factor, PortfolioConstructor, Strategy
from clyptq.core.types import BacktestResult, EngineMode, Fill, Order, OrderSide, Snapshot
from clyptq.data.stores.store import DataStore
from clyptq.trading.portfolio.state import PortfolioState
from clyptq.trading.risk.manager import RiskManager
from clyptq.infra.utils import get_logger


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        strategy: Strategy,
        data_store: DataStore,
        executor: Executor,
        initial_capital: float = 10000.0,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.strategy = strategy
        self.data_store = data_store
        self.executor = executor
        self.logger = get_logger(__name__, context={"strategy": strategy.__class__.__name__, "mode": "backtest"})
        self.portfolio = PortfolioState(initial_capital)
        self.risk_manager = risk_manager
        self.snapshots: List[Snapshot] = []
        self.trades: List[Fill] = []
        self._last_rebalance: Optional[datetime] = None
        self.factors: List[Factor] = strategy.factors()
        self.constructor: PortfolioConstructor = strategy.portfolio_constructor()
        self.constraints = strategy.constraints()

    def run(
        self, start: datetime, end: datetime, verbose: bool = False
    ) -> BacktestResult:
        """Run backtest from start to end."""
        timestamps = self._get_timestamps(start, end)

        self.logger.info("Backtest started", extra={"start": start.isoformat(), "end": end.isoformat(), "timestamps": len(timestamps)})

        warmup_days = self.strategy.warmup_periods()
        warmup = self._convert_warmup_to_periods(warmup_days)

        for i, timestamp in enumerate(timestamps):
            if i < warmup:
                continue

            try:
                self._process_timestamp(timestamp)

                if verbose and (i % 100 == 0 or i == len(timestamps) - 1):
                    pct = (i + 1) / len(timestamps) * 100
                    self.logger.info("Backtest progress", extra={"progress": f"{i+1}/{len(timestamps)}", "percent": pct})

            except KeyError as e:
                self.logger.error("Data access error during backtest", extra={"timestamp": timestamp.isoformat(), "error": str(e)})
                continue
            except ValueError as e:
                self.logger.error("Value error during backtest", extra={"timestamp": timestamp.isoformat(), "error": str(e)})
                continue
            except Exception as e:
                self.logger.error("Unexpected error during backtest", extra={"timestamp": timestamp.isoformat(), "error": str(e)})
                continue

        metrics = compute_metrics(self.snapshots, self.trades)
        self.logger.info("Backtest completed", extra={"total_trades": len(self.trades), "final_equity": self.snapshots[-1].equity if self.snapshots else 0})

        return BacktestResult(
            snapshots=self.snapshots,
            trades=self.trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
            mode=EngineMode.BACKTEST,
        )

    def run_monte_carlo(
        self,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """Run Monte Carlo simulation on current backtest results."""
        from clyptq.analytics.risk.monte_carlo import MonteCarloSimulator

        if not self.snapshots:
            raise ValueError("No backtest results available. Run run() first.")

        metrics = compute_metrics(self.snapshots, self.trades)
        backtest_result = BacktestResult(
            snapshots=self.snapshots,
            trades=self.trades,
            metrics=metrics,
            strategy_name=self.strategy.name,
            mode=EngineMode.BACKTEST,
        )

        if verbose:
            print(f"Running {num_simulations} Monte Carlo simulations...")

        simulator = MonteCarloSimulator(
            num_simulations=num_simulations,
            random_seed=random_seed,
        )

        result = simulator.run(backtest_result, initial_capital=self.portfolio.initial_cash)

        if verbose:
            from clyptq.analytics.risk.monte_carlo import print_monte_carlo_results

            print_monte_carlo_results(result)

        return result

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

    def _convert_warmup_to_periods(self, warmup_days: int) -> int:
        schedule = self.strategy.schedule()

        if schedule == "daily":
            return warmup_days
        elif schedule == "weekly":
            return max(1, warmup_days // 7)
        elif schedule == "monthly":
            return max(1, warmup_days // 30)
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

        self._check_and_liquidate_delisted(timestamp, available, prices)

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
                continue

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

    def reset(self) -> None:
        self.portfolio.reset()
        self.snapshots.clear()
        self.trades.clear()
        self._last_rebalance = None
