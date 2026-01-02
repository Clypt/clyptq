"""
Walk-forward optimization for robust parameter tuning.

Splits data into rolling train/test windows, optimizes on train,
validates on test, then aggregates results across all windows.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass

import pandas as pd

from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import Engine
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.core.base import Strategy
from clyptq.core.types import BacktestResult, CostModel, EngineMode


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    train_metric: float
    test_metric: float
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization result."""

    windows: List[WalkForwardWindow]
    avg_train_metric: float
    avg_test_metric: float
    best_params_frequency: Dict[str, int]
    combined_test_result: Optional[BacktestResult] = None


class WalkForwardOptimizer:
    """
    Walk-forward optimizer with rolling train/test windows.

    Prevents overfitting by testing parameters on out-of-sample data
    before moving to the next window.
    """

    def __init__(
        self,
        strategy_factory: Callable[..., Strategy],
        param_grid: Dict[str, List[Any]],
        train_days: int = 180,
        test_days: int = 30,
        metric: str = "sharpe_ratio",
        initial_capital: float = 10000.0,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_factory: Function that creates Strategy instance from params
            param_grid: Dict of {param_name: [values]}
            train_days: Training window size in days
            test_days: Test window size in days
            metric: Metric to optimize ("sharpe_ratio", "total_return", "max_drawdown")
            initial_capital: Initial capital for backtests
        """
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.train_days = train_days
        self.test_days = test_days
        self.metric = metric
        self.initial_capital = initial_capital

    def optimize(
        self,
        data_store: DataStore,
        start: datetime,
        end: datetime,
        verbose: bool = False,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data_store: DataStore with OHLCV data
            start: Overall start date
            end: Overall end date
            verbose: Print progress

        Returns:
            WalkForwardResult with all windows and aggregated metrics
        """
        windows = self._generate_windows(start, end)

        if verbose:
            print(f"Walk-forward optimization: {len(windows)} windows")
            print(f"Train: {self.train_days}d, Test: {self.test_days}d")
            print(f"Metric: {self.metric}")

        results: List[WalkForwardWindow] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if verbose:
                print(f"\nWindow {i+1}/{len(windows)}")
                print(f"  Train: {train_start.date()} -> {train_end.date()}")
                print(f"  Test: {test_start.date()} -> {test_end.date()}")

            best_params, train_metric = self._optimize_window(
                data_store, train_start, train_end, verbose
            )

            if verbose:
                print(f"  Best params: {best_params}")
                print(f"  Train {self.metric}: {train_metric:.4f}")

            test_result, test_metric = self._test_window(
                data_store, test_start, test_end, best_params
            )

            if verbose:
                print(f"  Test {self.metric}: {test_metric:.4f}")

            window_result = WalkForwardWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metric=train_metric,
                test_metric=test_metric,
                test_result=test_result,
            )
            results.append(window_result)

        return self._aggregate_results(results)

    def _generate_windows(
        self, start: datetime, end: datetime
    ) -> List[tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test window pairs."""
        windows = []
        current = start

        while True:
            train_start = current
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current = test_start

        return windows

    def _optimize_window(
        self, data_store: DataStore, start: datetime, end: datetime, verbose: bool
    ) -> tuple[Dict[str, Any], float]:
        """Grid search on training window."""
        param_combinations = self._generate_param_combinations()

        best_params = None
        best_metric = float("-inf") if self._is_higher_better() else float("inf")

        for params in param_combinations:
            try:
                strategy = self.strategy_factory(**params)
                cost_model = CostModel(
                    maker_fee=0.001,
                    taker_fee=0.001,
                    slippage_bps=5.0,
                )
                executor = BacktestExecutor(cost_model=cost_model)
                engine = Engine(
                    strategy=strategy,
                    data_store=data_store,
                    executor=executor,
                    initial_capital=self.initial_capital,
                )

                result = engine.run(start, end, verbose=False)
                metric_value = self._extract_metric(result)

                is_better = (
                    metric_value > best_metric
                    if self._is_higher_better()
                    else metric_value < best_metric
                )

                if is_better:
                    best_metric = metric_value
                    best_params = params

            except Exception as e:
                import traceback

                if verbose:
                    print(f"    Params {params} failed: {e}")
                    traceback.print_exc()
                continue

        if best_params is None:
            raise ValueError(f"No valid parameter combination found. Tried {len(param_combinations)} combinations.")

        return best_params, best_metric

    def _test_window(
        self, data_store: DataStore, start: datetime, end: datetime, params: Dict[str, Any]
    ) -> tuple[BacktestResult, float]:
        """Test parameters on out-of-sample window."""
        strategy = self.strategy_factory(**params)
        cost_model = CostModel(
            maker_fee=0.001,
            taker_fee=0.001,
            slippage_bps=5.0,
        )
        executor = BacktestExecutor(cost_model=cost_model)
        engine = Engine(
            strategy=strategy,
            data_store=data_store,
            executor=executor,
            initial_capital=self.initial_capital,
        )

        result = engine.run(start, end, verbose=False)
        metric_value = self._extract_metric(result)

        return result, metric_value

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def _extract_metric(self, result: BacktestResult) -> float:
        """Extract optimization metric from backtest result."""
        if self.metric == "sharpe_ratio":
            return result.metrics.sharpe_ratio
        elif self.metric == "total_return":
            return result.metrics.total_return
        elif self.metric == "max_drawdown":
            return result.metrics.max_drawdown
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _is_higher_better(self) -> bool:
        """Check if higher metric value is better."""
        return self.metric in ["sharpe_ratio", "total_return"]

    def _aggregate_results(self, windows: List[WalkForwardWindow]) -> WalkForwardResult:
        """Aggregate results across all windows."""
        avg_train = sum(w.train_metric for w in windows) / len(windows)
        avg_test = sum(w.test_metric for w in windows) / len(windows)

        param_freq: Dict[str, int] = {}
        for window in windows:
            key = str(sorted(window.best_params.items()))
            param_freq[key] = param_freq.get(key, 0) + 1

        combined_snapshots = []
        combined_trades = []
        for window in windows:
            combined_snapshots.extend(window.test_result.snapshots)
            combined_trades.extend(window.test_result.trades)

        from clyptq.analytics.performance.metrics import compute_metrics

        combined_metrics = compute_metrics(combined_snapshots, combined_trades)
        combined_result = BacktestResult(
            snapshots=combined_snapshots,
            trades=combined_trades,
            metrics=combined_metrics,
            strategy_name="WalkForwardOptimized",
            mode=EngineMode.BACKTEST,
        )

        return WalkForwardResult(
            windows=windows,
            avg_train_metric=avg_train,
            avg_test_metric=avg_test,
            best_params_frequency=param_freq,
            combined_test_result=combined_result,
        )
