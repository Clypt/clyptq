from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from clyptq.trading.execution.base import Executor
from clyptq.strategy.base import Strategy
from clyptq.core.types import BacktestResult
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine


@dataclass
class OutOfSampleResult:
    train_result: BacktestResult
    test_result: BacktestResult
    degradation_ratio: float
    sharpe_degradation: float
    return_degradation: float
    is_overfitted: bool
    stability_score: float

    def to_dict(self) -> dict:
        return {
            "train_sharpe": self.train_result.metrics.sharpe_ratio,
            "test_sharpe": self.test_result.metrics.sharpe_ratio,
            "train_return": self.train_result.metrics.total_return,
            "test_return": self.test_result.metrics.total_return,
            "degradation_ratio": self.degradation_ratio,
            "sharpe_degradation": self.sharpe_degradation,
            "return_degradation": self.return_degradation,
            "is_overfitted": self.is_overfitted,
            "stability_score": self.stability_score,
        }


@dataclass
class WalkForwardResult:
    periods: List[Tuple[BacktestResult, BacktestResult]]
    mean_train_sharpe: float
    mean_test_sharpe: float
    mean_degradation: float
    consistency_score: float
    overfitting_ratio: float

    def to_dict(self) -> dict:
        return {
            "num_periods": len(self.periods),
            "mean_train_sharpe": self.mean_train_sharpe,
            "mean_test_sharpe": self.mean_test_sharpe,
            "mean_degradation": self.mean_degradation,
            "consistency_score": self.consistency_score,
            "overfitting_ratio": self.overfitting_ratio,
        }


@dataclass
class ParameterStabilityResult:
    param_results: Dict[str, OutOfSampleResult]
    best_params: Dict[str, Any]
    stability_scores: Dict[str, float]
    robust_params: List[str]

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "num_configs_tested": len(self.param_results),
            "robust_configs": len(self.robust_params),
            "stability_scores": self.stability_scores,
        }


class HistoricalSimulator:
    def __init__(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Strategy],
        data_store: DataStore,
        executor: Executor,
        initial_capital: float = 100_000.0,
        overfitting_threshold: float = 0.3,
    ):
        self.strategy_factory = strategy_factory
        self.data_store = data_store
        self.executor = executor
        self.initial_capital = initial_capital
        self.overfitting_threshold = overfitting_threshold

    def run_out_of_sample(
        self,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> OutOfSampleResult:
        if params is None:
            params = {}

        strategy = self.strategy_factory(params)

        engine = BacktestEngine(
            strategy=strategy,
            data_store=self.data_store,
            executor=self.executor,
            initial_capital=self.initial_capital,
        )

        train_result = engine.run(start=train_start, end=train_end)

        strategy_test = self.strategy_factory(params)
        engine_test = BacktestEngine(
            strategy=strategy_test,
            data_store=self.data_store,
            executor=self.executor,
            initial_capital=self.initial_capital,
        )

        test_result = engine_test.run(start=test_start, end=test_end)

        degradation_ratio = self._calculate_degradation(
            train_result.metrics.sharpe_ratio, test_result.metrics.sharpe_ratio
        )

        sharpe_degradation = (
            train_result.metrics.sharpe_ratio - test_result.metrics.sharpe_ratio
        ) / max(abs(train_result.metrics.sharpe_ratio), 1e-10)

        return_degradation = (
            train_result.metrics.total_return - test_result.metrics.total_return
        ) / max(abs(train_result.metrics.total_return), 1e-10)

        is_overfitted = degradation_ratio > self.overfitting_threshold

        stability_score = self._calculate_stability(train_result, test_result)

        return OutOfSampleResult(
            train_result=train_result,
            test_result=test_result,
            degradation_ratio=degradation_ratio,
            sharpe_degradation=sharpe_degradation,
            return_degradation=return_degradation,
            is_overfitted=is_overfitted,
            stability_score=stability_score,
        )

    def run_walk_forward(
        self,
        total_start: datetime,
        total_end: datetime,
        train_window_days: int,
        test_window_days: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> WalkForwardResult:
        if params is None:
            params = {}

        periods = []
        current_date = total_start

        while True:
            train_start = current_date
            train_end = train_start + pd.Timedelta(days=train_window_days)

            if train_end >= total_end:
                break

            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=test_window_days)

            if test_end > total_end:
                break

            strategy_train = self.strategy_factory(params)
            engine_train = BacktestEngine(
                strategy=strategy_train,
                data_store=self.data_store,
                executor=self.executor,
                initial_capital=self.initial_capital,
            )
            train_result = engine_train.run(start=train_start, end=train_end)

            strategy_test = self.strategy_factory(params)
            engine_test = BacktestEngine(
                strategy=strategy_test,
                data_store=self.data_store,
                executor=self.executor,
                initial_capital=self.initial_capital,
            )
            test_result = engine_test.run(start=test_start, end=test_end)

            periods.append((train_result, test_result))

            current_date = test_end + pd.Timedelta(days=1)

        if not periods:
            raise ValueError("No valid walk-forward periods found")

        train_sharpes = [p[0].metrics.sharpe_ratio for p in periods]
        test_sharpes = [p[1].metrics.sharpe_ratio for p in periods]

        mean_train_sharpe = np.mean(train_sharpes)
        mean_test_sharpe = np.mean(test_sharpes)

        degradations = [
            self._calculate_degradation(train_sharpes[i], test_sharpes[i])
            for i in range(len(periods))
        ]
        mean_degradation = np.mean(degradations)

        consistency_score = self._calculate_consistency(test_sharpes)

        overfitting_ratio = sum(1 for d in degradations if d > self.overfitting_threshold) / len(
            degradations
        )

        return WalkForwardResult(
            periods=periods,
            mean_train_sharpe=mean_train_sharpe,
            mean_test_sharpe=mean_test_sharpe,
            mean_degradation=mean_degradation,
            consistency_score=consistency_score,
            overfitting_ratio=overfitting_ratio,
        )

    def analyze_parameter_stability(
        self,
        param_grid: List[Dict[str, Any]],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> ParameterStabilityResult:
        param_results = {}
        stability_scores = {}

        for i, params in enumerate(param_grid):
            param_key = f"config_{i}"
            result = self.run_out_of_sample(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                params=params,
            )
            param_results[param_key] = result
            stability_scores[param_key] = result.stability_score

        best_config_key = max(stability_scores.items(), key=lambda x: x[1])[0]
        best_params = param_grid[int(best_config_key.split("_")[1])]

        mean_stability = np.mean(list(stability_scores.values()))
        std_stability = np.std(list(stability_scores.values()))
        threshold = mean_stability - 0.5 * std_stability

        robust_params = [
            key for key, score in stability_scores.items() if score >= threshold
        ]

        return ParameterStabilityResult(
            param_results=param_results,
            best_params=best_params,
            stability_scores=stability_scores,
            robust_params=robust_params,
        )

    def _calculate_degradation(self, train_metric: float, test_metric: float) -> float:
        if abs(train_metric) < 1e-10:
            return 0.0
        return (train_metric - test_metric) / abs(train_metric)

    def _calculate_stability(
        self, train_result: BacktestResult, test_result: BacktestResult
    ) -> float:
        sharpe_stability = 1.0 - abs(
            self._calculate_degradation(train_result.metrics.sharpe_ratio, test_result.metrics.sharpe_ratio)
        )

        return_stability = 1.0 - abs(
            self._calculate_degradation(train_result.metrics.total_return, test_result.metrics.total_return)
        )

        drawdown_stability = 1.0 - abs(
            (train_result.metrics.max_drawdown - test_result.metrics.max_drawdown)
            / max(abs(train_result.metrics.max_drawdown), 1e-10)
        )

        stability = (sharpe_stability + return_stability + drawdown_stability) / 3.0

        return max(0.0, min(1.0, stability))

    def _calculate_consistency(self, metrics: List[float]) -> float:
        if len(metrics) < 2:
            return 1.0

        mean_metric = np.mean(metrics)
        if abs(mean_metric) < 1e-10:
            return 0.0

        std_metric = np.std(metrics)
        cv = std_metric / abs(mean_metric)

        consistency = 1.0 / (1.0 + cv)

        return consistency
