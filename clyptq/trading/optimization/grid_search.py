from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from clyptq.trading.execution.base import Executor
from clyptq.strategy.base import Strategy
from clyptq.core.types import BacktestResult
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine


@dataclass
class GridSearchResult:
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    param_scores: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "num_combinations_tested": len(self.all_results),
            "top_5_params": sorted(
                self.param_scores.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


class GridSearchOptimizer:
    def __init__(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Strategy],
        data_store: DataStore,
        executor: Executor,
        initial_capital: float = 100_000.0,
        scoring_metric: str = "sharpe_ratio",
    ):
        self.strategy_factory = strategy_factory
        self.data_store = data_store
        self.executor = executor
        self.initial_capital = initial_capital
        self.scoring_metric = scoring_metric

    def search(
        self,
        param_grid: Dict[str, List[Any]],
        start: datetime,
        end: datetime,
        cv_folds: Optional[int] = None,
    ) -> GridSearchResult:
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        all_combinations = list(product(*param_values))

        all_results = []
        param_scores = {}

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            if cv_folds and cv_folds > 1:
                score = self._cross_validate(params, start, end, cv_folds)
            else:
                score = self._evaluate_single(params, start, end)

            param_key = self._params_to_key(params)
            param_scores[param_key] = score

            all_results.append(
                {
                    "params": params,
                    "score": score,
                    "combination_index": i,
                }
            )

        best_result = max(all_results, key=lambda x: x["score"])

        return GridSearchResult(
            best_params=best_result["params"],
            best_score=best_result["score"],
            all_results=all_results,
            param_scores=param_scores,
        )

    def _evaluate_single(
        self, params: Dict[str, Any], start: datetime, end: datetime
    ) -> float:
        strategy = self.strategy_factory(params)

        engine = BacktestEngine(
            strategy=strategy,
            data_store=self.data_store,
            executor=self.executor,
            initial_capital=self.initial_capital,
        )

        result = engine.run(start=start, end=end)

        return self._extract_score(result)

    def _cross_validate(
        self, params: Dict[str, Any], start: datetime, end: datetime, n_folds: int
    ) -> float:
        import pandas as pd

        total_days = (end - start).days
        fold_size = total_days // n_folds

        scores = []
        for i in range(n_folds):
            fold_start = start + pd.Timedelta(days=i * fold_size)
            fold_end = start + pd.Timedelta(days=(i + 1) * fold_size)

            if fold_end > end:
                fold_end = end

            score = self._evaluate_single(params, fold_start, fold_end)
            scores.append(score)

        return np.mean(scores)

    def _extract_score(self, result: BacktestResult) -> float:
        if self.scoring_metric == "sharpe_ratio":
            return result.metrics.sharpe_ratio
        elif self.scoring_metric == "total_return":
            return result.metrics.total_return
        elif self.scoring_metric == "sortino_ratio":
            return result.metrics.sortino_ratio
        elif self.scoring_metric == "calmar_ratio":
            if abs(result.metrics.max_drawdown) > 1e-10:
                return result.metrics.total_return / abs(result.metrics.max_drawdown)
            return 0.0
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring_metric}")

    def _params_to_key(self, params: Dict[str, Any]) -> str:
        return str(sorted(params.items()))
