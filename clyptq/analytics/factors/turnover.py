from datetime import datetime
from typing import Dict, List, TYPE_CHECKING

import pandas as pd

from clyptq.core.types import BacktestResult

if TYPE_CHECKING:
    from clyptq.core.base import Strategy
    from clyptq.data.stores.store import DataStore
    from clyptq.trading.engine.backtest import BacktestEngine
    from clyptq.trading.execution.backtest import BacktestExecutor


def turnover_performance_frontier(
    results: List[BacktestResult], turnovers: List[float]
) -> pd.DataFrame:
    if len(results) != len(turnovers):
        raise ValueError("results and turnovers must have same length")

    data = []
    for result, turnover in zip(results, turnovers):
        sharpe = result.metrics.get("sharpe_ratio", 0.0)
        total_return = result.metrics.get("total_return", 0.0)
        max_dd = result.metrics.get("max_drawdown", 0.0)

        data.append({
            "turnover": turnover,
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "net_sharpe": sharpe,
        })

    df = pd.DataFrame(data)
    df = df.sort_values("turnover")

    return df


def optimal_rebalance_frequency(
    strategy: "Strategy",
    data: "DataStore",
    executor: "BacktestExecutor",
    initial_capital: float,
    start: datetime,
    end: datetime,
    frequencies: List[str] = None,
) -> Dict:
    from clyptq.trading.engine.backtest import BacktestEngine

    if frequencies is None:
        frequencies = ["daily", "weekly", "biweekly", "monthly"]

    results = {}
    for freq in frequencies:
        freq_strategy = type(strategy).__new__(type(strategy))
        freq_strategy.__dict__.update(strategy.__dict__)
        freq_strategy._schedule = freq

        engine = BacktestEngine(freq_strategy, data, executor, initial_capital)
        result = engine.run(start, end, verbose=False)

        results[freq] = {
            "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
            "total_return": result.metrics.get("total_return", 0.0),
            "max_drawdown": result.metrics.get("max_drawdown", 0.0),
            "num_trades": len(result.fills),
            "turnover": result.metrics.get("turnover", 0.0),
        }

    best_freq = max(results.items(), key=lambda x: x[1]["sharpe_ratio"])

    return {
        "optimal_frequency": best_freq[0],
        "optimal_metrics": best_freq[1],
        "all_results": results,
    }
