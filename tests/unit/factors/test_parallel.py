import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.core.base import Factor
from clyptq.trading.factors.parallel import ParallelFactorComputer


class FastFactor(Factor):
    def __init__(self, factor_id: int):
        self.factor_id = factor_id

    def compute(self, current_prices, history, timestamp):
        if len(history) < 5:
            return {}
        returns = history.iloc[-5:].pct_change().mean()
        return {
            symbol: float(value) * self.factor_id
            for symbol, value in returns.items()
        }


class SlowFactor(Factor):
    def __init__(self, factor_id: int, delay: float = 0.1):
        self.factor_id = factor_id
        self.delay = delay

    def compute(self, current_prices, history, timestamp):
        time.sleep(self.delay)
        if len(history) < 5:
            return {}
        returns = history.iloc[-5:].pct_change().mean()
        return {
            symbol: float(value) * self.factor_id
            for symbol, value in returns.items()
        }


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D", tz=timezone.utc)
    np.random.seed(42)
    data = {
        "BTC/USDT": 40000 + np.cumsum(np.random.randn(30) * 1000),
        "ETH/USDT": 2000 + np.cumsum(np.random.randn(30) * 50),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def factors():
    return [FastFactor(i) for i in range(1, 6)]


def test_parallel_thread_executor(sample_data, factors):
    computer = ParallelFactorComputer(max_workers=3, executor_type="thread")

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    results = computer.compute_factors(factors, current_prices, history, timestamp)

    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)
    assert all("BTC/USDT" in r and "ETH/USDT" in r for r in results)


def test_parallel_process_executor(sample_data, factors):
    computer = ParallelFactorComputer(max_workers=2, executor_type="process")

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    results = computer.compute_factors(factors, current_prices, history, timestamp)

    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)


def test_sequential_computation(sample_data, factors):
    computer = ParallelFactorComputer()

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    results = computer.compute_factors_sequential(
        factors, current_prices, history, timestamp
    )

    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)


def test_parallel_vs_sequential_consistency(sample_data, factors):
    computer = ParallelFactorComputer(max_workers=3, executor_type="thread")

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    parallel_results = computer.compute_factors(
        factors, current_prices, history, timestamp
    )
    sequential_results = computer.compute_factors_sequential(
        factors, current_prices, history, timestamp
    )

    assert len(parallel_results) == len(sequential_results)

    for p_res, s_res in zip(parallel_results, sequential_results):
        assert set(p_res.keys()) == set(s_res.keys())
        for symbol in p_res.keys():
            assert abs(p_res[symbol] - s_res[symbol]) < 1e-10


def test_single_factor_optimization(sample_data):
    computer = ParallelFactorComputer(max_workers=3, executor_type="thread")
    factor = FastFactor(1)

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    results = computer.compute_factors([factor], current_prices, history, timestamp)

    assert len(results) == 1
    assert isinstance(results[0], dict)


def test_parallel_speedup(sample_data):
    slow_factors = [SlowFactor(i, delay=0.05) for i in range(1, 5)]

    sequential_computer = ParallelFactorComputer()
    parallel_computer = ParallelFactorComputer(max_workers=4, executor_type="thread")

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    start = time.time()
    sequential_computer.compute_factors_sequential(
        slow_factors, current_prices, history, timestamp
    )
    sequential_time = time.time() - start

    start = time.time()
    parallel_computer.compute_factors(
        slow_factors, current_prices, history, timestamp
    )
    parallel_time = time.time() - start

    assert parallel_time < sequential_time * 0.7


def test_empty_history(factors):
    computer = ParallelFactorComputer(max_workers=3, executor_type="thread")

    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    current_prices = pd.Series({"BTC/USDT": 40000.0, "ETH/USDT": 2000.0})
    history = pd.DataFrame()

    results = computer.compute_factors(factors, current_prices, history, timestamp)

    assert len(results) == 5
    assert all(r == {} for r in results)


def test_auto_max_workers(sample_data, factors):
    computer = ParallelFactorComputer(max_workers=None, executor_type="thread")

    timestamp = sample_data.index[10]
    current_prices = sample_data.iloc[10]
    history = sample_data.iloc[:10]

    results = computer.compute_factors(factors, current_prices, history, timestamp)

    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)
