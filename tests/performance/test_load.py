import gc
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.stores.store import DataStore, DataView
from clyptq.trading.engine.backtest import BacktestEngine
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import Strategy


class SimpleMomentum(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, data: DataView) -> dict[str, float]:
        scores = {}
        for symbol in data.symbols:
            df = data.ohlcv(symbol, self.lookback)
            if len(df) < self.lookback:
                continue
            returns = df["close"].pct_change(self.lookback).iloc[-1]
            scores[symbol] = float(returns)
        return scores


class LoadTestStrategy(Strategy):
    def __init__(self, num_factors: int = 5):
        self._factors = [SimpleMomentum(lookback=10 + i * 5) for i in range(num_factors)]
        self._constructor = TopNConstructor(top_n=10)
        self._constraints = Constraints()

    def factors(self) -> List[Factor]:
        return self._factors

    def portfolio_constructor(self):
        return self._constructor

    def constraints(self) -> Constraints:
        return self._constraints

    def schedule(self) -> str:
        return "daily"

    def warmup_periods(self) -> int:
        return 50


def generate_large_universe(num_symbols: int, num_bars: int) -> DataStore:
    store = DataStore()
    start = datetime(2024, 1, 1)

    for i in range(num_symbols):
        symbol = f"SYM{i:03d}/USDT"
        dates = [start + timedelta(days=j) for j in range(num_bars)]

        base_price = 100.0 + np.random.uniform(-50, 50)
        returns = np.random.normal(0.001, 0.02, num_bars)
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, num_bars)),
                "high": prices * (1 + np.random.uniform(0, 0.02, num_bars)),
                "low": prices * (1 - np.random.uniform(0, 0.02, num_bars)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, num_bars),
            },
            index=dates,
        )

        store.add_ohlcv(symbol, df)

    return store


@pytest.mark.performance
def test_100_symbol_universe():
    num_symbols = 100
    num_bars = 365
    num_factors = 5

    print(f"\n=== 100 Symbol Universe Stress Test ===")
    print(f"Symbols: {num_symbols}, Bars: {num_bars}, Factors: {num_factors}")

    start_time = time.time()
    store = generate_large_universe(num_symbols, num_bars)
    data_gen_time = time.time() - start_time
    print(f"Data generation: {data_gen_time:.2f}s")

    strategy = LoadTestStrategy(num_factors=num_factors)
    cost_model = CostModel(taker_fee_bps=10, maker_fee_bps=10, slippage_bps=5)
    executor = BacktestExecutor(cost_model)
    engine = BacktestEngine(strategy, store, executor, initial_capital=100000)

    backtest_start = datetime(2024, 2, 1)
    backtest_end = datetime(2024, 12, 31)

    start_time = time.time()
    result = engine.run(backtest_start, backtest_end, verbose=False)
    backtest_time = time.time() - start_time

    print(f"Backtest time: {backtest_time:.2f}s")
    print(f"Trades: {len(result.trades)}")
    print(f"Final equity: ${result.equity:.2f}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.3f}")

    assert backtest_time < 60, f"Backtest too slow: {backtest_time:.2f}s"
    assert result.equity > 0


@pytest.mark.performance
def test_1000_factor_computations():
    num_symbols = 50
    num_bars = 200
    num_computations = 1000

    print(f"\n=== 1000+ Factor Computation Benchmark ===")
    print(f"Symbols: {num_symbols}, Computations: {num_computations}")

    store = generate_large_universe(num_symbols, num_bars)
    factor = MomentumFactor(lookback=20)

    date_range = store.get_date_range()
    timestamps = pd.date_range(start=date_range.start, end=date_range.end, freq='D')
    test_timestamps = timestamps[-100:].tolist()

    start_time = time.time()
    total_scores = 0

    for i in range(num_computations):
        timestamp = test_timestamps[i % len(test_timestamps)]
        view = store.get_view(timestamp)
        scores = factor.compute(view)
        total_scores += len(scores)

    computation_time = time.time() - start_time
    avg_time = (computation_time / num_computations) * 1000

    print(f"Total time: {computation_time:.2f}s")
    print(f"Average per computation: {avg_time:.2f}ms")
    print(f"Total scores computed: {total_scores}")

    assert avg_time < 100, f"Factor computation too slow: {avg_time:.2f}ms"
    assert total_scores > 0


@pytest.mark.performance
def test_memory_leak_detection():
    num_symbols = 30
    num_bars = 200
    num_iterations = 100

    print(f"\n=== Memory Leak Detection (Long-Running) ===")
    print(f"Iterations: {num_iterations}")

    store = generate_large_universe(num_symbols, num_bars)
    strategy = LoadTestStrategy(num_factors=3)
    cost_model = CostModel(taker_fee_bps=10, maker_fee_bps=10, slippage_bps=5)
    executor = BacktestExecutor(cost_model)

    tracemalloc.start()
    gc.collect()
    snapshot_start = tracemalloc.take_snapshot()

    backtest_start = datetime(2024, 2, 1)
    backtest_end = datetime(2024, 6, 1)

    for i in range(num_iterations):
        engine = BacktestEngine(strategy, store, executor, initial_capital=100000)
        result = engine.run(backtest_start, backtest_end, verbose=False)

        if i % 10 == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i}: Current={current / 1024 / 1024:.1f}MB, Peak={peak / 1024 / 1024:.1f}MB")

    gc.collect()
    snapshot_end = tracemalloc.take_snapshot()

    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
    total_growth = sum(stat.size_diff for stat in top_stats)
    growth_mb = total_growth / 1024 / 1024

    print(f"\n=== Memory Growth Analysis ===")
    print(f"Total memory growth: {growth_mb:.2f}MB")

    print("\nTop 5 memory allocations:")
    for stat in top_stats[:5]:
        print(f"{stat}")

    tracemalloc.stop()

    threshold_mb = 50
    assert growth_mb < threshold_mb, f"Memory leak detected: {growth_mb:.2f}MB growth (threshold: {threshold_mb}MB)"


def run_single_backtest(args):
    store, strategy_params, backtest_params = args

    strategy = LoadTestStrategy(**strategy_params)
    cost_model = CostModel(taker_fee_bps=10, maker_fee_bps=10, slippage_bps=5)
    executor = BacktestExecutor(cost_model)
    engine = BacktestEngine(strategy, store, executor, initial_capital=100000)

    result = engine.run(**backtest_params, verbose=False)
    return result.equity, result.sharpe_ratio, len(result.trades)


@pytest.mark.performance
def test_concurrent_engine_instances():
    num_engines = 10
    num_symbols = 30
    num_bars = 200

    print(f"\n=== Concurrent {num_engines} Engine Instances ===")

    store = generate_large_universe(num_symbols, num_bars)

    strategy_params = {"num_factors": 3}
    backtest_params = {"start": datetime(2024, 2, 1), "end": datetime(2024, 6, 1)}

    args_list = [(store, strategy_params, backtest_params) for _ in range(num_engines)]

    print("Running sequentially...")
    start_time = time.time()
    sequential_results = [run_single_backtest(args) for args in args_list]
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f}s")

    print("Running concurrently (threads)...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_engines) as executor:
        thread_results = list(executor.map(run_single_backtest, args_list))
    thread_time = time.time() - start_time
    print(f"Thread time: {thread_time:.2f}s")

    print("Running concurrently (processes)...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=min(4, num_engines)) as executor:
        process_results = list(executor.map(run_single_backtest, args_list))
    process_time = time.time() - start_time
    print(f"Process time: {process_time:.2f}s")

    print(f"\n=== Results Comparison ===")
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Threads: {thread_time:.2f}s (speedup: {sequential_time / thread_time:.2f}x)")
    print(f"Processes: {process_time:.2f}s (speedup: {sequential_time / process_time:.2f}x)")

    assert len(sequential_results) == num_engines
    assert len(thread_results) == num_engines
    assert len(process_results) == num_engines

    for i in range(num_engines):
        assert abs(sequential_results[i][0] - thread_results[i][0]) < 1e-6
        assert abs(sequential_results[i][0] - process_results[i][0]) < 1e-6


@pytest.mark.performance
def test_factor_cache_effectiveness():
    num_symbols = 50
    num_bars = 200
    num_iterations = 100

    print(f"\n=== Factor Cache Effectiveness ===")

    store = generate_large_universe(num_symbols, num_bars)
    factor = MomentumFactor(lookback=20)

    date_range = store.get_date_range()
    timestamps = pd.date_range(start=date_range.start, end=date_range.end, freq='D')
    test_timestamps = timestamps[-50:].tolist()

    print("Without caching (recomputing each time)...")
    start_time = time.time()
    for _ in range(num_iterations):
        timestamp = test_timestamps[0]
        view = store.get_view(timestamp)
        scores = factor.compute(view)
    no_cache_time = time.time() - start_time

    cache = {}

    print("With manual caching...")
    start_time = time.time()
    for _ in range(num_iterations):
        timestamp = test_timestamps[0]
        cache_key = (timestamp, factor.__class__.__name__)

        if cache_key not in cache:
            view = store.get_view(timestamp)
            cache[cache_key] = factor.compute(view)
        scores = cache[cache_key]
    cache_time = time.time() - start_time

    print(f"\n=== Cache Performance ===")
    print(f"No cache: {no_cache_time:.2f}s")
    print(f"With cache: {cache_time:.2f}s")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x")
    print(f"Cache hit ratio: {((num_iterations - 1) / num_iterations) * 100:.1f}%")

    assert cache_time < no_cache_time
    assert (no_cache_time / cache_time) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
