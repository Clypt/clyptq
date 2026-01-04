import time
from datetime import datetime, timezone

import pandas as pd
from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.parallel import ParallelFactorComputer
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class MomentumFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        if len(history) < self.lookback:
            return {}
        returns = history.iloc[-self.lookback :].pct_change().mean()
        return {symbol: float(value) for symbol, value in returns.items()}


class VolatilityFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        if len(history) < self.lookback:
            return {}
        volatility = history.iloc[-self.lookback :].pct_change().std()
        return {symbol: -float(value) for symbol, value in volatility.items()}


class MeanReversionFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        if len(history) < self.lookback:
            return {}

        mean_prices = history.iloc[-self.lookback :].mean()
        deviations = (current_prices - mean_prices) / mean_prices

        return {symbol: -float(value) for symbol, value in deviations.items()}


class ParallelStrategy(SimpleStrategy):
    def __init__(
        self,
        use_parallel: bool = True,
        max_workers: int | None = None,
        executor_type: str = "thread",
    ):
        self.use_parallel = use_parallel
        self.parallel_computer = ParallelFactorComputer(
            max_workers=max_workers, executor_type=executor_type
        )
        self._factors = [
            MomentumFactor(lookback=20),
            VolatilityFactor(lookback=20),
            MeanReversionFactor(lookback=20),
        ]

    @property
    def name(self) -> str:
        mode = "Parallel" if self.use_parallel else "Sequential"
        return f"Multi-{mode}"

    def factors(self):
        return self._factors

    def portfolio_constructor(self):
        return TopNConstructor(top_n=3)

    def constraints(self):
        return Constraints(
            max_position_size=0.4,
            max_gross_exposure=1.0,
            min_position_size=0.1,
            max_num_positions=5,
        )

    def schedule(self):
        return "daily"

    def warmup_periods(self):
        return 30

    def compute_combined_scores(self, current_prices, history, timestamp):
        if self.use_parallel:
            factor_scores = self.parallel_computer.compute_factors(
                self._factors, current_prices, history, timestamp
            )
        else:
            factor_scores = self.parallel_computer.compute_factors_sequential(
                self._factors, current_prices, history, timestamp
            )

        all_symbols = set()
        for scores in factor_scores:
            all_symbols.update(scores.keys())

        combined = {}
        for symbol in all_symbols:
            total = sum(scores.get(symbol, 0.0) for scores in factor_scores)
            combined[symbol] = total / len(factor_scores)

        return combined


def main():
    print("Parallel Factor Computation Example")
    print("=" * 60)

    print("\n1. Loading data...")
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]

    store = DataStore()
    for symbol in universe:
        df = load_crypto_data(
            symbol=symbol,
            exchange_id="binance",
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            limit=200,
            timeframe="1d",
        )
        store.add_ohlcv(symbol, df, frequency="1d", source="binance")

    executor = BacktestExecutor(
        cost_model=CostModel(
            taker_fee=0.001,
            maker_fee=0.0005,
            slippage_bps=5.0,
        )
    )

    start_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
    end_date = datetime(2024, 7, 31, tzinfo=timezone.utc)

    print("\n2. Sequential Factor Computation")
    print("-" * 60)
    sequential_strategy = ParallelStrategy(use_parallel=False)
    engine = BacktestEngine(
        strategy=sequential_strategy,
        data_store=store,
        executor=executor,
        initial_capital=100_000.0,
    )

    start_time = time.time()
    sequential_result = engine.run(start=start_date, end=end_date)
    sequential_time = time.time() - start_time

    print(f"Execution time: {sequential_time:.2f}s")
    print(f"Total return: {sequential_result.metrics.total_return:.2%}")
    print(f"Sharpe ratio: {sequential_result.metrics.sharpe_ratio:.2f}")

    print("\n3. Parallel Factor Computation (Thread)")
    print("-" * 60)
    parallel_strategy = ParallelStrategy(
        use_parallel=True, max_workers=3, executor_type="thread"
    )
    engine = BacktestEngine(
        strategy=parallel_strategy,
        data_store=store,
        executor=executor,
        initial_capital=100_000.0,
    )

    start_time = time.time()
    parallel_result = engine.run(start=start_date, end=end_date)
    parallel_time = time.time() - start_time

    print(f"Execution time: {parallel_time:.2f}s")
    print(f"Total return: {parallel_result.metrics.total_return:.2%}")
    print(f"Sharpe ratio: {parallel_result.metrics.sharpe_ratio:.2f}")

    print("\n4. Performance Comparison")
    print("-" * 60)
    speedup = sequential_time / parallel_time
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Speedup:         {speedup:.2f}x")

    print("\n5. Results Consistency Check")
    print("-" * 60)
    print(
        f"Sequential final value: ${sequential_result.metrics.final_value:,.2f}"
    )
    print(f"Parallel final value:   ${parallel_result.metrics.final_value:,.2f}")

    diff = abs(
        sequential_result.metrics.final_value - parallel_result.metrics.final_value
    )
    print(f"Difference:             ${diff:,.2f}")

    if diff < 1.0:
        print("Results are consistent!")
    else:
        print("Warning: Results differ")

    print("\n" + "=" * 60)
    print("Parallel factor computation complete!")


if __name__ == "__main__":
    main()
