from datetime import datetime, timezone

import pandas as pd
from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.optimization.grid_search import GridSearchOptimizer
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


class MomentumStrategy(SimpleStrategy):
    def __init__(self, lookback: int = 20, top_n: int = 3):
        self.lookback = lookback
        self.top_n = top_n

    @property
    def name(self) -> str:
        return f"Momentum-{self.lookback}-Top{self.top_n}"

    def factors(self):
        return [MomentumFactor(lookback=self.lookback)]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=self.top_n)

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


def main():
    print("Grid Search Parameter Optimization Example")
    print("=" * 60)

    print("\n1. Loading data...")
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]

    store = DataStore()
    for symbol in universe:
        df = load_crypto_data(
            symbol=symbol,
            exchange_id="binance",
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            limit=365,
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

    def strategy_factory(params):
        lookback = params.get("lookback", 20)
        top_n = params.get("top_n", 3)
        return MomentumStrategy(lookback=lookback, top_n=top_n)

    optimizer = GridSearchOptimizer(
        strategy_factory=strategy_factory,
        data_store=store,
        executor=executor,
        initial_capital=100_000.0,
        scoring_metric="sharpe_ratio",
    )

    start_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    print("\n2. Single Parameter Grid Search")
    print("-" * 60)
    param_grid = {
        "lookback": [10, 15, 20, 25, 30],
    }

    result = optimizer.search(
        param_grid=param_grid,
        start=start_date,
        end=end_date,
    )

    print(f"Best parameters: {result.best_params}")
    print(f"Best Sharpe ratio: {result.best_score:.2f}")
    print(f"\nAll results:")
    for item in result.all_results:
        print(f"  {item['params']}: {item['score']:.2f}")

    print("\n3. Multi-Parameter Grid Search")
    print("-" * 60)
    param_grid = {
        "lookback": [15, 20, 25],
        "top_n": [2, 3, 4],
    }

    result = optimizer.search(
        param_grid=param_grid,
        start=start_date,
        end=end_date,
    )

    print(f"Best parameters: {result.best_params}")
    print(f"Best Sharpe ratio: {result.best_score:.2f}")
    print(f"Total combinations tested: {len(result.all_results)}")

    result_dict = result.to_dict()
    print(f"\nTop 5 parameter combinations:")
    for params_str, score in result_dict["top_5_params"]:
        print(f"  {params_str}: {score:.2f}")

    print("\n4. Cross-Validation Search")
    print("-" * 60)
    param_grid = {
        "lookback": [15, 20, 25],
    }

    result = optimizer.search(
        param_grid=param_grid,
        start=start_date,
        end=end_date,
        cv_folds=3,
    )

    print(f"Best parameters (3-fold CV): {result.best_params}")
    print(f"Best avg Sharpe ratio: {result.best_score:.2f}")

    print("\n5. Different Scoring Metrics")
    print("-" * 60)
    metrics = ["sharpe_ratio", "total_return", "sortino_ratio", "calmar_ratio"]

    for metric in metrics:
        optimizer = GridSearchOptimizer(
            strategy_factory=strategy_factory,
            data_store=store,
            executor=executor,
            initial_capital=100_000.0,
            scoring_metric=metric,
        )

        param_grid = {"lookback": [15, 20, 25]}

        result = optimizer.search(
            param_grid=param_grid,
            start=start_date,
            end=end_date,
        )

        print(f"{metric:>15}: {result.best_params} → {result.best_score:.2f}")

    print("\n" + "=" * 60)
    print("Grid search optimization complete!")


if __name__ == "__main__":
    main()
