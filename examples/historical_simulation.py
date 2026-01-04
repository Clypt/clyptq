from datetime import datetime, timezone

import pandas as pd
from clyptq.core.base import Factor
from clyptq.trading.optimization.validation import HistoricalSimulator
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.execution import BacktestExecutor
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
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"Momentum-{self.lookback}"

    def factors(self):
        return [MomentumFactor(lookback=self.lookback)]

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


def main():
    print("Historical Simulation Testing Example")
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
        return MomentumStrategy(lookback=lookback)

    simulator = HistoricalSimulator(
        strategy_factory=strategy_factory,
        data_store=store,
        executor=executor,
        initial_capital=100_000.0,
        overfitting_threshold=0.3,
    )

    start_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    mid_date = datetime(2024, 7, 1, tzinfo=timezone.utc)

    print("\n2. Out-of-Sample Testing")
    print("-" * 60)
    oos_result = simulator.run_out_of_sample(
        train_start=start_date,
        train_end=mid_date,
        test_start=mid_date + pd.Timedelta(days=1),
        test_end=end_date,
        params={"lookback": 20},
    )

    print(f"Train Sharpe: {oos_result.train_result.metrics.sharpe_ratio:.2f}")
    print(f"Test Sharpe: {oos_result.test_result.metrics.sharpe_ratio:.2f}")
    print(f"Degradation Ratio: {oos_result.degradation_ratio:.2%}")
    print(f"Stability Score: {oos_result.stability_score:.2%}")
    print(f"Overfitted: {oos_result.is_overfitted}")

    print("\n3. Walk-Forward Analysis")
    print("-" * 60)
    wf_result = simulator.run_walk_forward(
        total_start=start_date,
        total_end=end_date,
        train_window_days=60,
        test_window_days=30,
        params={"lookback": 20},
    )

    print(f"Number of periods: {len(wf_result.periods)}")
    print(f"Mean train Sharpe: {wf_result.mean_train_sharpe:.2f}")
    print(f"Mean test Sharpe: {wf_result.mean_test_sharpe:.2f}")
    print(f"Mean degradation: {wf_result.mean_degradation:.2%}")
    print(f"Consistency score: {wf_result.consistency_score:.2%}")
    print(f"Overfitting ratio: {wf_result.overfitting_ratio:.2%}")

    print("\n4. Parameter Stability Analysis")
    print("-" * 60)
    param_grid = [
        {"lookback": 10},
        {"lookback": 15},
        {"lookback": 20},
        {"lookback": 25},
        {"lookback": 30},
    ]

    ps_result = simulator.analyze_parameter_stability(
        param_grid=param_grid,
        train_start=start_date,
        train_end=mid_date,
        test_start=mid_date + pd.Timedelta(days=1),
        test_end=end_date,
    )

    print(f"Best parameters: {ps_result.best_params}")
    print(f"Number of configs tested: {len(ps_result.param_results)}")
    print(f"Robust configurations: {len(ps_result.robust_params)}")
    print("\nStability scores:")
    for config_key, score in sorted(
        ps_result.stability_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        config_idx = int(config_key.split("_")[1])
        lookback = param_grid[config_idx]["lookback"]
        print(f"  Lookback {lookback}: {score:.2%}")

    print("\n5. Export Results")
    print("-" * 60)
    oos_dict = oos_result.to_dict()
    print(f"Train return: {oos_dict['train_return']:.2%}")
    print(f"Test return: {oos_dict['test_return']:.2%}")
    print(f"Sharpe degradation: {oos_dict['sharpe_degradation']:.2%}")
    print(f"Return degradation: {oos_dict['return_degradation']:.2%}")

    print("\n" + "=" * 60)
    print("Historical simulation testing complete!")


if __name__ == "__main__":
    main()
