from datetime import datetime, timezone

import pandas as pd

from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.adaptive import AdaptiveStrategy


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


class ValueFactor(Factor):
    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        if len(history) < self.lookback:
            return {}

        mean_price = history.iloc[-self.lookback :].mean()
        value_scores = {}
        for symbol in current_prices.index:
            if symbol in mean_price.index:
                ratio = mean_price[symbol] / current_prices[symbol]
                value_scores[symbol] = float(ratio - 1.0)

        return value_scores


def main():
    print("Adaptive Factor Weighting Strategy Example")
    print("=" * 60)
    print()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "MATIC/USDT"]
    print(f"Universe: {', '.join(symbols)}")
    print()

    print("Loading data...")
    data = load_crypto_data(
        exchange="binance",
        symbols=symbols,
        timeframe="1d",
        days=180,
    )
    store = DataStore(data)
    print(f"Loaded {len(data)} days of data")
    print()

    factors = [
        MomentumFactor(lookback=20),
        VolatilityFactor(lookback=20),
        ValueFactor(lookback=60),
    ]

    strategies = [
        ("IC-Weighted (Predictive Power)", "ic_weighted", 60, 0.05),
        ("Sharpe-Weighted (Risk-Adjusted)", "sharpe_weighted", 60, 0.05),
        ("EMA-Weighted (Trend Following)", "ema_weighted", 60, 0.1),
        ("IC-Weighted (Short Lookback)", "ic_weighted", 30, 0.05),
        ("EMA-Weighted (Fast Alpha=0.2)", "ema_weighted", 60, 0.2),
    ]

    print("Running backtests for different weighting methods:")
    print()

    results = []
    for name, method, lookback, param in strategies:
        if method == "ema_weighted":
            ema_alpha = param
            min_weight = 0.05
        else:
            ema_alpha = 0.1
            min_weight = param

        strategy = AdaptiveStrategy(
            factors_list=factors,
            constructor=TopNConstructor(top_n=3),
            constraints_config=Constraints(
                max_position_size=0.5,
                max_gross_exposure=1.0,
                min_position_size=0.05,
                max_num_positions=5,
            ),
            weighting_method=method,
            lookback=lookback,
            min_weight=min_weight,
            ema_alpha=ema_alpha,
            rebalance_schedule="weekly",
            warmup=90,
        )

        executor = BacktestExecutor(
            cost_model=CostModel(
                taker_fee=0.001,
                maker_fee=0.0005,
                slippage=0.0005,
            )
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_store=store,
            executor=executor,
            initial_capital=100_000.0,
        )

        start = datetime(2024, 7, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = engine.run(start=start, end=end)
        results.append((name, result, strategy))

        print(f"{name}:")
        print(f"  Total Return:  {result.total_return:>8.2%}")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:>8.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown:>8.2%}")
        print(f"  Win Rate:      {result.win_rate:>8.2%}")
        print(f"  Trades:        {result.num_trades:>8}")

        factor_weights = strategy.get_factor_weights()
        print(f"  Final Factor Weights:")
        for i, (factor_key, weight) in enumerate(factor_weights.items()):
            factor_name = ["Momentum", "Volatility", "Value"][i]
            print(f"    {factor_name:<12} {weight:>7.2%}")

        print()

    print()
    print("Strategy Comparison:")
    print("-" * 80)
    print(
        f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}"
    )
    print("-" * 80)

    for name, result, _ in results:
        print(
            f"{name:<35} "
            f"{result.total_return:>9.2%} "
            f"{result.sharpe_ratio:>8.2f} "
            f"{result.max_drawdown:>8.2%} "
            f"{result.num_trades:>8}"
        )

    print("-" * 80)
    print()

    print("Key Takeaways:")
    print("1. IC-weighted adapts based on factor predictive power")
    print("2. Sharpe-weighted focuses on risk-adjusted factor performance")
    print("3. EMA-weighted follows recent factor magnitude trends")
    print("4. Shorter lookback (30d) responds faster to market changes")
    print("5. Higher EMA alpha (0.2) gives more weight to recent observations")
    print("6. All methods dynamically adjust weights over time")
    print()

    print("Factor Weight Evolution:")
    print("-" * 60)
    ic_strategy = results[0][2]
    sharpe_strategy = results[1][2]
    ema_strategy = results[2][2]

    print("IC-Weighted:")
    for i, (factor_key, weight) in enumerate(ic_strategy.get_factor_weights().items()):
        factor_name = ["Momentum", "Volatility", "Value"][i]
        print(f"  {factor_name:<12} {weight:>7.2%}")

    print("\nSharpe-Weighted:")
    for i, (factor_key, weight) in enumerate(
        sharpe_strategy.get_factor_weights().items()
    ):
        factor_name = ["Momentum", "Volatility", "Value"][i]
        print(f"  {factor_name:<12} {weight:>7.2%}")

    print("\nEMA-Weighted:")
    for i, (factor_key, weight) in enumerate(ema_strategy.get_factor_weights().items()):
        factor_name = ["Momentum", "Volatility", "Value"][i]
        print(f"  {factor_name:<12} {weight:>7.2%}")

    print()


if __name__ == "__main__":
    main()
