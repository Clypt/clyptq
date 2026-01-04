from datetime import datetime, timezone

import pandas as pd

from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.portfolio.mean_variance import MeanVarianceConstructor
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


class MeanVarianceStrategy(SimpleStrategy):
    def __init__(
        self,
        return_model: str = "historical",
        risk_model: str = "sample_cov",
        risk_aversion: float = 1.0,
        turnover_penalty: float = 0.1,
    ):
        self.return_model = return_model
        self.risk_model = risk_model
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty

    @property
    def name(self) -> str:
        return f"MV-{self.return_model}-{self.risk_model}"

    def factors(self) -> list[Factor]:
        return [MomentumFactor(lookback=20)]

    def portfolio_constructor(self) -> MeanVarianceConstructor:
        return MeanVarianceConstructor(
            return_model=self.return_model,
            risk_model=self.risk_model,
            risk_aversion=self.risk_aversion,
            turnover_penalty=self.turnover_penalty,
            lookback=60,
        )

    def constraints(self) -> Constraints:
        return Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=10,
        )

    def schedule(self) -> str:
        return "weekly"

    def warmup_periods(self) -> int:
        return 70


def main():
    print("Mean-Variance Portfolio Optimization Example")
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

    strategies = [
        ("Historical Mean + Sample Cov", "historical", "sample_cov", 1.0),
        ("Shrinkage Mean + Ledoit-Wolf", "shrinkage", "ledoit_wolf", 1.0),
        ("Factor Model + Sample Cov", "factor_model", "sample_cov", 1.0),
        ("High Risk Aversion (λ=5)", "historical", "sample_cov", 5.0),
        ("With Turnover Penalty", "historical", "sample_cov", 1.0),
    ]

    print("Running backtests for different configurations:")
    print()

    results = []
    for name, return_model, risk_model, risk_aversion in strategies:
        turnover_penalty = 0.5 if "Turnover" in name else 0.0

        strategy = MeanVarianceStrategy(
            return_model=return_model,
            risk_model=risk_model,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
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
        results.append((name, result))

        print(f"{name}:")
        print(f"  Total Return:  {result.total_return:>8.2%}")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:>8.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown:>8.2%}")
        print(f"  Win Rate:      {result.win_rate:>8.2%}")
        print(f"  Trades:        {result.num_trades:>8}")
        print()

    print()
    print("Strategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 80)

    for name, result in results:
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
    print("1. Shrinkage estimators reduce estimation error in return forecasts")
    print("2. Ledoit-Wolf covariance improves stability in small samples")
    print("3. Higher risk aversion reduces volatility but may sacrifice returns")
    print("4. Turnover penalty reduces trading costs but may miss opportunities")
    print()


if __name__ == "__main__":
    main()
