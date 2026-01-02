from datetime import datetime, timezone

import pandas as pd

from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.portfolio.risk_budget import RiskBudgetConstructor
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


class RiskBudgetStrategy(SimpleStrategy):
    def __init__(
        self,
        risk_model: str = "sample_cov",
        risk_budgets: dict[str, float] | None = None,
    ):
        self.risk_model = risk_model
        self.risk_budgets = risk_budgets

    @property
    def name(self) -> str:
        if self.risk_budgets is None:
            return "RB-ERC"
        return "RB-Custom"

    def factors(self) -> list[Factor]:
        return [MomentumFactor(lookback=20)]

    def portfolio_constructor(self) -> RiskBudgetConstructor:
        return RiskBudgetConstructor(
            risk_model=self.risk_model,
            risk_budgets=self.risk_budgets,
            lookback=60,
        )

    def constraints(self) -> Constraints:
        return Constraints(
            max_position_size=0.5,
            max_gross_exposure=1.0,
            min_position_size=0.05,
            max_num_positions=10,
        )

    def schedule(self) -> str:
        return "weekly"

    def warmup_periods(self) -> int:
        return 70


def main():
    print("Risk Budgeting Portfolio Construction Example")
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
        ("Equal Risk Contribution (ERC)", "sample_cov", None),
        ("Custom Risk Budget (50/30/20)", "sample_cov",
         {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}),
        ("Ledoit-Wolf Covariance", "ledoit_wolf", None),
    ]

    print("Running backtests for different configurations:")
    print()

    results = []
    for name, risk_model, risk_budgets in strategies:
        strategy = RiskBudgetStrategy(
            risk_model=risk_model,
            risk_budgets=risk_budgets,
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

        if len(result.portfolio_snapshots) > 0:
            final_snapshot = result.portfolio_snapshots[-1]
            weights = {
                pos.symbol: pos.market_value / final_snapshot.total_value
                for pos in final_snapshot.positions.values()
                if pos.market_value > 0
            }
            if weights:
                constructor = strategy.portfolio_constructor()
                constructor.fit(store.prices)
                risk_contrib = constructor.get_risk_contributions(weights)
                print(f"  Risk Contributions:")
                for symbol, contrib in sorted(risk_contrib.items(), key=lambda x: -x[1]):
                    print(f"    {symbol:<12} {contrib:>7.2%}")

        print()

    print()
    print("Strategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
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
    print("1. ERC allocates risk equally across assets")
    print("2. Custom budgets allow targeting specific risk exposures")
    print("3. Ledoit-Wolf shrinkage improves covariance estimation")
    print("4. Risk budgeting doesn't require return forecasts")
    print("5. More stable than mean-variance in volatile markets")
    print()


if __name__ == "__main__":
    main()
