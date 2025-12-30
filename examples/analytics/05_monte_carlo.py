"""
Monte Carlo simulation for backtest validation and risk assessment.

Uses bootstrap sampling of historical returns to generate distribution
of possible outcomes with confidence intervals and risk metrics.
"""

from datetime import datetime

import pandas as pd

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.monte_carlo import print_monte_carlo_results
from clyptq.data.store import DataStore
from clyptq.engine import Engine
from clyptq.execution import BacktestExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.factors.library.volatility import VolatilityFactor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.strategy.base import SimpleStrategy


def create_sample_data(store: DataStore) -> None:
    """Create sample OHLCV data for demonstration."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

    for i, symbol in enumerate(symbols):
        base_price = 100.0 * (i + 1)
        # Create trending + random walk pattern
        prices = [base_price + j * 0.8 + (j % 15) * 3.0 for j in range(200)]

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": [p * 1.01 for p in prices],
                "volume": [10000.0 + j * 50.0 for j in range(200)],
            },
            index=dates,
        )

        store.add_ohlcv(symbol, data)


def main():
    # 1. Setup data
    print("1. Creating sample data...")
    store = DataStore()
    create_sample_data(store)

    # 2. Define strategy
    print("2. Setting up strategy...")
    factor1 = MomentumFactor(lookback=20)
    factor2 = VolatilityFactor(lookback=20)
    constructor = TopNConstructor(top_n=3)
    constraints = Constraints(max_position_size=0.4, allow_short=False)

    strategy = SimpleStrategy(
        factors_list=[factor1, factor2],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=25,
        name="MomentumVolatility",
    )

    # 3. Run backtest
    print("3. Running backtest...")
    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        mode=EngineMode.BACKTEST,
        executor=executor,
        initial_capital=10000.0,
    )

    start = datetime(2023, 2, 1)
    end = datetime(2023, 7, 1)
    result = engine.run(start, end, verbose=False)

    print("\nBacktest Results:")
    print(f"  Total Return: {result.metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    print(f"  Num Trades: {result.metrics.num_trades}")

    # 4. Run Monte Carlo simulation
    print("\n4. Running Monte Carlo simulation (1000 runs)...")
    mc_result = engine.run_monte_carlo(num_simulations=1000, random_seed=42, verbose=False)

    # 5. Display results
    print("\n" + "=" * 70)
    print_monte_carlo_results(mc_result)

    # 6. Interpret results
    print("\nInterpretation:")
    print(f"  - Expected return range: {mc_result.ci_5_return:.2%} to {mc_result.ci_95_return:.2%}")
    print(f"  - Probability of loss: {mc_result.probability_of_loss:.1%}")
    print(f"  - Worst case (5th percentile): {mc_result.expected_shortfall_5:.2%} return")
    print(f"  - Median max drawdown: {mc_result.max_drawdown_50:.2%}")

    if mc_result.probability_of_loss > 0.3:
        print("\n  Risk Assessment: High risk - consider risk management improvements")
    elif mc_result.probability_of_loss > 0.15:
        print("\n  Risk Assessment: Moderate risk - acceptable with monitoring")
    else:
        print("\n  Risk Assessment: Low risk - strategy shows robustness")


if __name__ == "__main__":
    main()
