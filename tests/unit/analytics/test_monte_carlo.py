"""Tests for Monte Carlo simulation."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq import Constraints, CostModel, EngineMode
from clyptq.analytics.risk.monte_carlo import MonteCarloSimulator
from clyptq.data.stores.store import DataStore
from clyptq.trading.engine import Engine
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


def create_test_data():
    """Create deterministic test data."""
    store = DataStore()

    dates = pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    for i, symbol in enumerate(symbols):
        base_price = 100.0 * (i + 1)
        # Create trending + random walk
        prices = [base_price + j * 0.5 + (j % 10) * 2.0 for j in range(100)]

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": [p * 1.005 for p in prices],
                "volume": [1000.0 + j * 10.0 for j in range(100)],
            },
            index=dates,
        )

        store.add_ohlcv(symbol, data)

    return store


def create_test_strategy():
    """Create simple test strategy."""
    factor = MomentumFactor(lookback=10)
    constructor = TopNConstructor(top_n=2)
    constraints = Constraints(max_position_size=0.5, allow_short=False)

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=15,
        name="TestMC",
    )

    return strategy


def test_monte_carlo_basic():
    """Test basic Monte Carlo simulation execution."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Run backtest
    start = datetime(2023, 1, 20)
    end = datetime(2023, 4, 10)
    engine.run(start, end, verbose=False)

    # Run Monte Carlo
    mc_result = engine.run_monte_carlo(num_simulations=100, random_seed=42)

    # Check result structure
    assert mc_result.num_simulations == 100
    assert len(mc_result.final_equities) == 100
    assert len(mc_result.equity_paths) == 100

    # Check statistics are calculated
    assert mc_result.mean_return is not None
    assert mc_result.median_return is not None
    assert mc_result.std_return > 0

    # Check confidence intervals
    assert mc_result.ci_5_return < mc_result.ci_50_return
    assert mc_result.ci_50_return < mc_result.ci_95_return

    # Check risk metrics
    assert 0 <= mc_result.probability_of_loss <= 1
    assert mc_result.max_drawdown_5 >= 0
    assert mc_result.max_drawdown_50 >= 0
    assert mc_result.max_drawdown_95 >= 0


def test_monte_carlo_reproducibility():
    """Test that Monte Carlo with same seed gives same results."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 4, 10)
    engine.run(start, end, verbose=False)

    # Run twice with same seed
    mc1 = engine.run_monte_carlo(num_simulations=50, random_seed=42)

    # Reset and run again
    engine.reset()
    engine.run(start, end, verbose=False)
    mc2 = engine.run_monte_carlo(num_simulations=50, random_seed=42)

    # Results should be identical
    assert mc1.mean_return == mc2.mean_return
    assert mc1.median_return == mc2.median_return
    assert mc1.ci_5_return == mc2.ci_5_return
    assert mc1.ci_95_return == mc2.ci_95_return


def test_monte_carlo_no_backtest_error():
    """Test error when Monte Carlo called without backtest."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    # Try to run Monte Carlo without backtest
    with pytest.raises(ValueError, match="No backtest results available"):
        engine.run_monte_carlo(num_simulations=10)


def test_monte_carlo_statistics_properties():
    """Test statistical properties of Monte Carlo results."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 4, 10)
    engine.run(start, end, verbose=False)

    mc_result = engine.run_monte_carlo(num_simulations=200, random_seed=42)

    # Median should be close to 50th percentile
    assert abs(mc_result.median_return - mc_result.ci_50_return) < 0.01

    # Max drawdown should increase from 5th to 95th percentile
    assert mc_result.max_drawdown_5 <= mc_result.max_drawdown_50
    assert mc_result.max_drawdown_50 <= mc_result.max_drawdown_95

    # Expected shortfall should be negative (loss)
    assert mc_result.expected_shortfall_5 < 0

    # Sharpe confidence interval should be ordered
    assert mc_result.ci_5_sharpe <= mc_result.mean_sharpe
    assert mc_result.mean_sharpe <= mc_result.ci_95_sharpe


def test_monte_carlo_to_dict():
    """Test MonteCarloResult serialization."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 4, 10)
    engine.run(start, end, verbose=False)

    mc_result = engine.run_monte_carlo(num_simulations=50, random_seed=42)

    # Convert to dict
    result_dict = mc_result.to_dict()

    # Check structure
    assert "num_simulations" in result_dict
    assert "summary" in result_dict
    assert "confidence_intervals" in result_dict
    assert "risk_metrics" in result_dict
    assert "sharpe_distribution" in result_dict
    assert "parameters" in result_dict

    # Check values
    assert result_dict["num_simulations"] == 50
    assert "mean_return" in result_dict["summary"]
    assert "5th_percentile" in result_dict["confidence_intervals"]
    assert "probability_of_loss" in result_dict["risk_metrics"]


def test_simulator_direct_usage():
    """Test MonteCarloSimulator can be used directly."""
    store = create_test_data()
    strategy = create_test_strategy()

    cost_model = CostModel()
    executor = BacktestExecutor(cost_model)

    engine = Engine(
        strategy=strategy,
        data_store=store,
        executor=executor,
        initial_capital=10000.0,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 4, 10)
    backtest_result = engine.run(start, end, verbose=False)

    # Use simulator directly
    simulator = MonteCarloSimulator(num_simulations=50, random_seed=42)
    mc_result = simulator.run(backtest_result, initial_capital=10000.0)

    assert mc_result.num_simulations == 50
    assert len(mc_result.final_equities) == 50
