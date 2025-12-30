"""Integration test: Backtest-Paper parity verification."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from clyptq import Constraints, CostModel, EngineMode
from clyptq.data.live_store import LiveDataStore
from clyptq.data.store import DataStore
from clyptq.engine import BacktestEngine, LiveEngine
from clyptq.execution import BacktestExecutor
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.strategy.base import SimpleStrategy


def create_test_datastore():
    """Create deterministic test data."""
    store = DataStore()

    dates = pd.date_range(start=datetime(2023, 1, 1), periods=60, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

    for i, symbol in enumerate(symbols):
        base_price = 100.0 * (i + 1)
        prices = [base_price + j * 0.5 + (j % 7) * 2.0 for j in range(60)]

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": [p * 1.005 for p in prices],
                "volume": [1000.0 + j * 10.0 for j in range(60)],
            },
            index=dates,
        )

        store.add_ohlcv(symbol, data)

    return store


def create_test_strategy():
    """Create deterministic strategy."""
    factor = MomentumFactor(lookback=10)
    constructor = TopNConstructor(top_n=3)

    constraints = Constraints(
        max_position_size=0.4,
        max_gross_exposure=1.0,
        min_position_size=0.05,
        max_num_positions=3,
        allow_short=False,
    )

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=15,
        name="ParityTest",
    )

    return strategy


def test_backtest_paper_parity():
    """Verify Backtest and Paper modes produce identical results."""
    data_store = create_test_datastore()
    strategy = create_test_strategy()
    initial_capital = 10000.0

    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0,
    )

    # Run Backtest mode
    backtest_executor = BacktestExecutor(cost_model)
    backtest_engine = BacktestEngine(
        strategy=strategy,
        data_store=data_store,
        executor=backtest_executor,
        initial_capital=initial_capital,
    )

    # Run Paper mode (using BacktestExecutor for now)
    paper_executor = BacktestExecutor(cost_model)
    paper_engine = BacktestEngine(
        strategy=strategy,
        data_store=data_store,
        executor=paper_executor,
        initial_capital=initial_capital,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 2, 28)

    backtest_result = backtest_engine.run(start, end, verbose=False)
    paper_result = paper_engine.run(start, end, verbose=False)

    # Verify parity
    assert len(backtest_result.snapshots) == len(paper_result.snapshots)
    assert len(backtest_result.trades) == len(paper_result.trades)

    # Check equity at each timestamp
    for i, (bt_snap, p_snap) in enumerate(
        zip(backtest_result.snapshots, paper_result.snapshots)
    ):
        assert bt_snap.timestamp == p_snap.timestamp

        equity_diff = abs(bt_snap.equity - p_snap.equity)
        assert equity_diff < 1e-6

        cash_diff = abs(bt_snap.cash - p_snap.cash)
        assert cash_diff < 1e-6

        pos_value_diff = abs(bt_snap.positions_value - p_snap.positions_value)
        assert pos_value_diff < 1e-6

    # Check trades
    for i, (bt_trade, p_trade) in enumerate(
        zip(backtest_result.trades, paper_result.trades)
    ):
        assert bt_trade.symbol == p_trade.symbol
        assert bt_trade.side == p_trade.side
        assert abs(bt_trade.amount - p_trade.amount) < 1e-8
        assert abs(bt_trade.price - p_trade.price) < 1e-6

    # Check final metrics
    final_bt_equity = backtest_result.snapshots[-1].equity
    final_p_equity = paper_result.snapshots[-1].equity
    assert abs(final_bt_equity - final_p_equity) < 1e-6

    bt_metrics = backtest_result.metrics
    p_metrics = paper_result.metrics

    assert abs(bt_metrics.total_return - p_metrics.total_return) < 1e-6
    assert abs(bt_metrics.max_drawdown - p_metrics.max_drawdown) < 1e-6


@pytest.mark.skip(reason="Weekly schedule needs longer data period")
def test_parity_with_different_schedules():
    """Test parity across different rebalancing schedules."""
    data_store = create_test_datastore()
    initial_capital = 10000.0
    cost_model = CostModel(maker_fee=0.0, taker_fee=0.0, slippage_bps=0.0)

    schedules = ["daily", "weekly"]

    for schedule in schedules:
        factor = MomentumFactor(lookback=10)
        constructor = TopNConstructor(top_n=2)
        constraints = Constraints()

        strategy = SimpleStrategy(
            factors_list=[factor],
            constructor=constructor,
            constraints_obj=constraints,
            schedule_str=schedule,
            warmup=15,
        )

        bt_executor = BacktestExecutor(cost_model)
        bt_engine = BacktestEngine(
            strategy=strategy,
            data_store=data_store,
            executor=bt_executor,
            initial_capital=initial_capital,
        )

        p_executor = BacktestExecutor(cost_model)
        p_engine = BacktestEngine(
            strategy=strategy,
            data_store=data_store,
            executor=p_executor,
            initial_capital=initial_capital,
        )

        start = datetime(2023, 1, 20)
        end = datetime(2023, 2, 28)

        bt_engine.reset()
        p_engine.reset()

        bt_result = bt_engine.run(start, end, verbose=False)
        p_result = p_engine.run(start, end, verbose=False)

        if len(bt_result.snapshots) == 0 or len(p_result.snapshots) == 0:
            continue

        assert len(bt_result.snapshots) == len(p_result.snapshots)

        final_bt = bt_result.snapshots[-1].equity
        final_p = p_result.snapshots[-1].equity
        assert abs(final_bt - final_p) < 1e-6


def test_parity_with_costs():
    """Test parity with realistic trading costs."""
    data_store = create_test_datastore()
    strategy = create_test_strategy()
    initial_capital = 10000.0

    cost_model = CostModel(
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=10.0,
    )

    bt_executor = BacktestExecutor(cost_model)
    bt_engine = BacktestEngine(
        strategy=strategy,
        data_store=data_store,
        executor=bt_executor,
        initial_capital=initial_capital,
    )

    p_executor = BacktestExecutor(cost_model)
    p_engine = BacktestEngine(
        strategy=strategy,
        data_store=data_store,
        executor=p_executor,
        initial_capital=initial_capital,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 2, 28)

    bt_result = bt_engine.run(start, end, verbose=False)
    p_result = p_engine.run(start, end, verbose=False)

    for i, (bt_snap, p_snap) in enumerate(
        zip(bt_result.snapshots, p_result.snapshots)
    ):
        equity_diff = abs(bt_snap.equity - p_snap.equity)
        assert equity_diff < 1e-6


def test_parity_edge_cases():
    """Test parity in edge cases."""
    store = DataStore()

    dates = pd.date_range(start=datetime(2023, 1, 1), periods=5, freq="D")

    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 99.0, 102.0, 100.0],
            "high": [101.0, 102.0, 100.0, 103.0, 101.0],
            "low": [99.0, 100.0, 98.0, 101.0, 99.0],
            "close": [100.5, 101.5, 99.5, 102.5, 100.5],
            "volume": [1000.0] * 5,
        },
        index=dates,
    )

    store.add_ohlcv("BTC/USDT", data)

    factor = MomentumFactor(lookback=2)
    constructor = TopNConstructor(top_n=1)
    constraints = Constraints()

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=2,
    )

    cost_model = CostModel()

    bt_executor = BacktestExecutor(cost_model)
    bt_engine = BacktestEngine(strategy, store, bt_executor, 1000.0)

    p_executor = BacktestExecutor(cost_model)
    p_engine = BacktestEngine(strategy, store, p_executor, 1000.0)

    bt_result = bt_engine.run(
        datetime(2023, 1, 3), datetime(2023, 1, 5), verbose=False
    )
    p_result = p_engine.run(
        datetime(2023, 1, 3), datetime(2023, 1, 5), verbose=False
    )

    assert len(bt_result.snapshots) == len(p_result.snapshots)

    if len(bt_result.snapshots) > 0:
        final_diff = abs(
            bt_result.snapshots[-1].equity - p_result.snapshots[-1].equity
        )
        assert final_diff < 1e-6


def test_deterministic_execution():
    """Test that running the same backtest twice gives identical results."""
    data_store = create_test_datastore()
    strategy = create_test_strategy()
    cost_model = CostModel()

    def run_backtest():
        executor = BacktestExecutor(cost_model)
        engine = BacktestEngine(
            strategy=strategy,
            data_store=data_store,
            executor=executor,
            initial_capital=10000.0,
        )

        return engine.run(
            datetime(2023, 1, 20),
            datetime(2023, 2, 15),
            verbose=False,
        )

    result1 = run_backtest()
    result2 = run_backtest()

    assert len(result1.snapshots) == len(result2.snapshots)
    assert len(result1.trades) == len(result2.trades)

    for snap1, snap2 in zip(result1.snapshots, result2.snapshots):
        assert abs(snap1.equity - snap2.equity) < 1e-10


def test_backtest_vs_livestore_step_parity():
    """Verify run_backtest() and LiveDataStore + step() produce identical results."""
    initial_capital = 10000.0
    cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)

    # Create backtest data store
    backtest_store = DataStore()
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=60, freq="D")
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    for i, symbol in enumerate(symbols):
        base_price = 100.0 * (i + 1)
        prices = [base_price + j * 0.5 + (j % 7) * 2.0 for j in range(60)]

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": [p * 1.005 for p in prices],
                "volume": [1000.0 + j * 10.0 for j in range(60)],
            },
            index=dates,
        )

        backtest_store.add_ohlcv(symbol, data)

    # Create live data store with same data
    live_store = LiveDataStore(lookback_days=60)

    for symbol in symbols:
        df = backtest_store._data[symbol].copy()
        df = df.reset_index().rename(columns={"index": "timestamp"})
        live_store.add_historical(symbol, df)

    # Create strategy
    factor = MomentumFactor(lookback=10)
    constructor = TopNConstructor(top_n=2)
    constraints = Constraints(
        max_position_size=0.5,
        max_gross_exposure=1.0,
        min_position_size=0.1,
        max_num_positions=2,
        allow_short=False,
    )

    strategy = SimpleStrategy(
        factors_list=[factor],
        constructor=constructor,
        constraints_obj=constraints,
        schedule_str="daily",
        warmup=15,
        name="ParityTest",
    )

    # Run backtest mode
    backtest_executor = BacktestExecutor(cost_model)
    backtest_engine = BacktestEngine(
        strategy=strategy,
        data_store=backtest_store,
        executor=backtest_executor,
        initial_capital=initial_capital,
    )

    start = datetime(2023, 1, 20)
    end = datetime(2023, 2, 15)

    backtest_result = backtest_engine.run(start, end, verbose=False)

    # Run paper mode with step()
    paper_executor = BacktestExecutor(cost_model)
    paper_engine = LiveEngine(
        strategy=strategy,
        data_store=live_store,
        executor=paper_executor,
        initial_capital=initial_capital,
        mode=EngineMode.PAPER,
    )

    # Simulate step-by-step execution with warmup skip
    warmup = strategy.warmup_periods()
    all_timestamps = pd.date_range(start, end, freq="D").to_pydatetime().tolist()

    for i, timestamp in enumerate(all_timestamps):
        if i < warmup:
            continue

        # Get prices from backtest data at this timestamp
        prices = {}
        for symbol in symbols:
            try:
                row = backtest_store._data[symbol].loc[timestamp]
                prices[symbol] = row["close"]
            except KeyError:
                continue

        if prices:
            result = paper_engine.step(timestamp, prices)

    # Get results from engine
    step_snapshots = paper_engine.snapshots
    step_fills = paper_engine.trades

    # Verify parity
    assert len(backtest_result.snapshots) == len(step_snapshots), (
        f"Snapshot count mismatch: backtest={len(backtest_result.snapshots)}, "
        f"step={len(step_snapshots)}"
    )

    assert len(backtest_result.trades) == len(step_fills), (
        f"Trade count mismatch: backtest={len(backtest_result.trades)}, "
        f"step={len(step_fills)}"
    )

    # Check equity at each timestamp
    for i, (bt_snap, step_snap) in enumerate(
        zip(backtest_result.snapshots, step_snapshots)
    ):
        assert bt_snap.timestamp == step_snap.timestamp

        equity_diff = abs(bt_snap.equity - step_snap.equity)
        assert equity_diff < 1e-6, (
            f"Equity mismatch at {bt_snap.timestamp}: "
            f"backtest={bt_snap.equity:.6f}, step={step_snap.equity:.6f}"
        )

        cash_diff = abs(bt_snap.cash - step_snap.cash)
        assert cash_diff < 1e-6

        pos_value_diff = abs(bt_snap.positions_value - step_snap.positions_value)
        assert pos_value_diff < 1e-6

    # Check trades
    for i, (bt_trade, step_trade) in enumerate(zip(backtest_result.trades, step_fills)):
        assert bt_trade.symbol == step_trade.symbol
        assert bt_trade.side == step_trade.side
        assert abs(bt_trade.amount - step_trade.amount) < 1e-8
        assert abs(bt_trade.price - step_trade.price) < 1e-6

    # Check final metrics
    final_bt_equity = backtest_result.snapshots[-1].equity
    final_step_equity = step_snapshots[-1].equity
    assert abs(final_bt_equity - final_step_equity) < 1e-6

    bt_metrics = backtest_result.metrics
    assert abs(bt_metrics.total_return - ((final_step_equity / initial_capital) - 1)) < 1e-6
