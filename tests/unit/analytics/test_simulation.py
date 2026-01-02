from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.analytics.simulation import HistoricalSimulator
from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.stores.store import DataStore
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class SimpleFactor(Factor):
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


class TestStrategy(SimpleStrategy):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"Test-{self.lookback}"

    def factors(self):
        return [SimpleFactor(lookback=self.lookback)]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=2)

    def constraints(self):
        return Constraints(
            max_position_size=0.6,
            max_gross_exposure=1.0,
            min_position_size=0.1,
            max_num_positions=3,
        )

    def schedule(self):
        return "weekly"

    def warmup_periods(self):
        return 30


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D", tz=timezone.utc)
    np.random.seed(42)
    data = {
        "BTC/USDT": 40000 + np.cumsum(np.random.randn(200) * 1000),
        "ETH/USDT": 2000 + np.cumsum(np.random.randn(200) * 50),
        "SOL/USDT": 100 + np.cumsum(np.random.randn(200) * 5),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def data_store(sample_data):
    store = DataStore()
    for symbol in sample_data.columns:
        df = pd.DataFrame({
            "open": sample_data[symbol],
            "high": sample_data[symbol] * 1.01,
            "low": sample_data[symbol] * 0.99,
            "close": sample_data[symbol],
            "volume": 1000000.0,
        }, index=sample_data.index)
        store.add_ohlcv(symbol, df, frequency="1d", source="test")
    return store


@pytest.fixture
def executor():
    return BacktestExecutor(
        cost_model=CostModel(
            taker_fee=0.001,
            maker_fee=0.0005,
            slippage_bps=5.0,
        )
    )


@pytest.fixture
def strategy_factory():
    def factory(params):
        lookback = params.get("lookback", 20)
        return TestStrategy(lookback=lookback)

    return factory


@pytest.fixture
def simulator(strategy_factory, data_store, executor):
    return HistoricalSimulator(
        strategy_factory=strategy_factory,
        data_store=data_store,
        executor=executor,
        initial_capital=100_000.0,
        overfitting_threshold=0.3,
    )


def test_simulator_initialization(simulator):
    assert simulator.initial_capital == 100_000.0
    assert simulator.overfitting_threshold == 0.3


def test_out_of_sample_basic(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    result = simulator.run_out_of_sample(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        params={"lookback": 20},
    )

    assert result.train_result is not None
    assert result.test_result is not None
    assert isinstance(result.degradation_ratio, float)
    assert isinstance(result.is_overfitted, bool)
    assert 0.0 <= result.stability_score <= 1.0


def test_out_of_sample_to_dict(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    result = simulator.run_out_of_sample(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        params={"lookback": 20},
    )

    result_dict = result.to_dict()

    assert "train_sharpe" in result_dict
    assert "test_sharpe" in result_dict
    assert "degradation_ratio" in result_dict
    assert "is_overfitted" in result_dict
    assert "stability_score" in result_dict


def test_out_of_sample_default_params(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    result = simulator.run_out_of_sample(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    assert result is not None


def test_degradation_calculation_positive(simulator):
    degradation = simulator._calculate_degradation(2.0, 1.5)
    assert degradation == 0.25


def test_degradation_calculation_negative(simulator):
    degradation = simulator._calculate_degradation(1.5, 2.0)
    assert degradation == (1.5 - 2.0) / 1.5


def test_degradation_calculation_zero_train(simulator):
    degradation = simulator._calculate_degradation(0.0, 1.0)
    assert degradation == 0.0


def test_stability_score_perfect(simulator):
    from clyptq.core.types import BacktestResult, PerformanceMetrics, EngineMode
    from datetime import datetime, timezone

    metrics = PerformanceMetrics(
        total_return=0.2,
        annualized_return=0.2,
        daily_returns=[],
        volatility=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=-0.1,
        num_trades=10,
        win_rate=0.6,
        profit_factor=2.0,
        avg_trade_pnl=1000.0,
        avg_leverage=0.8,
        max_leverage=1.0,
        avg_num_positions=2.5,
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        duration_days=365,
    )

    result_train = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode=EngineMode.BACKTEST,
    )

    result_test = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=metrics,
        strategy_name="Test",
        mode=EngineMode.BACKTEST,
    )

    stability = simulator._calculate_stability(result_train, result_test)
    assert stability == 1.0


def test_stability_score_degraded(simulator):
    from clyptq.core.types import BacktestResult, PerformanceMetrics, EngineMode
    from datetime import datetime, timezone

    metrics_train = PerformanceMetrics(
        total_return=0.2,
        annualized_return=0.2,
        daily_returns=[],
        volatility=0.1,
        sharpe_ratio=2.0,
        sortino_ratio=2.5,
        max_drawdown=-0.1,
        num_trades=10,
        win_rate=0.6,
        profit_factor=2.0,
        avg_trade_pnl=1000.0,
        avg_leverage=0.8,
        max_leverage=1.0,
        avg_num_positions=2.5,
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        duration_days=365,
    )

    metrics_test = PerformanceMetrics(
        total_return=0.1,
        annualized_return=0.1,
        daily_returns=[],
        volatility=0.15,
        sharpe_ratio=1.0,
        sortino_ratio=1.5,
        max_drawdown=-0.2,
        num_trades=10,
        win_rate=0.5,
        profit_factor=1.5,
        avg_trade_pnl=500.0,
        avg_leverage=0.6,
        max_leverage=0.8,
        avg_num_positions=2.0,
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        duration_days=365,
    )

    result_train = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=metrics_train,
        strategy_name="Test",
        mode=EngineMode.BACKTEST,
    )

    result_test = BacktestResult(
        snapshots=[],
        trades=[],
        metrics=metrics_test,
        strategy_name="Test",
        mode=EngineMode.BACKTEST,
    )

    stability = simulator._calculate_stability(result_train, result_test)
    assert 0.0 <= stability < 1.0


def test_consistency_calculation_high(simulator):
    metrics = [1.5, 1.6, 1.4, 1.55, 1.45]
    consistency = simulator._calculate_consistency(metrics)
    assert consistency > 0.8


def test_consistency_calculation_low(simulator):
    metrics = [1.0, 2.0, 0.5, 2.5, 0.8]
    consistency = simulator._calculate_consistency(metrics)
    assert consistency < 0.7


def test_consistency_single_value(simulator):
    metrics = [1.5]
    consistency = simulator._calculate_consistency(metrics)
    assert consistency == 1.0


def test_consistency_zero_mean(simulator):
    metrics = [0.0, 0.0, 0.0]
    consistency = simulator._calculate_consistency(metrics)
    assert consistency == 0.0


def test_walk_forward_basic(simulator, sample_data):
    total_start = sample_data.index[30]
    total_end = sample_data.index[180]

    result = simulator.run_walk_forward(
        total_start=total_start,
        total_end=total_end,
        train_window_days=40,
        test_window_days=20,
        params={"lookback": 20},
    )

    assert len(result.periods) > 0
    assert isinstance(result.mean_train_sharpe, float)
    assert isinstance(result.mean_test_sharpe, float)
    assert isinstance(result.mean_degradation, float)
    assert 0.0 <= result.consistency_score <= 1.0
    assert 0.0 <= result.overfitting_ratio <= 1.0


def test_walk_forward_to_dict(simulator, sample_data):
    total_start = sample_data.index[30]
    total_end = sample_data.index[180]

    result = simulator.run_walk_forward(
        total_start=total_start,
        total_end=total_end,
        train_window_days=40,
        test_window_days=20,
        params={"lookback": 20},
    )

    result_dict = result.to_dict()

    assert "num_periods" in result_dict
    assert "mean_train_sharpe" in result_dict
    assert "mean_test_sharpe" in result_dict
    assert "mean_degradation" in result_dict
    assert "consistency_score" in result_dict
    assert "overfitting_ratio" in result_dict


def test_walk_forward_no_valid_periods(simulator, sample_data):
    total_start = sample_data.index[30]
    total_end = sample_data.index[50]

    with pytest.raises(ValueError, match="No valid walk-forward periods found"):
        simulator.run_walk_forward(
            total_start=total_start,
            total_end=total_end,
            train_window_days=100,
            test_window_days=50,
            params={"lookback": 20},
        )


def test_parameter_stability_basic(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    param_grid = [
        {"lookback": 10},
        {"lookback": 20},
        {"lookback": 30},
    ]

    result = simulator.analyze_parameter_stability(
        param_grid=param_grid,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    assert len(result.param_results) == 3
    assert "lookback" in result.best_params
    assert len(result.stability_scores) == 3
    assert len(result.robust_params) > 0


def test_parameter_stability_to_dict(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    param_grid = [
        {"lookback": 10},
        {"lookback": 20},
    ]

    result = simulator.analyze_parameter_stability(
        param_grid=param_grid,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    result_dict = result.to_dict()

    assert "best_params" in result_dict
    assert "num_configs_tested" in result_dict
    assert "robust_configs" in result_dict
    assert "stability_scores" in result_dict


def test_parameter_stability_best_selection(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    param_grid = [
        {"lookback": 15},
        {"lookback": 25},
    ]

    result = simulator.analyze_parameter_stability(
        param_grid=param_grid,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    assert result.best_params["lookback"] in [15, 25]


def test_overfitting_detection_overfitted(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    simulator.overfitting_threshold = 0.1

    result = simulator.run_out_of_sample(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        params={"lookback": 20},
    )

    if result.degradation_ratio > 0.1:
        assert result.is_overfitted is True
    else:
        assert result.is_overfitted is False


def test_overfitting_detection_not_overfitted(simulator, sample_data):
    train_start = sample_data.index[30]
    train_end = sample_data.index[100]
    test_start = sample_data.index[101]
    test_end = sample_data.index[150]

    simulator.overfitting_threshold = 0.9

    result = simulator.run_out_of_sample(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        params={"lookback": 20},
    )

    assert result.is_overfitted is False
