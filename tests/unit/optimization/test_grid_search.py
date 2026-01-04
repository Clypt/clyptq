from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.core.base import Factor
from clyptq.core.types import Constraints, CostModel
from clyptq.data.stores.store import DataStore
from clyptq.trading.execution import BacktestExecutor
from clyptq.trading.optimization.grid_search import GridSearchOptimizer
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.base import SimpleStrategy


class SimpleFactor(Factor):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, current_prices, history, timestamp):
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
        df = pd.DataFrame(
            {
                "open": sample_data[symbol],
                "high": sample_data[symbol] * 1.01,
                "low": sample_data[symbol] * 0.99,
                "close": sample_data[symbol],
                "volume": 1000000.0,
            },
            index=sample_data.index,
        )
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
def optimizer(strategy_factory, data_store, executor):
    return GridSearchOptimizer(
        strategy_factory=strategy_factory,
        data_store=data_store,
        executor=executor,
        initial_capital=100_000.0,
        scoring_metric="sharpe_ratio",
    )


def test_grid_search_basic(optimizer, sample_data):
    param_grid = {
        "lookback": [10, 20, 30],
    }

    result = optimizer.search(
        param_grid=param_grid,
        start=sample_data.index[30],
        end=sample_data.index[150],
    )

    assert result.best_params is not None
    assert "lookback" in result.best_params
    assert result.best_params["lookback"] in [10, 20, 30]
    assert isinstance(result.best_score, float)
    assert len(result.all_results) == 3


def test_grid_search_to_dict(optimizer, sample_data):
    param_grid = {"lookback": [15, 25]}

    result = optimizer.search(
        param_grid=param_grid,
        start=sample_data.index[30],
        end=sample_data.index[100],
    )

    result_dict = result.to_dict()

    assert "best_params" in result_dict
    assert "best_score" in result_dict
    assert "num_combinations_tested" in result_dict
    assert result_dict["num_combinations_tested"] == 2


def test_grid_search_cross_validation(optimizer, sample_data):
    param_grid = {"lookback": [10, 20]}

    result = optimizer.search(
        param_grid=param_grid,
        start=sample_data.index[30],
        end=sample_data.index[150],
        cv_folds=3,
    )

    assert result.best_params is not None
    assert isinstance(result.best_score, float)
    assert len(result.all_results) == 2


def test_grid_search_different_metrics(strategy_factory, data_store, executor, sample_data):
    metrics = ["sharpe_ratio", "total_return", "sortino_ratio", "calmar_ratio"]

    for metric in metrics:
        optimizer = GridSearchOptimizer(
            strategy_factory=strategy_factory,
            data_store=data_store,
            executor=executor,
            initial_capital=100_000.0,
            scoring_metric=metric,
        )

        param_grid = {"lookback": [15, 25]}

        result = optimizer.search(
            param_grid=param_grid,
            start=sample_data.index[30],
            end=sample_data.index[100],
        )

        assert result.best_params is not None
        assert isinstance(result.best_score, float)


def test_grid_search_top_params(optimizer, sample_data):
    param_grid = {"lookback": [10, 15, 20, 25, 30]}

    result = optimizer.search(
        param_grid=param_grid,
        start=sample_data.index[30],
        end=sample_data.index[150],
    )

    result_dict = result.to_dict()
    top_5 = result_dict["top_5_params"]

    assert len(top_5) <= 5
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_5)
    assert top_5 == sorted(top_5, key=lambda x: x[1], reverse=True)
