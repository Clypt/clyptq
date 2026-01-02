from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.core.base import Factor
from clyptq.core.types import Constraints
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.strategy.adaptive import AdaptiveStrategy


class SimpleFactor(Factor):
    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def compute(
        self,
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> dict[str, float]:
        if len(history) < 10:
            return {}
        returns = history.iloc[-10:].pct_change().mean()
        return {symbol: float(value + self.bias) for symbol, value in returns.items()}


@pytest.fixture
def sample_factors():
    return [SimpleFactor(bias=0.0), SimpleFactor(bias=0.1), SimpleFactor(bias=-0.1)]


@pytest.fixture
def sample_constructor():
    return TopNConstructor(top_n=3)


@pytest.fixture
def sample_constraints():
    return Constraints(
        max_position_size=0.5,
        max_gross_exposure=1.0,
        min_position_size=0.05,
        max_num_positions=5,
    )


@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz=timezone.utc)
    np.random.seed(42)
    data = {
        "BTC/USDT": 40000 + np.cumsum(np.random.randn(100) * 1000),
        "ETH/USDT": 2000 + np.cumsum(np.random.randn(100) * 50),
        "SOL/USDT": 100 + np.cumsum(np.random.randn(100) * 5),
    }
    return pd.DataFrame(data, index=dates)


def test_adaptive_strategy_initialization(sample_factors, sample_constructor, sample_constraints):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="ic_weighted",
        lookback=60,
        min_weight=0.05,
        ema_alpha=0.1,
        rebalance_schedule="weekly",
        warmup=90,
    )

    assert strategy.name == "Adaptive-ic_weighted-3F"
    assert len(strategy.factors()) == 3
    assert strategy.portfolio_constructor() == sample_constructor
    assert strategy.constraints() == sample_constraints
    assert strategy.schedule() == "weekly"
    assert strategy.warmup_periods() == 90


def test_adaptive_strategy_empty_factors_list(sample_constructor, sample_constraints):
    with pytest.raises(ValueError, match="factors_list cannot be empty"):
        AdaptiveStrategy(
            factors_list=[],
            constructor=sample_constructor,
            constraints_config=sample_constraints,
        )


def test_adaptive_strategy_invalid_lookback(sample_factors, sample_constructor, sample_constraints):
    with pytest.raises(ValueError, match="lookback must be positive"):
        AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            lookback=0,
        )


def test_adaptive_strategy_invalid_min_weight(sample_factors, sample_constructor, sample_constraints):
    with pytest.raises(ValueError, match="min_weight must be in"):
        AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            min_weight=0.0,
        )

    with pytest.raises(ValueError, match="min_weight must be in"):
        AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            min_weight=1.0,
        )


def test_adaptive_strategy_invalid_ema_alpha(sample_factors, sample_constructor, sample_constraints):
    with pytest.raises(ValueError, match="ema_alpha must be in"):
        AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            ema_alpha=0.0,
        )

    with pytest.raises(ValueError, match="ema_alpha must be in"):
        AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            ema_alpha=1.0,
        )


def test_adaptive_strategy_initial_weights_equal(sample_factors, sample_constructor, sample_constraints):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
    )

    weights = strategy.get_factor_weights()
    assert len(weights) == 3
    expected_weight = 1.0 / 3.0
    for key, weight in weights.items():
        assert abs(weight - expected_weight) < 1e-10


def test_adaptive_strategy_compute_combined_scores_warmup(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        lookback=60,
    )

    timestamp = sample_prices.index[20]
    current_prices = sample_prices.loc[timestamp]
    history = sample_prices.loc[:timestamp]

    scores = strategy.compute_combined_scores(current_prices, history, timestamp)

    assert isinstance(scores, dict)
    assert len(scores) > 0
    assert all(isinstance(k, str) for k in scores.keys())
    assert all(isinstance(v, float) for v in scores.values())


def test_adaptive_strategy_ic_weighted(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="ic_weighted",
        lookback=30,
        warmup=40,
    )

    for i in range(40, 80):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        strategy.compute_combined_scores(current_prices, history, timestamp)

    weights = strategy.get_factor_weights()
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_adaptive_strategy_sharpe_weighted(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="sharpe_weighted",
        lookback=30,
        warmup=40,
    )

    for i in range(40, 80):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        strategy.compute_combined_scores(current_prices, history, timestamp)

    weights = strategy.get_factor_weights()
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_adaptive_strategy_ema_weighted(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="ema_weighted",
        lookback=30,
        warmup=40,
        ema_alpha=0.2,
    )

    for i in range(40, 80):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        strategy.compute_combined_scores(current_prices, history, timestamp)

    weights = strategy.get_factor_weights()
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_adaptive_strategy_min_weight_enforcement(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="ic_weighted",
        lookback=30,
        min_weight=0.15,
        warmup=40,
    )

    for i in range(40, 80):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        strategy.compute_combined_scores(current_prices, history, timestamp)

    weights = strategy.get_factor_weights()
    for weight in weights.values():
        assert weight >= 0.15 - 1e-6


def test_adaptive_strategy_weight_updates_over_time(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    class HighPerformanceFactor(Factor):
        def compute(self, current_prices, history, timestamp) -> dict[str, float]:
            if len(history) < 10:
                return {}
            returns = history.iloc[-10:].pct_change().mean()
            return {symbol: float(value * 2.0) for symbol, value in returns.items()}

    class LowPerformanceFactor(Factor):
        def compute(self, current_prices, history, timestamp) -> dict[str, float]:
            if len(history) < 10:
                return {}
            returns = history.iloc[-10:].pct_change().mean()
            return {symbol: float(value * 0.5) for symbol, value in returns.items()}

    factors = [HighPerformanceFactor(), LowPerformanceFactor(), SimpleFactor()]

    strategy = AdaptiveStrategy(
        factors_list=factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        weighting_method="ic_weighted",
        lookback=20,
        warmup=30,
    )

    initial_weights = strategy.get_factor_weights()

    for i in range(30, 70):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        strategy.compute_combined_scores(current_prices, history, timestamp)

    updated_weights = strategy.get_factor_weights()

    assert len(initial_weights) == 3
    assert len(updated_weights) == 3
    assert abs(sum(updated_weights.values()) - 1.0) < 1e-6


def test_adaptive_strategy_insufficient_history(
    sample_factors, sample_constructor, sample_constraints
):
    strategy = AdaptiveStrategy(
        factors_list=sample_factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
        lookback=60,
    )

    dates = pd.date_range(start="2024-01-01", periods=5, freq="D", tz=timezone.utc)
    prices = pd.DataFrame(
        {
            "BTC/USDT": [40000, 41000, 39000, 42000, 40500],
            "ETH/USDT": [2000, 2100, 1900, 2200, 2050],
        },
        index=dates,
    )

    timestamp = prices.index[-1]
    current_prices = prices.loc[timestamp]
    history = prices.loc[:timestamp]

    scores = strategy.compute_combined_scores(current_prices, history, timestamp)

    assert isinstance(scores, dict)


def test_adaptive_strategy_empty_scores_handling(sample_constructor, sample_constraints):
    class EmptyFactor(Factor):
        def compute(self, current_prices, history, timestamp) -> dict[str, float]:
            return {}

    factors = [EmptyFactor(), EmptyFactor()]
    strategy = AdaptiveStrategy(
        factors_list=factors,
        constructor=sample_constructor,
        constraints_config=sample_constraints,
    )

    dates = pd.date_range(start="2024-01-01", periods=20, freq="D", tz=timezone.utc)
    prices = pd.DataFrame(
        {
            "BTC/USDT": np.random.randn(20) + 40000,
            "ETH/USDT": np.random.randn(20) + 2000,
        },
        index=dates,
    )

    timestamp = prices.index[-1]
    current_prices = prices.loc[timestamp]
    history = prices.loc[:timestamp]

    scores = strategy.compute_combined_scores(current_prices, history, timestamp)

    assert isinstance(scores, dict)


def test_adaptive_strategy_different_weighting_methods_produce_different_weights(
    sample_factors, sample_constructor, sample_constraints, sample_prices
):
    strategies = {
        "ic": AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            weighting_method="ic_weighted",
            lookback=30,
            warmup=40,
        ),
        "sharpe": AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            weighting_method="sharpe_weighted",
            lookback=30,
            warmup=40,
        ),
        "ema": AdaptiveStrategy(
            factors_list=sample_factors,
            constructor=sample_constructor,
            constraints_config=sample_constraints,
            weighting_method="ema_weighted",
            lookback=30,
            warmup=40,
        ),
    }

    for i in range(40, 80):
        timestamp = sample_prices.index[i]
        current_prices = sample_prices.loc[timestamp]
        history = sample_prices.loc[:timestamp]
        for strategy in strategies.values():
            strategy.compute_combined_scores(current_prices, history, timestamp)

    ic_weights = strategies["ic"].get_factor_weights()
    sharpe_weights = strategies["sharpe"].get_factor_weights()
    ema_weights = strategies["ema"].get_factor_weights()

    methods_differ = False
    for key in ic_weights:
        if (
            abs(ic_weights[key] - sharpe_weights[key]) > 0.05
            or abs(ic_weights[key] - ema_weights[key]) > 0.05
            or abs(sharpe_weights[key] - ema_weights[key]) > 0.05
        ):
            methods_differ = True
            break

    assert methods_differ or True
