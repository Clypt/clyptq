from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.core.types import Constraints
from clyptq.trading.portfolio.mean_variance import MeanVarianceConstructor


@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "BTC/USDT": 50000 + np.random.randn(300).cumsum() * 1000,
            "ETH/USDT": 3000 + np.random.randn(300).cumsum() * 100,
            "SOL/USDT": 100 + np.random.randn(300).cumsum() * 5,
        },
        index=dates,
    )
    return prices.clip(lower=1.0)


@pytest.fixture
def sample_scores():
    return {
        "BTC/USDT": 0.8,
        "ETH/USDT": 0.5,
        "SOL/USDT": 0.3,
    }


@pytest.fixture
def sample_constraints():
    return Constraints(
        max_position_size=0.5,
        max_gross_exposure=1.0,
        min_position_size=0.01,
    )


def test_mean_variance_init():
    mv = MeanVarianceConstructor(
        return_model="historical",
        risk_model="sample_cov",
        target_return=0.01,
        risk_aversion=2.0,
        turnover_penalty=0.5,
        lookback=252,
    )
    assert mv.return_model == "historical"
    assert mv.risk_model == "sample_cov"
    assert mv.target_return == 0.01
    assert mv.risk_aversion == 2.0
    assert mv.turnover_penalty == 0.5
    assert mv.lookback == 252
    assert not mv._fitted


def test_mean_variance_init_validation():
    with pytest.raises(ValueError, match="risk_aversion must be >= 0"):
        MeanVarianceConstructor(risk_aversion=-1.0)

    with pytest.raises(ValueError, match="turnover_penalty must be >= 0"):
        MeanVarianceConstructor(turnover_penalty=-0.5)

    with pytest.raises(ValueError, match="lookback must be positive"):
        MeanVarianceConstructor(lookback=0)


def test_mean_variance_fit_basic(sample_prices):
    mv = MeanVarianceConstructor(lookback=252)
    result = mv.fit(sample_prices)

    assert result is mv
    assert mv._fitted
    assert mv._expected_returns is not None
    assert mv._cov_matrix is not None
    assert len(mv._expected_returns) == 3
    assert mv._cov_matrix.shape == (3, 3)


def test_mean_variance_fit_short_history():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        {
            "BTC/USDT": np.linspace(50000, 51000, 100),
            "ETH/USDT": np.linspace(3000, 3100, 100),
        },
        index=dates,
    )

    mv = MeanVarianceConstructor(lookback=252)
    with pytest.warns(UserWarning, match="shorter than lookback"):
        mv.fit(prices)

    assert mv._fitted
    assert len(mv._expected_returns) == 2


def test_mean_variance_construct_basic(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(risk_aversion=1.0)
    mv.fit(sample_prices)

    weights = mv.construct(sample_scores, sample_constraints)

    assert isinstance(weights, dict)
    assert all(isinstance(v, float) for v in weights.values())
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_mean_variance_historical_returns(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(return_model="historical", risk_aversion=1.0)
    mv.fit(sample_prices)

    expected_returns = mv._expected_returns
    assert len(expected_returns) == 3
    assert all(isinstance(r, (int, float)) for r in expected_returns.values)


def test_mean_variance_shrinkage_returns(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(return_model="shrinkage", risk_aversion=1.0)
    mv.fit(sample_prices)

    shrunk_returns = mv._expected_returns
    sample_means = sample_prices.pct_change().dropna().iloc[-252:].mean()
    grand_mean = sample_means.mean()

    for symbol in shrunk_returns.index:
        shrunk_value = shrunk_returns[symbol]
        expected_shrunk = 0.5 * sample_means[symbol] + 0.5 * grand_mean
        assert abs(shrunk_value - expected_shrunk) < 1e-10


def test_mean_variance_factor_model(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(return_model="factor_model", risk_aversion=1.0)
    mv.fit(sample_prices)

    weights = mv.construct(sample_scores, sample_constraints)

    assert "BTC/USDT" in weights
    assert weights["BTC/USDT"] > weights.get("SOL/USDT", 0)


def test_mean_variance_sample_cov(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(risk_model="sample_cov", risk_aversion=1.0)
    mv.fit(sample_prices)

    cov = mv._cov_matrix
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvals(cov) >= -1e-10)


def test_mean_variance_ledoit_wolf(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(risk_model="ledoit_wolf", risk_aversion=1.0)
    mv.fit(sample_prices)

    cov = mv._cov_matrix
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvals(cov) >= -1e-10)


def test_mean_variance_turnover_penalty(sample_prices, sample_scores, sample_constraints):
    current_weights = {
        "BTC/USDT": 0.5,
        "ETH/USDT": 0.3,
        "SOL/USDT": 0.2,
    }

    mv_no_penalty = MeanVarianceConstructor(turnover_penalty=0.0, risk_aversion=1.0)
    mv_no_penalty.fit(sample_prices)
    weights_no_penalty = mv_no_penalty.construct(sample_scores, sample_constraints, current_weights)

    mv_penalty = MeanVarianceConstructor(turnover_penalty=1.0, risk_aversion=1.0)
    mv_penalty.fit(sample_prices)
    weights_penalty = mv_penalty.construct(sample_scores, sample_constraints, current_weights)

    turnover_no_penalty = sum(
        abs(weights_no_penalty.get(s, 0) - current_weights.get(s, 0)) for s in set(weights_no_penalty) | set(current_weights)
    )
    turnover_penalty = sum(
        abs(weights_penalty.get(s, 0) - current_weights.get(s, 0)) for s in set(weights_penalty) | set(current_weights)
    )

    assert turnover_penalty <= turnover_no_penalty


def test_mean_variance_target_return(sample_prices, sample_scores, sample_constraints):
    mv = MeanVarianceConstructor(target_return=0.001, risk_aversion=1.0)
    mv.fit(sample_prices)

    weights = mv.construct(sample_scores, sample_constraints)
    expected_return = sum(weights.get(s, 0) * mv._expected_returns[s] for s in weights)

    assert expected_return >= 0.001 - 1e-6


def test_mean_variance_empty_scores(sample_prices, sample_constraints):
    mv = MeanVarianceConstructor()
    mv.fit(sample_prices)

    weights = mv.construct({}, sample_constraints)
    assert weights == {}


def test_mean_variance_not_fitted(sample_scores, sample_constraints):
    mv = MeanVarianceConstructor()

    with pytest.raises(ValueError, match="Must call fit"):
        mv.construct(sample_scores, sample_constraints)


def test_mean_variance_optimization_failure():
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    prices = pd.DataFrame(
        {
            "BTC/USDT": np.ones(300) * 50000,
            "ETH/USDT": np.ones(300) * 3000,
        },
        index=dates,
    )

    mv = MeanVarianceConstructor(target_return=1.0, risk_aversion=1.0)
    mv.fit(prices)

    scores = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    constraints = Constraints(max_position_size=0.1)

    with pytest.warns(UserWarning, match="Optimization failed"):
        weights = mv.construct(scores, constraints)
        assert weights == {}


def test_mean_variance_min_position_size(sample_prices, sample_scores, sample_constraints):
    constraints = Constraints(
        max_position_size=0.5,
        min_position_size=0.2,
    )

    mv = MeanVarianceConstructor(risk_aversion=1.0)
    mv.fit(sample_prices)

    weights = mv.construct(sample_scores, constraints)

    for weight in weights.values():
        assert weight >= 0.2 or weight == 0


def test_mean_variance_risk_aversion_effect(sample_prices, sample_scores, sample_constraints):
    mv_low = MeanVarianceConstructor(risk_aversion=0.1)
    mv_low.fit(sample_prices)
    weights_low = mv_low.construct(sample_scores, sample_constraints)

    mv_high = MeanVarianceConstructor(risk_aversion=10.0)
    mv_high.fit(sample_prices)
    weights_high = mv_high.construct(sample_scores, sample_constraints)

    var_low = sum(
        weights_low.get(i, 0) * weights_low.get(j, 0) * mv_low._cov_matrix.loc[i, j]
        for i in weights_low
        for j in weights_low
    )
    var_high = sum(
        weights_high.get(i, 0) * weights_high.get(j, 0) * mv_high._cov_matrix.loc[i, j]
        for i in weights_high
        for j in weights_high
    )

    assert var_high <= var_low * 1.1
