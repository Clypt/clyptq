from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clyptq.core.types import Constraints
from clyptq.trading.portfolio.risk_budget import RiskBudgetConstructor


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


def test_risk_budget_init():
    rb = RiskBudgetConstructor(
        risk_model="sample_cov",
        risk_budgets={"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2},
        lookback=252,
    )
    assert rb.risk_model == "sample_cov"
    assert rb.risk_budgets == {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
    assert rb.lookback == 252
    assert not rb._fitted


def test_risk_budget_init_validation():
    with pytest.raises(ValueError, match="lookback must be positive"):
        RiskBudgetConstructor(lookback=0)

    with pytest.raises(ValueError, match="risk_budgets must sum to 1.0"):
        RiskBudgetConstructor(risk_budgets={"BTC/USDT": 0.6, "ETH/USDT": 0.6})


def test_risk_budget_fit_basic(sample_prices):
    rb = RiskBudgetConstructor(lookback=252)
    result = rb.fit(sample_prices)

    assert result is rb
    assert rb._fitted
    assert rb._cov_matrix is not None
    assert rb._cov_matrix.shape == (3, 3)


def test_risk_budget_fit_short_history():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        {
            "BTC/USDT": np.linspace(50000, 51000, 100),
            "ETH/USDT": np.linspace(3000, 3100, 100),
        },
        index=dates,
    )

    rb = RiskBudgetConstructor(lookback=252)
    with pytest.warns(UserWarning, match="shorter than lookback"):
        rb.fit(prices)

    assert rb._fitted
    assert rb._cov_matrix.shape == (2, 2)


def test_risk_budget_construct_erc(sample_prices, sample_scores, sample_constraints):
    rb = RiskBudgetConstructor()
    rb.fit(sample_prices)

    weights = rb.construct(sample_scores, sample_constraints)

    assert isinstance(weights, dict)
    assert all(isinstance(v, float) for v in weights.values())
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 0.01

    risk_contrib = rb.get_risk_contributions(weights)
    contributions = list(risk_contrib.values())
    for i in range(len(contributions) - 1):
        assert abs(contributions[i] - contributions[i + 1]) < 0.15


def test_risk_budget_construct_custom_budgets(sample_prices, sample_scores, sample_constraints):
    custom_budgets = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
    rb = RiskBudgetConstructor(risk_budgets=custom_budgets)
    rb.fit(sample_prices)

    weights = rb.construct(sample_scores, sample_constraints)

    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1.0) < 0.01

    risk_contrib = rb.get_risk_contributions(weights)
    assert abs(risk_contrib["BTC/USDT"] - 0.5) < 0.15
    assert abs(risk_contrib["ETH/USDT"] - 0.3) < 0.15
    assert abs(risk_contrib["SOL/USDT"] - 0.2) < 0.15


def test_risk_budget_sample_cov(sample_prices, sample_scores, sample_constraints):
    rb = RiskBudgetConstructor(risk_model="sample_cov")
    rb.fit(sample_prices)

    cov = rb._cov_matrix
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvals(cov) >= -1e-10)


def test_risk_budget_ledoit_wolf(sample_prices, sample_scores, sample_constraints):
    rb = RiskBudgetConstructor(risk_model="ledoit_wolf")
    rb.fit(sample_prices)

    cov = rb._cov_matrix
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvals(cov) >= -1e-10)


def test_risk_budget_get_risk_contributions(sample_prices):
    rb = RiskBudgetConstructor()
    rb.fit(sample_prices)

    weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
    risk_contrib = rb.get_risk_contributions(weights)

    assert len(risk_contrib) == 3
    assert abs(sum(risk_contrib.values()) - 1.0) < 1e-6
    assert all(rc >= 0 for rc in risk_contrib.values())


def test_risk_budget_empty_scores(sample_prices, sample_constraints):
    rb = RiskBudgetConstructor()
    rb.fit(sample_prices)

    weights = rb.construct({}, sample_constraints)
    assert weights == {}


def test_risk_budget_not_fitted(sample_scores, sample_constraints):
    rb = RiskBudgetConstructor()

    with pytest.raises(ValueError, match="Must call fit"):
        rb.construct(sample_scores, sample_constraints)


def test_risk_budget_not_fitted_risk_contributions():
    rb = RiskBudgetConstructor()

    with pytest.raises(ValueError, match="Must call fit"):
        rb.get_risk_contributions({"BTC/USDT": 0.5, "ETH/USDT": 0.5})


def test_risk_budget_min_position_size(sample_prices, sample_scores):
    constraints = Constraints(
        max_position_size=0.5,
        min_position_size=0.2,
    )

    rb = RiskBudgetConstructor()
    rb.fit(sample_prices)

    weights = rb.construct(sample_scores, constraints)

    for weight in weights.values():
        assert weight >= 0.2 or weight == 0


def test_risk_budget_erc_equal_contributions(sample_prices, sample_scores, sample_constraints):
    rb = RiskBudgetConstructor()
    rb.fit(sample_prices)

    weights = rb.construct(sample_scores, sample_constraints)
    risk_contrib = rb.get_risk_contributions(weights)

    contributions = list(risk_contrib.values())
    mean_contrib = np.mean(contributions)

    for contrib in contributions:
        assert abs(contrib - mean_contrib) < 0.2


def test_risk_budget_optimization_failure():
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    prices = pd.DataFrame(
        {
            "BTC/USDT": np.ones(300) * 50000,
            "ETH/USDT": np.ones(300) * 3000,
        },
        index=dates,
    )

    rb = RiskBudgetConstructor()
    rb.fit(prices)

    scores = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    constraints = Constraints(max_position_size=0.1)

    weights = rb.construct(scores, constraints)
    assert isinstance(weights, dict)


def test_risk_budget_zero_variance():
    rb = RiskBudgetConstructor()
    rb._fitted = True
    rb._cov_matrix = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]],
        index=["BTC/USDT", "ETH/USDT"],
        columns=["BTC/USDT", "ETH/USDT"],
    )

    weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
    risk_contrib = rb.get_risk_contributions(weights)

    assert risk_contrib == {"BTC/USDT": 0.0, "ETH/USDT": 0.0}


def test_risk_budget_lookback_window(sample_prices, sample_scores, sample_constraints):
    rb_short = RiskBudgetConstructor(lookback=60)
    rb_short.fit(sample_prices)
    weights_short = rb_short.construct(sample_scores, sample_constraints)

    rb_long = RiskBudgetConstructor(lookback=252)
    rb_long.fit(sample_prices)
    weights_long = rb_long.construct(sample_scores, sample_constraints)

    symbols = sorted(weights_short.keys())
    for symbol in symbols:
        if symbol in weights_long:
            weight_diff = abs(weights_short[symbol] - weights_long[symbol])
            if weight_diff > 0.01:
                return
    assert False, "Weights should differ with different lookback periods"
