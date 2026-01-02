import numpy as np
import pytest

from clyptq.trading.factors.ops.factor_combination import (
    orthogonalize_factors,
    pca_factors,
    remove_correlation,
)


def test_orthogonalize_factors_basic():
    factor_scores = {
        "momentum": {"A": 1.0, "B": 2.0, "C": 3.0},
        "trend": {"A": 1.5, "B": 2.5, "C": 3.5},
    }

    result = orthogonalize_factors(factor_scores)

    assert "momentum" in result
    assert "trend" in result
    assert len(result["momentum"]) == 3
    assert len(result["trend"]) == 3

    symbols = sorted(result["momentum"].keys())
    vec1 = np.array([result["momentum"][s] for s in symbols])
    vec2 = np.array([result["trend"][s] for s in symbols])

    dot_product = np.dot(vec1, vec2)
    assert abs(dot_product) < 1e-6


def test_orthogonalize_factors_three_factors():
    factor_scores = {
        "f1": {"A": 1.0, "B": 0.0, "C": 0.0},
        "f2": {"A": 1.0, "B": 1.0, "C": 0.0},
        "f3": {"A": 1.0, "B": 1.0, "C": 1.0},
    }

    result = orthogonalize_factors(factor_scores)

    assert len(result) == 3

    symbols = sorted(result["f1"].keys())
    v1 = np.array([result["f1"][s] for s in symbols])
    v2 = np.array([result["f2"][s] for s in symbols])
    v3 = np.array([result["f3"][s] for s in symbols])

    assert abs(np.dot(v1, v2)) < 1e-6
    assert abs(np.dot(v1, v3)) < 1e-6
    assert abs(np.dot(v2, v3)) < 1e-6


def test_orthogonalize_factors_empty():
    result = orthogonalize_factors({})
    assert result == {}


def test_pca_factors_basic():
    factor_scores = {
        "f1": {"A": 1.0, "B": 2.0, "C": 3.0},
        "f2": {"A": 2.0, "B": 4.0, "C": 6.0},
        "f3": {"A": 1.5, "B": 2.5, "C": 3.5},
    }

    result = pca_factors(factor_scores, n_components=2)

    assert "PC1" in result
    assert "PC2" in result
    assert len(result) == 2
    assert len(result["PC1"]) == 3


def test_pca_factors_n_components_capped():
    factor_scores = {
        "f1": {"A": 1.0, "B": 2.0},
        "f2": {"A": 2.0, "B": 4.0},
    }

    result = pca_factors(factor_scores, n_components=10)

    assert len(result) <= 2


def test_pca_factors_empty():
    result = pca_factors({}, n_components=3)
    assert result == {}


def test_remove_correlation_basic():
    target = {"A": 5.0, "B": 8.0, "C": 14.0}
    conditioning = [
        {"A": 1.0, "B": 2.0, "C": 3.0},
    ]

    result = remove_correlation(target, conditioning)

    assert "A" in result
    assert "B" in result
    assert "C" in result

    result_vec = np.array([result[s] for s in ["A", "B", "C"]])
    cond_vec = np.array([conditioning[0][s] for s in ["A", "B", "C"]])

    if np.std(result_vec) > 1e-6:
        corr = np.corrcoef(result_vec, cond_vec)[0, 1]
        assert abs(corr) < 0.3


def test_remove_correlation_multiple_factors():
    target = {"A": 10.0, "B": 20.0, "C": 30.0}
    conditioning = [
        {"A": 1.0, "B": 2.0, "C": 3.0},
        {"A": 2.0, "B": 1.0, "C": 0.5},
    ]

    result = remove_correlation(target, conditioning)

    assert len(result) == 3
    assert "A" in result


def test_remove_correlation_empty_conditioning():
    target = {"A": 1.0, "B": 2.0}
    conditioning = []

    result = remove_correlation(target, conditioning)

    assert result == target


def test_remove_correlation_empty_target():
    target = {}
    conditioning = [{"A": 1.0}]

    result = remove_correlation(target, conditioning)

    assert result == {}
