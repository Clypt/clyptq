from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def orthogonalize_factors(
    factor_scores: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    if not factor_scores:
        return {}

    symbols = set()
    for scores in factor_scores.values():
        symbols.update(scores.keys())
    symbols = sorted(symbols)

    if len(symbols) == 0:
        return {}

    factor_names = sorted(factor_scores.keys())
    if len(factor_names) == 0:
        return {}

    matrix = []
    for factor_name in factor_names:
        scores = factor_scores[factor_name]
        row = [scores.get(symbol, 0.0) for symbol in symbols]
        matrix.append(row)

    matrix = np.array(matrix)

    orthogonal = np.zeros_like(matrix)
    orthogonal[0] = matrix[0]

    for i in range(1, len(matrix)):
        orthogonal[i] = matrix[i].copy()
        for j in range(i):
            proj = np.dot(matrix[i], orthogonal[j]) / (np.dot(orthogonal[j], orthogonal[j]) + 1e-10)
            orthogonal[i] -= proj * orthogonal[j]

    result = {}
    for i, factor_name in enumerate(factor_names):
        result[factor_name] = {
            symbol: float(orthogonal[i][j])
            for j, symbol in enumerate(symbols)
        }

    return result


def pca_factors(
    factor_scores: Dict[str, Dict[str, float]],
    n_components: int = 5
) -> Dict[str, Dict[str, float]]:
    if not factor_scores:
        return {}

    symbols = set()
    for scores in factor_scores.values():
        symbols.update(scores.keys())
    symbols = sorted(symbols)

    if len(symbols) == 0:
        return {}

    factor_names = sorted(factor_scores.keys())
    if len(factor_names) == 0:
        return {}

    n_components = min(n_components, len(factor_names), len(symbols))

    df = pd.DataFrame({
        factor_name: [factor_scores[factor_name].get(symbol, 0.0) for symbol in symbols]
        for factor_name in factor_names
    })

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.values)

    result = {}
    for i in range(n_components):
        result[f"PC{i+1}"] = {
            symbol: float(components[j, i])
            for j, symbol in enumerate(symbols)
        }

    return result


def remove_correlation(
    target_factor: Dict[str, float],
    conditioning_factors: List[Dict[str, float]]
) -> Dict[str, float]:
    if not target_factor or not conditioning_factors:
        return target_factor

    symbols = set(target_factor.keys())
    for factor in conditioning_factors:
        symbols.update(factor.keys())
    symbols = sorted(symbols)

    if len(symbols) == 0:
        return {}

    y = np.array([target_factor.get(symbol, 0.0) for symbol in symbols])

    X = []
    for factor in conditioning_factors:
        X.append([factor.get(symbol, 0.0) for symbol in symbols])
    X = np.array(X).T

    if X.shape[1] == 0:
        return target_factor

    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    residual = y - y_pred

    return {symbol: float(residual[i]) for i, symbol in enumerate(symbols)}
