from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from clyptq.core.base import Factor
from clyptq.data.stores.store import DataStore


class FactorAnalyzer:
    def information_coefficient(
        self, factor_scores: Dict[str, float], returns: Dict[str, float]
    ) -> float:
        common_symbols = set(factor_scores.keys()) & set(returns.keys())
        if len(common_symbols) < 2:
            return 0.0

        scores = [factor_scores[s] for s in common_symbols]
        rets = [returns[s] for s in common_symbols]

        return float(np.corrcoef(scores, rets)[0, 1])

    def rank_ic(
        self, factor_scores: Dict[str, float], returns: Dict[str, float]
    ) -> float:
        common_symbols = set(factor_scores.keys()) & set(returns.keys())
        if len(common_symbols) < 2:
            return 0.0

        scores = np.array([factor_scores[s] for s in common_symbols])
        rets = np.array([returns[s] for s in common_symbols])

        rank_scores = pd.Series(scores).rank()
        rank_rets = pd.Series(rets).rank()

        return float(np.corrcoef(rank_scores, rank_rets)[0, 1])

    def factor_correlation(
        self, factors: List[Factor], data: DataStore
    ) -> pd.DataFrame:
        factor_scores = {}
        for factor in factors:
            view = data.get_view(data.get_all_timestamps()[-1])
            scores = factor.compute(view)
            factor_scores[factor.__class__.__name__] = scores

        all_symbols = set()
        for scores in factor_scores.values():
            all_symbols.update(scores.keys())

        factor_matrix = {}
        for factor_name, scores in factor_scores.items():
            factor_matrix[factor_name] = [
                scores.get(symbol, np.nan) for symbol in all_symbols
            ]

        df = pd.DataFrame(factor_matrix)
        return df.corr()

    def turnover_analysis(
        self, factor: Factor, data: DataStore, top_n: int = 10
    ) -> Dict:
        timestamps = data.get_all_timestamps()
        if len(timestamps) < 2:
            return {"mean_turnover": 0.0, "std_turnover": 0.0, "turnovers": []}

        turnovers = []
        prev_top = set()

        for timestamp in timestamps:
            view = data.get_view(timestamp)
            scores = factor.compute(view)

            if not scores:
                continue

            sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            current_top = set([s for s, _ in sorted_symbols[:top_n]])

            if prev_top:
                added = len(current_top - prev_top)
                removed = len(prev_top - current_top)
                turnover = (added + removed) / (2 * top_n) if top_n > 0 else 0.0
                turnovers.append(turnover)

            prev_top = current_top

        return {
            "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
            "std_turnover": float(np.std(turnovers)) if turnovers else 0.0,
            "turnovers": turnovers,
        }

    def factor_decay(
        self, factor_scores: Dict[str, float], future_returns: List[Dict[str, float]]
    ) -> List[float]:
        ics = []
        for returns in future_returns:
            ic = self.information_coefficient(factor_scores, returns)
            ics.append(ic)

        return ics

    def quintile_analysis(
        self, factor_scores: Dict[str, float], returns: Dict[str, float]
    ) -> Dict:
        common_symbols = set(factor_scores.keys()) & set(returns.keys())
        if len(common_symbols) < 5:
            return {}

        data = [(factor_scores[s], returns[s]) for s in common_symbols]
        data.sort(key=lambda x: x[0])

        n_per_quintile = len(data) // 5
        quintiles = {}

        for q in range(5):
            start_idx = q * n_per_quintile
            end_idx = start_idx + n_per_quintile if q < 4 else len(data)
            quintile_returns = [r for _, r in data[start_idx:end_idx]]

            quintiles[f"Q{q+1}"] = {
                "mean_return": float(np.mean(quintile_returns)),
                "std_return": float(np.std(quintile_returns)),
                "count": len(quintile_returns),
            }

        quintiles["spread"] = quintiles["Q5"]["mean_return"] - quintiles["Q1"]["mean_return"]

        return quintiles

    def ic_decay_analysis(
        self, factor: Factor, data: DataStore, max_horizon: int = 20
    ) -> pd.DataFrame:
        symbols = data.symbols()
        if not symbols:
            return pd.DataFrame()

        first_symbol = symbols[0]
        timestamps = data._data[first_symbol].index.to_pydatetime().tolist()

        if len(timestamps) < max_horizon + 2:
            return pd.DataFrame()

        results = {f"day_{i+1}": [] for i in range(max_horizon)}

        for i in range(len(timestamps) - max_horizon):
            current_ts = timestamps[i]
            view = data.get_view(current_ts)

            try:
                scores = factor.compute(view)
            except (ValueError, KeyError):
                continue

            if not scores:
                continue

            for horizon in range(1, max_horizon + 1):
                future_ts = timestamps[i + horizon]
                future_returns = self._calculate_forward_returns(
                    data, current_ts, future_ts, scores.keys()
                )

                if future_returns:
                    ic = self.information_coefficient(scores, future_returns)
                    results[f"day_{horizon}"].append(ic)

        decay_df = pd.DataFrame({
            "horizon": list(range(1, max_horizon + 1)),
            "mean_ic": [np.mean(results[f"day_{i+1}"]) if results[f"day_{i+1}"] else 0.0
                       for i in range(max_horizon)],
            "std_ic": [np.std(results[f"day_{i+1}"]) if results[f"day_{i+1}"] else 0.0
                      for i in range(max_horizon)],
            "abs_mean_ic": [np.mean(np.abs(results[f"day_{i+1}"])) if results[f"day_{i+1}"] else 0.0
                           for i in range(max_horizon)]
        })

        return decay_df

    def _calculate_forward_returns(
        self, data: DataStore, start_ts: datetime, end_ts: datetime, symbols: List[str]
    ) -> Dict[str, float]:
        returns = {}
        for symbol in symbols:
            try:
                start_view = data.get_view(start_ts)
                end_view = data.get_view(end_ts)

                start_price = start_view.close(symbol, 1)[0]
                end_price = end_view.close(symbol, 1)[0]

                returns[symbol] = (end_price - start_price) / start_price
            except (KeyError, ValueError, IndexError):
                continue

        return returns
