from typing import Dict, List

import numpy as np


class SignalQuality:
    def __init__(self, factor_scores: Dict[str, float], returns: Dict[str, float]):
        common_symbols = set(factor_scores.keys()) & set(returns.keys())
        self.symbols = list(common_symbols)
        self.scores = np.array([factor_scores[s] for s in self.symbols])
        self.returns = np.array([returns[s] for s in self.symbols])

    def information_coefficient(self) -> float:
        if len(self.symbols) < 2:
            return 0.0
        return float(np.corrcoef(self.scores, self.returns)[0, 1])

    def rank_ic(self) -> float:
        if len(self.symbols) < 2:
            return 0.0

        import pandas as pd

        rank_scores = pd.Series(self.scores).rank()
        rank_returns = pd.Series(self.returns).rank()

        return float(np.corrcoef(rank_scores, rank_returns)[0, 1])

    def information_ratio(self, periods: int = 252) -> float:
        ic = self.information_coefficient()
        if ic == 0:
            return 0.0

        return ic * np.sqrt(periods)

    def hit_rate(self) -> float:
        if len(self.symbols) == 0:
            return 0.0

        median_score = np.median(self.scores)
        median_return = np.median(self.returns)

        above_median_score = self.scores > median_score
        above_median_return = self.returns > median_return

        hits = np.sum(above_median_score == above_median_return)
        return float(hits / len(self.symbols))

    def directional_accuracy(self) -> float:
        if len(self.symbols) == 0:
            return 0.0

        score_positive = self.scores > 0
        return_positive = self.returns > 0

        correct = np.sum(score_positive == return_positive)
        return float(correct / len(self.symbols))

    def quintile_analysis(self) -> Dict:
        if len(self.symbols) < 5:
            return {}

        sorted_indices = np.argsort(self.scores)
        sorted_returns = self.returns[sorted_indices]

        n_per_quintile = len(sorted_returns) // 5
        quintiles = {}

        for q in range(5):
            start_idx = q * n_per_quintile
            end_idx = start_idx + n_per_quintile if q < 4 else len(sorted_returns)
            quintile_returns = sorted_returns[start_idx:end_idx]

            quintiles[f"Q{q+1}"] = {
                "mean_return": float(np.mean(quintile_returns)),
                "std_return": float(np.std(quintile_returns)),
                "sharpe": float(np.mean(quintile_returns) / np.std(quintile_returns))
                if np.std(quintile_returns) > 0
                else 0.0,
                "count": len(quintile_returns),
            }

        quintiles["Q5-Q1_spread"] = (
            quintiles["Q5"]["mean_return"] - quintiles["Q1"]["mean_return"]
        )

        return quintiles

    def signal_decay(self, future_returns_list: List[np.ndarray]) -> List[float]:
        ics = []
        for future_returns in future_returns_list:
            if len(future_returns) != len(self.scores):
                ics.append(0.0)
                continue

            ic = float(np.corrcoef(self.scores, future_returns)[0, 1])
            ics.append(ic)

        return ics

    def top_bottom_spread(self, top_pct: float = 0.2) -> Dict:
        if len(self.symbols) == 0:
            return {}

        n_top = max(1, int(len(self.symbols) * top_pct))

        sorted_indices = np.argsort(self.scores)
        bottom_returns = self.returns[sorted_indices[:n_top]]
        top_returns = self.returns[sorted_indices[-n_top:]]

        return {
            "top_mean": float(np.mean(top_returns)),
            "bottom_mean": float(np.mean(bottom_returns)),
            "spread": float(np.mean(top_returns) - np.mean(bottom_returns)),
            "top_std": float(np.std(top_returns)),
            "bottom_std": float(np.std(bottom_returns)),
        }
