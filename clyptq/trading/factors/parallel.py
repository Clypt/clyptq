from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Literal

import pandas as pd

from clyptq.core.base import Factor


def _compute_factor_wrapper(
    factor: Factor,
    current_prices: pd.Series,
    history: pd.DataFrame,
    timestamp: datetime,
) -> Dict[str, float]:
    return factor.compute(current_prices, history, timestamp)


class ParallelFactorComputer:
    def __init__(
        self,
        max_workers: int | None = None,
        executor_type: Literal["thread", "process"] = "thread",
    ):
        self.max_workers = max_workers
        self.executor_type = executor_type

    def compute_factors(
        self,
        factors: List[Factor],
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> List[Dict[str, float]]:
        if len(factors) == 1:
            return [factors[0].compute(current_prices, history, timestamp)]

        executor_class = (
            ThreadPoolExecutor if self.executor_type == "thread" else ProcessPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            for factor in factors:
                future = executor.submit(
                    _compute_factor_wrapper,
                    factor,
                    current_prices,
                    history,
                    timestamp,
                )
                futures.append(future)

            results = [future.result() for future in futures]

        return results

    def compute_factors_sequential(
        self,
        factors: List[Factor],
        current_prices: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime,
    ) -> List[Dict[str, float]]:
        return [
            factor.compute(current_prices, history, timestamp) for factor in factors
        ]
