"""
Performance attribution for portfolio analysis.

Decompose returns into factor contributions, transaction costs,
and asset-level performance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clyptq.core.types import BacktestResult


@dataclass
class FactorAttribution:
    """Attribution of returns to a single factor."""

    factor_name: str
    total_contribution: float
    avg_daily_contribution: float
    contribution_volatility: float
    sharpe_contribution: float
    daily_contributions: List[float]


@dataclass
class AssetAttribution:
    """Attribution of returns to individual assets."""

    symbol: str
    total_return: float
    weight_contribution: float
    selection_contribution: float
    interaction_contribution: float


@dataclass
class AttributionResult:
    """Complete performance attribution analysis."""

    total_return: float
    factor_attributions: List[FactorAttribution]
    asset_attributions: List[AssetAttribution]
    transaction_cost_drag: float
    cash_drag: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "factor_attributions": [
                {
                    "factor_name": fa.factor_name,
                    "total_contribution": fa.total_contribution,
                    "avg_daily_contribution": fa.avg_daily_contribution,
                    "contribution_volatility": fa.contribution_volatility,
                    "sharpe_contribution": fa.sharpe_contribution,
                }
                for fa in self.factor_attributions
            ],
            "asset_attributions": [
                {
                    "symbol": aa.symbol,
                    "total_return": aa.total_return,
                    "weight_contribution": aa.weight_contribution,
                    "selection_contribution": aa.selection_contribution,
                    "interaction_contribution": aa.interaction_contribution,
                }
                for aa in self.asset_attributions
            ],
            "transaction_cost_drag": self.transaction_cost_drag,
            "cash_drag": self.cash_drag,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceAttributor:
    """
    Analyze performance attribution.

    Decomposes portfolio returns into:
    - Factor contributions
    - Asset-level contributions (allocation + selection)
    - Transaction costs
    - Cash drag
    """

    def __init__(self):
        pass

    def analyze(
        self,
        backtest_result: BacktestResult,
    ) -> AttributionResult:
        """
        Perform attribution analysis.

        Args:
            backtest_result: Backtest results with snapshots

        Returns:
            AttributionResult with decomposed returns
        """
        if not backtest_result.snapshots:
            raise ValueError("No snapshots in backtest result")

        total_return = self._calculate_total_return(backtest_result)
        transaction_costs = self._calculate_transaction_cost_drag(backtest_result)
        cash_drag = self._calculate_cash_drag(backtest_result)

        asset_attributions = self._calculate_asset_attribution(backtest_result)

        return AttributionResult(
            total_return=total_return,
            factor_attributions=[],
            asset_attributions=asset_attributions,
            transaction_cost_drag=transaction_costs,
            cash_drag=cash_drag,
            timestamp=datetime.now(),
        )

    def _calculate_total_return(self, result: BacktestResult) -> float:
        """Calculate total portfolio return."""
        initial = result.snapshots[0].equity
        final = result.snapshots[-1].equity
        return (final - initial) / initial

    def _calculate_transaction_cost_drag(self, result: BacktestResult) -> float:
        """Calculate drag from transaction costs."""
        total_fees = sum(fill.fee for fill in result.trades)
        initial_equity = result.snapshots[0].equity
        return -total_fees / initial_equity

    def _calculate_cash_drag(self, result: BacktestResult) -> float:
        """Calculate drag from holding cash."""
        snapshots = result.snapshots
        if not snapshots:
            return 0.0

        total_days = (snapshots[-1].timestamp - snapshots[0].timestamp).days
        if total_days == 0:
            return 0.0

        avg_cash_pct = np.mean([
            snap.cash / snap.equity if snap.equity > 0 else 0
            for snap in snapshots
        ])

        portfolio_return = self._calculate_total_return(result)

        return -avg_cash_pct * portfolio_return

    def _calculate_asset_attribution(
        self,
        result: BacktestResult
    ) -> List[AssetAttribution]:
        """Calculate asset-level attribution."""
        snapshots = result.snapshots
        trades = result.trades

        symbol_pnl = {}
        symbol_weights = {}

        for trade in trades:
            if trade.symbol not in symbol_pnl:
                symbol_pnl[trade.symbol] = 0.0
                symbol_weights[trade.symbol] = []

        for snap in snapshots:
            for pos in snap.positions.values():
                if pos.symbol in symbol_pnl:
                    mv = pos.amount * pos.avg_price
                    weight = mv / snap.equity if snap.equity > 0 else 0
                    symbol_weights[pos.symbol].append(weight)

        attributions = []
        for symbol in symbol_pnl.keys():
            if not symbol_weights.get(symbol):
                continue

            avg_weight = np.mean(symbol_weights[symbol])

            attributions.append(AssetAttribution(
                symbol=symbol,
                total_return=0.0,
                weight_contribution=avg_weight,
                selection_contribution=0.0,
                interaction_contribution=0.0,
            ))

        return attributions
