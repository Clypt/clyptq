"""Backtest executor with deterministic fills."""

from datetime import datetime
from typing import Dict, List

from clyptq.risk.costs import apply_slippage, calculate_fee
from clyptq.execution.base import Executor
from clyptq.core.types import CostModel, Fill, FillStatus, Order


class BacktestExecutor(Executor):
    """Deterministic execution for backtesting."""

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        fills = []

        for order in orders:
            if order.symbol not in prices:
                continue

            market_price = prices[order.symbol]
            exec_price = apply_slippage(market_price, order.side, self.cost_model.slippage_bps)
            trade_value = abs(order.amount) * exec_price
            fee = calculate_fee(trade_value, order.side, self.cost_model, is_maker=False)

            fills.append(
                Fill(
                    symbol=order.symbol,
                    side=order.side,
                    amount=abs(order.amount),
                    price=exec_price,
                    fee=fee,
                    timestamp=timestamp,
                    status=FillStatus.FILLED,
                )
            )

        return fills
