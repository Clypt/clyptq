"""Risk management: SL, TP, max DD."""

from datetime import datetime
from typing import Dict, List, Optional

from clypt.types import Order, OrderSide, Position


class RiskManager:
    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = 0.0

    def check_position_exits(
        self, positions: Dict[str, Position], prices: Dict[str, float]
    ) -> List[Order]:
        exit_orders = []

        for symbol, pos in positions.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            pnl_pct = (current_price - pos.avg_price) / pos.avg_price

            # Stop-loss check
            if self.stop_loss_pct and pnl_pct <= -abs(self.stop_loss_pct):
                exit_orders.append(
                    Order(symbol=symbol, side=OrderSide.SELL, amount=pos.amount)
                )
                continue

            # Take-profit check
            if self.take_profit_pct and pnl_pct >= abs(self.take_profit_pct):
                exit_orders.append(
                    Order(symbol=symbol, side=OrderSide.SELL, amount=pos.amount)
                )

        return exit_orders

    def check_max_drawdown(self, current_equity: float) -> bool:
        if not self.max_drawdown_pct:
            return False

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown >= abs(self.max_drawdown_pct):
                return True

        return False

    def reset(self):
        self.peak_equity = 0.0
