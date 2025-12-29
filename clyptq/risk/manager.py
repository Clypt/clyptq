"""Risk management: SL, TP, max DD."""

from datetime import datetime
from typing import Dict, List, Optional

from clyptq.types import Order, OrderSide, Position


class RiskManager:
    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        max_position_pct: Optional[float] = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
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

    def apply_position_limits(
        self,
        orders: List[Order],
        current_positions: Dict[str, Position],
        prices: Dict[str, float],
        equity: float,
    ) -> List[Order]:
        """Adjust orders to respect position size limits."""
        if not self.max_position_pct:
            return orders

        max_position_value = equity * abs(self.max_position_pct)
        adjusted_orders = []

        for order in orders:
            if order.symbol not in prices:
                continue

            price = prices[order.symbol]
            current_pos = current_positions.get(order.symbol)
            current_value = current_pos.amount * price if current_pos else 0.0

            if order.side == OrderSide.BUY:
                new_value = current_value + (order.amount * price)
                if new_value > max_position_value:
                    allowed_value = max(0, max_position_value - current_value)
                    allowed_amount = allowed_value / price if price > 0 else 0.0
                    if allowed_amount > 1e-8:
                        adjusted_orders.append(
                            Order(symbol=order.symbol, side=order.side, amount=allowed_amount)
                        )
                else:
                    adjusted_orders.append(order)
            else:
                adjusted_orders.append(order)

        return adjusted_orders

    def reset(self):
        self.peak_equity = 0.0
