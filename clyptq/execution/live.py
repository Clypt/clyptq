"""
Unified CCXT executor for crypto trading.
Paper mode = simulate, Live mode = real money.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

import ccxt

from clyptq.risk.costs import apply_slippage, calculate_fee
from clyptq.execution.base import Executor
from clyptq.execution.orders.tracker import OrderTracker, TrackedOrder
from clyptq.types import CostModel, Fill, FillStatus, Order, OrderSide


class CCXTExecutor(Executor):
    """Unified executor for paper and live crypto trading."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        paper_mode: bool = True,
        sandbox: bool = False,
        timeout: int = 30000,
        cost_model: Optional[CostModel] = None,
    ):
        self.paper_mode = paper_mode
        self.cost_model = cost_model or CostModel()
        self.order_tracker = OrderTracker()

        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "enableRateLimit": True,
            "timeout": timeout,
            "options": {"defaultType": "spot"},
        }

        # only add credentials if provided (public API doesn't need them)
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            config["sandbox"] = sandbox

        self.exchange = exchange_class(config)

        self.exchange_id = exchange_id
        self.timeout = timeout
        self.exchange.load_markets()

    def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get live prices."""
        prices = {}
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                prices[symbol] = ticker["last"]
            except Exception as e:
                print(f"Price fetch failed {symbol}: {e}")
        return prices

    def _round_lot_size(self, symbol: str, amount: float) -> float:
        """Round to exchange lot size."""
        if symbol not in self.exchange.markets:
            return amount

        market = self.exchange.markets[symbol]
        lot_size = market.get("precision", {}).get("amount")

        if lot_size is None:
            return amount

        if isinstance(lot_size, int):
            return round(amount, lot_size)
        return round(amount / lot_size) * lot_size

    def _wait_for_fill(self, order_id: str, symbol: str, timeout_sec: int = 30) -> Optional[dict]:
        """Poll order until filled or timeout."""
        start = time.time()

        while time.time() - start < timeout_sec:
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                if order["status"] in ("closed", "filled", "canceled", "rejected"):
                    return order
                time.sleep(1.0)
            except Exception as e:
                print(f"Poll error: {e}")
                time.sleep(1.0)

        return None

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        if self.paper_mode:
            return self._execute_paper(orders, timestamp, prices)
        return self._execute_live(orders, timestamp)

    def _execute_paper(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """Paper mode: simulate with live prices."""
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

    def _execute_live(self, orders: List[Order], timestamp: datetime) -> List[Fill]:
        """Live mode: real orders with tracking."""
        fills = []

        for order in orders:
            tracked = self.order_tracker.create_order(order)

            try:
                rounded_amt = self._round_lot_size(order.symbol, abs(order.amount))
                if rounded_amt < 1e-8:
                    tracked.mark_rejected("Amount too small after lot size rounding")
                    continue

                side = "buy" if order.side == OrderSide.BUY else "sell"
                ccxt_order = self.exchange.create_market_order(
                    symbol=order.symbol, side=side, amount=rounded_amt
                )

                tracked.mark_submitted(ccxt_order["id"])

                filled_order = self._wait_for_fill(
                    ccxt_order["id"], order.symbol, timeout_sec=self.timeout // 1000
                )

                if filled_order is None:
                    tracked.mark_cancelled()
                    try:
                        self.exchange.cancel_order(ccxt_order["id"], order.symbol)
                    except:
                        pass
                    continue

                if filled_order["status"] in ("closed", "filled"):
                    filled_amt = filled_order.get("filled", 0.0)
                    avg_price = filled_order.get("average", 0.0)
                    fee_cost = filled_order.get("fee", {}).get("cost", 0.0)

                    fill = Fill(
                        symbol=order.symbol,
                        side=order.side,
                        amount=filled_amt,
                        price=avg_price,
                        fee=fee_cost,
                        timestamp=timestamp,
                        order_id=ccxt_order["id"],
                        status=FillStatus.FILLED
                        if filled_amt >= rounded_amt * 0.99
                        else FillStatus.PARTIAL,
                    )

                    tracked.add_fill(fill)
                    fills.append(fill)

            except ccxt.InsufficientFunds as e:
                tracked.mark_rejected(f"Insufficient funds: {e}")
                fills.append(
                    Fill(
                        symbol=order.symbol,
                        side=order.side,
                        amount=0.0,
                        price=0.0,
                        fee=0.0,
                        timestamp=timestamp,
                        status=FillStatus.REJECTED,
                    )
                )

            except Exception as e:
                tracked.mark_rejected(f"Order failed: {e}")
                fills.append(
                    Fill(
                        symbol=order.symbol,
                        side=order.side,
                        amount=0.0,
                        price=0.0,
                        fee=0.0,
                        timestamp=timestamp,
                        status=FillStatus.REJECTED,
                    )
                )

        return fills

    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance."""
        balance = self.exchange.fetch_balance()
        return balance.get(currency, {}).get("free", 0.0)

    def fetch_positions(self) -> Dict[str, Dict]:
        """Get current exchange positions.
        Returns: {symbol: {amount: float, avg_price: float}}
        """
        if self.paper_mode:
            return {}

        try:
            balance = self.exchange.fetch_balance()
            positions = {}

            for symbol, info in balance.get("total", {}).items():
                if symbol == "USDT" or info == 0:
                    continue

                amount = info
                if amount < 1e-8:
                    continue

                base_symbol = f"{symbol}/USDT"
                if base_symbol in self.exchange.markets:
                    positions[base_symbol] = {
                        "amount": amount,
                        "avg_price": 0.0,
                    }

            return positions
        except Exception as e:
            print(f"Failed to fetch positions: {e}")
            return {}

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        """Get non-terminal orders."""
        return self.order_tracker.get_pending_orders(symbol)

    def cleanup_old_orders(self, max_age_seconds: int = 86400) -> None:
        """Remove old terminal orders to save memory."""
        self.order_tracker.cleanup_old_orders(max_age_seconds)

    def close(self):
        """Close connection."""
        if hasattr(self.exchange, "close"):
            self.exchange.close()
