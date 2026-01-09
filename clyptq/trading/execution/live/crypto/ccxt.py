"""
CCXT executor for cryptocurrency exchanges.

Supports:
- Paper mode: Simulated execution with live prices
- Live mode: Real order execution

Exchanges: Binance, Coinbase, Kraken, etc. (any CCXT-supported exchange)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

import ccxt

from clyptq.trading.execution.live.base import LiveExecutor
from clyptq.core.types import CostModel, Fill, FillStatus, Order, OrderSide
from clyptq.trading.execution.order_tracker import OrderTracker, TrackedOrder
from clyptq.trading.execution.costs import apply_slippage, calculate_fee
from clyptq.infra.security import SecretsManager
from clyptq.infra.utils import get_logger, RateLimiter


class CCXTExecutor(LiveExecutor):
    """Executor for cryptocurrency exchanges via CCXT.

    Responsibilities (execution only):
    - Execute market orders (paper or live)
    - Track order status
    - Fetch account balance and positions

    NOT responsible for (use DataProvider + Collector instead):
    - Fetching price data
    - Fetching historical OHLCV

    Example:
        ```python
        executor = CCXTExecutor(
            exchange_id="binance",
            api_key="...",
            api_secret="...",
            paper_mode=True,
        )

        fills = executor.execute(orders, timestamp, prices)
        ```
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        paper_mode: bool = True,
        sandbox: bool = False,
        timeout: int = 30000,
        cost_model: Optional[CostModel] = None,
        secrets_manager: Optional[SecretsManager] = None,
        credentials_id: Optional[str] = None,
        rate_limit_requests: Optional[int] = None,
    ):
        self.paper_mode = paper_mode
        self.cost_model = cost_model or CostModel()
        self.order_tracker = OrderTracker()
        self._exchange_id = exchange_id
        self.timeout = timeout
        self._supported_symbols: Optional[List[str]] = None

        self.logger = get_logger(
            __name__,
            context={
                "exchange": exchange_id,
                "mode": "paper" if paper_mode else "live",
                "asset_class": "crypto",
            },
        )

        self.rate_limiter: Optional[RateLimiter] = None
        if rate_limit_requests:
            self.rate_limiter = RateLimiter(max_requests=rate_limit_requests, window=60.0)
            self.logger.info(
                "Rate limiter enabled",
                extra={"max_requests": rate_limit_requests, "window": 60.0},
            )

        # Load credentials from secrets manager if provided
        if secrets_manager and credentials_id:
            credentials = secrets_manager.get_dict(credentials_id)
            if credentials:
                api_key = credentials.get("api_key", api_key)
                api_secret = credentials.get("api_secret", api_secret)
                self.logger.info("Loaded credentials from secrets manager")

        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "enableRateLimit": True,
            "timeout": timeout,
            "options": {"defaultType": "spot"},
        }

        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            config["sandbox"] = sandbox

        self.exchange = exchange_class(config)

        try:
            self.exchange.load_markets()
            self.logger.info("Exchange markets loaded successfully")
        except Exception as e:
            self.logger.error("Failed to load exchange markets", extra={"error": str(e)})
            raise

    @property
    def asset_class(self) -> str:
        """Return asset class identifier."""
        return "crypto"

    @property
    def exchange_id(self) -> str:
        """Return exchange identifier."""
        return self._exchange_id

    def get_supported_symbols(self) -> List[str]:
        """Get list of tradeable symbols on this exchange."""
        if self._supported_symbols is None:
            self._supported_symbols = list(self.exchange.markets.keys())
        return self._supported_symbols

    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance in specified currency."""
        balance = self.exchange.fetch_balance()
        return balance.get(currency, {}).get("free", 0.0)

    def fetch_positions(self) -> Dict[str, Dict]:
        """Get current exchange positions.

        Returns:
            {symbol: {"amount": float, "avg_price": float}}
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

            self.logger.info("Positions fetched", extra={"count": len(positions)})
            return positions

        except ccxt.NetworkError as e:
            self.logger.error("Network error fetching positions", extra={"error": str(e)})
            return {}
        except ccxt.ExchangeError as e:
            self.logger.error("Exchange error fetching positions", extra={"error": str(e)})
            return {}
        except Exception as e:
            self.logger.error("Unexpected error fetching positions", extra={"error": str(e)})
            return {}

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """Execute orders and return fills."""
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
            except ccxt.NetworkError as e:
                self.logger.warning(
                    "Network error polling order",
                    extra={"order_id": order_id, "symbol": symbol, "error": str(e)},
                )
                time.sleep(1.0)
            except ccxt.ExchangeError as e:
                self.logger.warning(
                    "Exchange error polling order",
                    extra={"order_id": order_id, "symbol": symbol, "error": str(e)},
                )
                time.sleep(1.0)
            except Exception as e:
                self.logger.error(
                    "Unexpected error polling order",
                    extra={"order_id": order_id, "symbol": symbol, "error": str(e)},
                )
                time.sleep(1.0)

        self.logger.warning(
            "Order polling timeout",
            extra={"order_id": order_id, "symbol": symbol, "timeout_sec": timeout_sec},
        )
        return None

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        """Get non-terminal orders."""
        return self.order_tracker.get_pending_orders(symbol)

    def cleanup_old_orders(self, max_age_seconds: int = 86400) -> None:
        """Remove old terminal orders to save memory."""
        self.order_tracker.cleanup_old_orders(max_age_seconds)

    def close(self) -> None:
        """Close connection."""
        if hasattr(self.exchange, "close"):
            self.exchange.close()
