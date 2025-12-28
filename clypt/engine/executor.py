"""
Order execution engines for different modes.

Provides three execution modes:
1. BacktestExecutor - Deterministic execution for backtesting
2. PaperExecutor - Real-time execution without actual trades
3. LiveExecutor - Real execution via CCXT with enhancements
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import ccxt

from clypt.engine.cost_model import apply_slippage, calculate_fee
from clypt.types import CostModel, Fill, FillStatus, Order, OrderSide


class Executor(ABC):
    """Abstract base class for order executors."""

    @abstractmethod
    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """
        Execute a list of orders.

        Args:
            orders: List of orders to execute
            timestamp: Current timestamp
            prices: Dictionary of {symbol: current_price}

        Returns:
            List of fills
        """
        pass


class BacktestExecutor(Executor):
    """
    Backtest executor with deterministic fills.

    Executes all orders immediately at market price with cost model applied.
    """

    def __init__(self, cost_model: CostModel):
        """
        Initialize backtest executor.

        Args:
            cost_model: Cost model for fees and slippage
        """
        self.cost_model = cost_model

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """
        Execute orders with deterministic fills.

        Args:
            orders: List of orders to execute
            timestamp: Current timestamp
            prices: Current market prices

        Returns:
            List of fills (all orders filled)
        """
        fills = []

        for order in orders:
            if order.symbol not in prices:
                # Skip symbols without price data
                continue

            # Get market price
            market_price = prices[order.symbol]

            # Apply slippage
            exec_price = apply_slippage(market_price, order.side, self.cost_model.slippage_bps)

            # Calculate fee
            trade_value = abs(order.amount) * exec_price
            fee = calculate_fee(trade_value, order.side, self.cost_model, is_maker=False)

            # Create fill
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                amount=abs(order.amount),
                price=exec_price,
                fee=fee,
                timestamp=timestamp,
                status=FillStatus.FILLED,
            )

            fills.append(fill)

        return fills


class PaperExecutor(Executor):
    """
    Paper trading executor.

    Simulates execution using real market data without actual trades.
    Identical to backtest executor but operates in real-time.
    """

    def __init__(self, cost_model: CostModel):
        """
        Initialize paper executor.

        Args:
            cost_model: Cost model for fees and slippage
        """
        self.cost_model = cost_model

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """
        Execute orders as paper trades.

        Args:
            orders: List of orders to execute
            timestamp: Current timestamp
            prices: Current market prices

        Returns:
            List of simulated fills
        """
        fills = []

        for order in orders:
            if order.symbol not in prices:
                continue

            # Get market price
            market_price = prices[order.symbol]

            # Apply slippage
            exec_price = apply_slippage(market_price, order.side, self.cost_model.slippage_bps)

            # Calculate fee
            trade_value = abs(order.amount) * exec_price
            fee = calculate_fee(trade_value, order.side, self.cost_model, is_maker=False)

            # Create fill
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                amount=abs(order.amount),
                price=exec_price,
                fee=fee,
                timestamp=timestamp,
                status=FillStatus.FILLED,
            )

            fills.append(fill)

        return fills


class LiveExecutor(Executor):
    """
    Live executor via CCXT.

    ENHANCED FEATURES:
    1. Timeout handling for order status polling
    2. Lot size rounding for exchange compliance
    3. Partial fill handling
    4. Comprehensive error handling
    5. Rate limiting protection
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = False,
        timeout: int = 30000,
        max_retries: int = 3,
    ):
        """
        Initialize live executor with CCXT.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: API key
            api_secret: API secret
            sandbox: Use sandbox/testnet mode
            timeout: Request timeout in milliseconds
            max_retries: Maximum retries for failed requests

        Raises:
            ImportError: If exchange_id not supported by CCXT
        """
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "sandbox": sandbox,
                "enableRateLimit": True,
                "timeout": timeout,
                "options": {"defaultType": "spot"},
            }
        )

        self.exchange_id = exchange_id
        self.timeout = timeout
        self.max_retries = max_retries

        # Load markets
        self.exchange.load_markets()

    def _round_to_lot_size(self, symbol: str, amount: float) -> float:
        """
        Round amount to exchange's lot size.

        Args:
            symbol: Trading symbol
            amount: Raw amount

        Returns:
            Rounded amount

        Raises:
            ValueError: If symbol not found in markets
        """
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol {symbol} not found in exchange markets")

        market = self.exchange.markets[symbol]

        # Get lot size (precision)
        lot_size = market.get("precision", {}).get("amount", None)

        if lot_size is None:
            # No lot size info, return as-is
            return amount

        # Round to lot size
        if isinstance(lot_size, int):
            # Precision is decimal places
            return round(amount, lot_size)
        else:
            # Precision is step size
            return round(amount / lot_size) * lot_size

    def _wait_for_fill(
        self, order_id: str, symbol: str, timeout_seconds: int = 30
    ) -> Optional[dict]:
        """
        Poll order status until filled or timeout.

        Args:
            order_id: Order ID to monitor
            symbol: Trading symbol
            timeout_seconds: Timeout in seconds

        Returns:
            Order info dict or None if timeout

        Raises:
            ccxt.NetworkError: On network errors
            ccxt.ExchangeError: On exchange errors
        """
        start_time = time.time()
        poll_interval = 1.0  # Poll every second

        while time.time() - start_time < timeout_seconds:
            try:
                order = self.exchange.fetch_order(order_id, symbol)

                if order["status"] in ("closed", "filled"):
                    return order
                elif order["status"] in ("canceled", "rejected"):
                    return order

                # Wait before next poll
                time.sleep(poll_interval)

            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                # Log error but continue polling
                print(f"Error polling order {order_id}: {e}")
                time.sleep(poll_interval)

        return None  # Timeout

    def execute(
        self, orders: List[Order], timestamp: datetime, prices: Dict[str, float]
    ) -> List[Fill]:
        """
        Execute orders via CCXT exchange.

        Args:
            orders: List of orders to execute
            timestamp: Current timestamp
            prices: Current market prices (not used, fetches live)

        Returns:
            List of fills (may include partial fills)

        Raises:
            ccxt.InsufficientFunds: If insufficient balance
            ccxt.InvalidOrder: If order invalid
            ccxt.NetworkError: On network errors
        """
        fills = []

        for order in orders:
            try:
                # Round amount to lot size
                rounded_amount = self._round_to_lot_size(order.symbol, abs(order.amount))

                if rounded_amount < 1e-8:
                    # Amount too small after rounding
                    continue

                # Determine side
                side = "buy" if order.side == OrderSide.BUY else "sell"

                # Place market order
                ccxt_order = self.exchange.create_market_order(
                    symbol=order.symbol, side=side, amount=rounded_amount
                )

                # Wait for fill
                filled_order = self._wait_for_fill(
                    ccxt_order["id"], order.symbol, timeout_seconds=self.timeout // 1000
                )

                if filled_order is None:
                    # Timeout - try to cancel
                    try:
                        self.exchange.cancel_order(ccxt_order["id"], order.symbol)
                    except:
                        pass
                    continue

                # Extract fill information
                if filled_order["status"] in ("closed", "filled"):
                    filled_amount = filled_order.get("filled", 0.0)
                    avg_price = filled_order.get("average", 0.0)
                    fee_cost = filled_order.get("fee", {}).get("cost", 0.0)

                    # Create fill
                    fill = Fill(
                        symbol=order.symbol,
                        side=order.side,
                        amount=filled_amount,
                        price=avg_price,
                        fee=fee_cost,
                        timestamp=timestamp,
                        order_id=ccxt_order["id"],
                        status=FillStatus.FILLED
                        if filled_amount >= rounded_amount * 0.99
                        else FillStatus.PARTIAL,
                    )

                    fills.append(fill)

            except ccxt.InsufficientFunds as e:
                print(f"Insufficient funds for {order.symbol}: {e}")
                # Create rejected fill
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

            except (ccxt.InvalidOrder, ccxt.NetworkError, ccxt.ExchangeError) as e:
                print(f"Error executing {order.symbol}: {e}")
                # Create rejected fill
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
        """
        Get account balance for a currency.

        Args:
            currency: Currency code (e.g., 'USDT', 'USD')

        Returns:
            Available balance
        """
        balance = self.exchange.fetch_balance()
        return balance.get(currency, {}).get("free", 0.0)

    def close(self) -> None:
        """Close exchange connection."""
        if hasattr(self.exchange, "close"):
            self.exchange.close()
