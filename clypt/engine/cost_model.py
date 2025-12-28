"""
Trading cost model for realistic simulation.

Estimates trading costs including:
- Exchange fees (maker/taker)
- Slippage
- Impact costs (optional)
"""

from typing import Dict, List

from clypt.types import CostModel, Fill, Order, OrderSide, OrderType


def estimate_fill_cost(
    order: Order, price: float, cost_model: CostModel, is_maker: bool = False
) -> float:
    """
    Estimate total cost of executing an order.

    Args:
        order: Order to execute
        price: Execution price
        cost_model: Cost model parameters
        is_maker: If True, use maker fee; otherwise use taker fee

    Returns:
        Total cost in quote currency (fee + slippage)
    """
    # Base trade value
    trade_value = abs(order.amount) * price

    # Fee
    fee_rate = cost_model.maker_fee if is_maker else cost_model.taker_fee
    fee = trade_value * fee_rate

    # Slippage (in basis points)
    slippage_rate = cost_model.slippage_bps / 10000.0  # Convert bps to decimal
    slippage_cost = trade_value * slippage_rate

    return fee + slippage_cost


def apply_slippage(price: float, side: OrderSide, slippage_bps: float) -> float:
    """
    Apply slippage to execution price.

    Args:
        price: Base price
        side: Order side (BUY or SELL)
        slippage_bps: Slippage in basis points

    Returns:
        Price after slippage
    """
    slippage_rate = slippage_bps / 10000.0

    if side == OrderSide.BUY:
        # Buys execute at higher price
        return price * (1 + slippage_rate)
    else:
        # Sells execute at lower price
        return price * (1 - slippage_rate)


def calculate_fee(
    trade_value: float, side: OrderSide, cost_model: CostModel, is_maker: bool = False
) -> float:
    """
    Calculate trading fee.

    Args:
        trade_value: Value of trade in quote currency
        side: Order side
        cost_model: Cost model parameters
        is_maker: If True, use maker fee; otherwise use taker fee

    Returns:
        Fee amount in quote currency
    """
    fee_rate = cost_model.maker_fee if is_maker else cost_model.taker_fee
    return abs(trade_value) * fee_rate


def estimate_transaction_costs(
    orders: List[Order], prices: Dict[str, float], cost_model: CostModel
) -> float:
    """
    Estimate total transaction costs for a list of orders.

    Args:
        orders: List of orders to execute
        prices: Dictionary of {symbol: price}
        cost_model: Cost model parameters

    Returns:
        Total estimated cost in quote currency
    """
    total_cost = 0.0

    for order in orders:
        if order.symbol not in prices:
            continue

        price = prices[order.symbol]

        # Apply slippage
        exec_price = apply_slippage(price, order.side, cost_model.slippage_bps)

        # Calculate trade value and fee
        trade_value = abs(order.amount) * exec_price
        fee = calculate_fee(trade_value, order.side, cost_model, is_maker=False)

        # Slippage cost
        slippage_cost = abs(order.amount) * abs(exec_price - price)

        total_cost += fee + slippage_cost

    return total_cost


def create_fill_from_order(
    order: Order, price: float, cost_model: CostModel, timestamp, is_maker: bool = False
) -> Fill:
    """
    Create a Fill from an Order with realistic pricing.

    Args:
        order: Original order
        price: Market price
        cost_model: Cost model parameters
        timestamp: Execution timestamp
        is_maker: If True, use maker fee

    Returns:
        Fill with execution details
    """
    # Apply slippage
    exec_price = apply_slippage(price, order.side, cost_model.slippage_bps)

    # Calculate fee
    trade_value = abs(order.amount) * exec_price
    fee = calculate_fee(trade_value, order.side, cost_model, is_maker)

    return Fill(
        symbol=order.symbol,
        side=order.side,
        amount=abs(order.amount),
        price=exec_price,
        fee=fee,
        timestamp=timestamp,
    )


def calculate_turnover(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
) -> float:
    """
    Calculate portfolio turnover.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        equity: Total portfolio value

    Returns:
        Turnover in quote currency (sum of absolute weight changes)
    """
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())

    turnover = 0.0
    for symbol in all_symbols:
        current = current_weights.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        turnover += abs(target - current)

    return turnover * equity


def estimate_rebalance_cost(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
    cost_model: CostModel,
) -> float:
    """
    Estimate cost of rebalancing portfolio.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        equity: Total portfolio value
        cost_model: Cost model parameters

    Returns:
        Estimated rebalancing cost in quote currency
    """
    turnover = calculate_turnover(current_weights, target_weights, equity)

    # Average fee
    avg_fee = (cost_model.maker_fee + cost_model.taker_fee) / 2

    # Slippage
    slippage_rate = cost_model.slippage_bps / 10000.0

    # Total cost
    cost_rate = avg_fee + slippage_rate
    return turnover * cost_rate
