"""Trading costs: fees + slippage."""

from typing import Dict, List

from clypt.types import CostModel, Fill, Order, OrderSide, OrderType


def estimate_fill_cost(
    order: Order, price: float, cost_model: CostModel, is_maker: bool = False
) -> float:
    trade_value = abs(order.amount) * price
    fee_rate = cost_model.maker_fee if is_maker else cost_model.taker_fee
    fee = trade_value * fee_rate
    slippage_rate = cost_model.slippage_bps / 10000.0
    slippage_cost = trade_value * slippage_rate
    return fee + slippage_cost


def apply_slippage(price: float, side: OrderSide, slippage_bps: float) -> float:
    slippage_rate = slippage_bps / 10000.0
    if side == OrderSide.BUY:
        return price * (1 + slippage_rate)
    else:
        return price * (1 - slippage_rate)


def calculate_fee(
    trade_value: float, side: OrderSide, cost_model: CostModel, is_maker: bool = False
) -> float:
    fee_rate = cost_model.maker_fee if is_maker else cost_model.taker_fee
    return abs(trade_value) * fee_rate


def estimate_transaction_costs(
    orders: List[Order], prices: Dict[str, float], cost_model: CostModel
) -> float:
    total_cost = 0.0

    for order in orders:
        if order.symbol not in prices:
            continue

        price = prices[order.symbol]
        exec_price = apply_slippage(price, order.side, cost_model.slippage_bps)
        trade_value = abs(order.amount) * exec_price
        fee = calculate_fee(trade_value, order.side, cost_model, is_maker=False)
        slippage_cost = abs(order.amount) * abs(exec_price - price)
        total_cost += fee + slippage_cost

    return total_cost


def create_fill_from_order(
    order: Order, price: float, cost_model: CostModel, timestamp, is_maker: bool = False
) -> Fill:
    exec_price = apply_slippage(price, order.side, cost_model.slippage_bps)
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
    turnover = calculate_turnover(current_weights, target_weights, equity)
    avg_fee = (cost_model.maker_fee + cost_model.taker_fee) / 2
    slippage_rate = cost_model.slippage_bps / 10000.0
    cost_rate = avg_fee + slippage_rate
    return turnover * cost_rate
