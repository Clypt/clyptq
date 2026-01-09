"""Portfolio position management utilities.

Standard target-position-based rebalancing workflow:
1. Compute alpha scores
2. Apply transforms (demean, neutralize, etc.)
3. L1 normalize to get weights: sum(|w|) = 1
4. Multiply by book size to get target positions: V_target = w * book_size
5. Compute delta from current positions: delta = V_target - V_current
6. Execute orders for delta amounts

Example:
    >>> from clyptq.operator import demean, l1_norm
    >>> from clyptq.trading.execution.portfolio import (
    ...     compute_target_positions,
    ...     compute_order_deltas,
    ... )
    >>>
    >>> # Alpha -> Weights -> Target Positions
    >>> alpha = demean(raw_scores)
    >>> weights = l1_norm(alpha)
    >>> target = compute_target_positions(weights, book_size=100_000)
    >>>
    >>> # Current -> Delta -> Orders
    >>> current = {"BTC/USDT": 5000, "ETH/USDT": 3000}
    >>> orders = compute_order_deltas(target, current)
    >>> # orders = {"BTC/USDT": +2000, "ETH/USDT": -1000, ...}
"""

from typing import Dict, Optional, Union
import pandas as pd


def compute_target_positions(
    weights: Union[pd.Series, Dict[str, float]],
    book_size: float,
    min_position: float = 0.0,
) -> Dict[str, float]:
    """Convert L1-normalized weights to target notional positions.

    Args:
        weights: Portfolio weights (should sum to ~1 in absolute value)
        book_size: Total capital to allocate (USD or base currency)
        min_position: Minimum absolute position size (filter small positions)

    Returns:
        Dict of {symbol: target_notional_value}

    Example:
        >>> weights = {"BTC": 0.4, "ETH": 0.3, "SOL": -0.3}
        >>> target = compute_target_positions(weights, book_size=100_000)
        >>> # {"BTC": 40000, "ETH": 30000, "SOL": -30000}
    """
    if isinstance(weights, pd.Series):
        weights = weights.to_dict()

    targets = {}
    for symbol, w in weights.items():
        if pd.isna(w):
            continue
        position = w * book_size
        if abs(position) >= min_position:
            targets[symbol] = position

    return targets


def compute_order_deltas(
    target_positions: Dict[str, float],
    current_positions: Dict[str, float],
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Compute order amounts as delta between target and current.

    This is the standard rebalancing formula:
        order_amount = target - current

    Positive = buy/increase position
    Negative = sell/decrease position

    Args:
        target_positions: {symbol: target_notional}
        current_positions: {symbol: current_notional}
        tolerance: Minimum delta to generate an order

    Returns:
        Dict of {symbol: order_amount}

    Examples:
        Case 1 (add to long): current=5, target=10 -> order=+5
        Case 2 (reduce long): current=5, target=3 -> order=-2
        Case 3 (flip to short): current=5, target=-3 -> order=-8

    Example:
        >>> current = {"BTC": 5000, "ETH": 3000}
        >>> target = {"BTC": 7000, "ETH": 2000, "SOL": 1000}
        >>> orders = compute_order_deltas(target, current)
        >>> # {"BTC": 2000, "ETH": -1000, "SOL": 1000}
    """
    all_symbols = set(target_positions.keys()) | set(current_positions.keys())
    orders = {}

    for symbol in all_symbols:
        target = target_positions.get(symbol, 0.0)
        current = current_positions.get(symbol, 0.0)
        delta = target - current

        if abs(delta) > tolerance:
            orders[symbol] = delta

    return orders


def notional_to_quantity(
    notional: float,
    price: float,
    lot_size: float = 1e-8,
) -> float:
    """Convert notional value to quantity.

    Args:
        notional: Target notional value in quote currency
        price: Current price
        lot_size: Minimum quantity increment

    Returns:
        Quantity rounded to lot size
    """
    if price <= 0:
        return 0.0
    qty = notional / price
    return round(qty / lot_size) * lot_size


def quantity_to_notional(
    quantity: float,
    price: float,
) -> float:
    """Convert quantity to notional value.

    Args:
        quantity: Position quantity
        price: Current price

    Returns:
        Notional value in quote currency
    """
    return quantity * price


def compute_turnover(
    target_weights: Union[pd.Series, Dict[str, float]],
    current_weights: Union[pd.Series, Dict[str, float]],
) -> float:
    """Compute portfolio turnover as sum of absolute weight changes.

    Turnover = 0.5 * sum(|w_target - w_current|)

    The 0.5 factor accounts for buys matching sells.

    Args:
        target_weights: New portfolio weights
        current_weights: Current portfolio weights

    Returns:
        Turnover fraction (0 = no change, 1 = full turnover)
    """
    if isinstance(target_weights, pd.Series):
        target_weights = target_weights.to_dict()
    if isinstance(current_weights, pd.Series):
        current_weights = current_weights.to_dict()

    all_symbols = set(target_weights.keys()) | set(current_weights.keys())
    total_change = 0.0

    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        if pd.isna(target):
            target = 0.0
        if pd.isna(current):
            current = 0.0
        total_change += abs(target - current)

    return total_change / 2


def apply_turnover_constraint(
    target_weights: Union[pd.Series, Dict[str, float]],
    current_weights: Union[pd.Series, Dict[str, float]],
    max_turnover: float,
) -> Dict[str, float]:
    """Constrain target weights to maximum turnover.

    If turnover exceeds max_turnover, linearly interpolate between
    current and target weights.

    Args:
        target_weights: Desired portfolio weights
        current_weights: Current portfolio weights
        max_turnover: Maximum allowed turnover (0-1)

    Returns:
        Constrained target weights
    """
    if isinstance(target_weights, pd.Series):
        target_weights = target_weights.to_dict()
    if isinstance(current_weights, pd.Series):
        current_weights = current_weights.to_dict()

    turnover = compute_turnover(target_weights, current_weights)

    if turnover <= max_turnover:
        return dict(target_weights)

    # Scale down the change
    scale = max_turnover / turnover
    result = {}
    all_symbols = set(target_weights.keys()) | set(current_weights.keys())

    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        if pd.isna(target):
            target = 0.0
        if pd.isna(current):
            current = 0.0
        # Interpolate: current + scale * (target - current)
        result[symbol] = current + scale * (target - current)

    return result


def apply_position_limits(
    target_positions: Dict[str, float],
    max_position: Optional[float] = None,
    max_concentration: Optional[float] = None,
    book_size: Optional[float] = None,
) -> Dict[str, float]:
    """Apply position limits to target positions.

    Args:
        target_positions: {symbol: notional_value}
        max_position: Maximum absolute position per symbol
        max_concentration: Maximum position as fraction of book
        book_size: Total book size (required if max_concentration set)

    Returns:
        Clipped target positions
    """
    result = {}

    for symbol, position in target_positions.items():
        clipped = position

        if max_position is not None:
            clipped = max(-max_position, min(max_position, clipped))

        if max_concentration is not None and book_size is not None:
            max_pos = max_concentration * book_size
            clipped = max(-max_pos, min(max_pos, clipped))

        result[symbol] = clipped

    return result
