"""
Portfolio constraint validation and enforcement.

Validates portfolio weights against constraints:
- Position size limits
- Number of positions
- Gross exposure
- Long/short restrictions
"""

from typing import Dict

from clypt.types import Constraints


class ConstraintViolation(Exception):
    """Raised when portfolio violates constraints."""

    pass


def validate_weights(weights: Dict[str, float], constraints: Constraints) -> None:
    """
    Validate portfolio weights against constraints.

    Args:
        weights: Dictionary of {symbol: weight}
        constraints: Portfolio constraints

    Raises:
        ConstraintViolation: If any constraint is violated
    """
    if not weights:
        return  # Empty portfolio is valid

    # Check number of positions
    num_positions = len(weights)
    if num_positions > constraints.max_num_positions:
        raise ConstraintViolation(
            f"Too many positions: {num_positions} > {constraints.max_num_positions}"
        )

    # Check position sizes
    for symbol, weight in weights.items():
        abs_weight = abs(weight)

        # Check minimum position size
        if abs_weight < constraints.min_position_size:
            raise ConstraintViolation(
                f"{symbol}: Position too small: {abs_weight:.4f} < {constraints.min_position_size:.4f}"
            )

        # Check maximum position size
        if abs_weight > constraints.max_position_size:
            raise ConstraintViolation(
                f"{symbol}: Position too large: {abs_weight:.4f} > {constraints.max_position_size:.4f}"
            )

        # Check short restrictions
        if weight < 0 and not constraints.allow_short:
            raise ConstraintViolation(
                f"{symbol}: Short position not allowed: {weight:.4f}"
            )

    # Check gross exposure
    gross_exposure = sum(abs(w) for w in weights.values())
    if gross_exposure > constraints.max_gross_exposure:
        raise ConstraintViolation(
            f"Gross exposure too high: {gross_exposure:.4f} > {constraints.max_gross_exposure:.4f}"
        )


def enforce_constraints(
    weights: Dict[str, float], constraints: Constraints, tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Enforce constraints on portfolio weights by adjusting them.

    This is a best-effort function that tries to make weights compliant:
    1. Remove positions below minimum size
    2. Clip positions above maximum size
    3. Remove short positions if not allowed
    4. Scale down if gross exposure too high
    5. Remove positions if too many

    Args:
        weights: Dictionary of {symbol: weight}
        constraints: Portfolio constraints
        tolerance: Numerical tolerance for comparisons

    Returns:
        Adjusted weights that satisfy constraints
    """
    if not weights:
        return {}

    adjusted = weights.copy()

    # 1. Remove positions below minimum size
    adjusted = {
        s: w
        for s, w in adjusted.items()
        if abs(w) >= constraints.min_position_size - tolerance
    }

    # 2. Remove short positions if not allowed
    if not constraints.allow_short:
        adjusted = {s: w for s, w in adjusted.items() if w >= -tolerance}

    # 3. Clip positions above maximum size
    for symbol in list(adjusted.keys()):
        weight = adjusted[symbol]
        if weight > constraints.max_position_size + tolerance:
            adjusted[symbol] = constraints.max_position_size
        elif weight < -constraints.max_position_size - tolerance:
            adjusted[symbol] = -constraints.max_position_size

    # 4. Limit number of positions (keep largest absolute weights)
    if len(adjusted) > constraints.max_num_positions:
        sorted_weights = sorted(
            adjusted.items(), key=lambda x: abs(x[1]), reverse=True
        )
        adjusted = dict(sorted_weights[: constraints.max_num_positions])

    # 5. Scale down if gross exposure too high
    gross_exposure = sum(abs(w) for w in adjusted.values())
    if gross_exposure > constraints.max_gross_exposure + tolerance:
        scale = constraints.max_gross_exposure / gross_exposure
        adjusted = {s: w * scale for s, w in adjusted.items()}

    return adjusted


def check_constraints(
    weights: Dict[str, float], constraints: Constraints, tolerance: float = 1e-6
) -> bool:
    """
    Check if portfolio weights satisfy constraints.

    Args:
        weights: Dictionary of {symbol: weight}
        constraints: Portfolio constraints
        tolerance: Numerical tolerance for comparisons

    Returns:
        True if all constraints satisfied, False otherwise
    """
    try:
        validate_weights(weights, constraints)
        return True
    except ConstraintViolation:
        return False


def get_constraint_violations(
    weights: Dict[str, float], constraints: Constraints, tolerance: float = 1e-6
) -> Dict[str, str]:
    """
    Get detailed information about constraint violations.

    Args:
        weights: Dictionary of {symbol: weight}
        constraints: Portfolio constraints
        tolerance: Numerical tolerance

    Returns:
        Dictionary of {constraint_name: violation_message}
        Empty dict if no violations
    """
    violations = {}

    if not weights:
        return violations

    # Check number of positions
    num_positions = len(weights)
    if num_positions > constraints.max_num_positions:
        violations["num_positions"] = (
            f"{num_positions} positions > max {constraints.max_num_positions}"
        )

    # Check gross exposure
    gross_exposure = sum(abs(w) for w in weights.values())
    if gross_exposure > constraints.max_gross_exposure + tolerance:
        violations["gross_exposure"] = (
            f"{gross_exposure:.4f} > max {constraints.max_gross_exposure:.4f}"
        )

    # Check individual positions
    position_violations = []
    for symbol, weight in weights.items():
        abs_weight = abs(weight)

        if abs_weight < constraints.min_position_size - tolerance:
            position_violations.append(
                f"{symbol}: {abs_weight:.4f} < min {constraints.min_position_size:.4f}"
            )

        if abs_weight > constraints.max_position_size + tolerance:
            position_violations.append(
                f"{symbol}: {abs_weight:.4f} > max {constraints.max_position_size:.4f}"
            )

        if weight < -tolerance and not constraints.allow_short:
            position_violations.append(f"{symbol}: short position not allowed")

    if position_violations:
        violations["positions"] = "; ".join(position_violations)

    return violations
