"""Position synchronization for live trading."""

from typing import Dict, List, Optional

from clyptq.core.types import Position


class PositionDiscrepancy:
    """Represents a position mismatch."""

    def __init__(
        self,
        symbol: str,
        internal_amount: float,
        exchange_amount: float,
        internal_avg_price: float,
        exchange_avg_price: float,
    ):
        self.symbol = symbol
        self.internal_amount = internal_amount
        self.exchange_amount = exchange_amount
        self.internal_avg_price = internal_avg_price
        self.exchange_avg_price = exchange_avg_price

    @property
    def amount_diff(self) -> float:
        return self.exchange_amount - self.internal_amount

    @property
    def is_critical(self) -> bool:
        return abs(self.amount_diff) > 1e-6


class PositionSynchronizer:
    """Syncs internal positions with exchange."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check_discrepancies(
        self,
        internal_positions: Dict[str, Position],
        exchange_positions: Dict[str, Dict],
    ) -> List[PositionDiscrepancy]:
        discrepancies = []

        all_symbols = set(internal_positions.keys()) | set(exchange_positions.keys())

        for symbol in all_symbols:
            internal_pos = internal_positions.get(symbol)
            exchange_pos = exchange_positions.get(symbol, {})

            internal_amount = internal_pos.amount if internal_pos else 0.0
            internal_avg_price = internal_pos.avg_price if internal_pos else 0.0

            exchange_amount = exchange_pos.get("amount", 0.0)
            exchange_avg_price = exchange_pos.get("avg_price", 0.0)

            diff = abs(exchange_amount - internal_amount)
            if diff > self.tolerance:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        internal_amount=internal_amount,
                        exchange_amount=exchange_amount,
                        internal_avg_price=internal_avg_price,
                        exchange_avg_price=exchange_avg_price,
                    )
                )

        return discrepancies

    def sync_positions(
        self,
        internal_positions: Dict[str, Position],
        exchange_positions: Dict[str, Dict],
    ) -> Dict[str, Position]:
        """Force sync internal to match exchange."""
        synced = {}

        for symbol, exchange_pos in exchange_positions.items():
            amount = exchange_pos.get("amount", 0.0)
            avg_price = exchange_pos.get("avg_price", 0.0)

            if amount > self.tolerance:
                synced[symbol] = Position(
                    symbol=symbol,
                    amount=amount,
                    avg_price=avg_price,
                    realized_pnl=0.0,
                )

        return synced

    def reconcile_position(
        self,
        internal_pos: Optional[Position],
        exchange_amount: float,
        exchange_avg_price: float,
    ) -> Optional[Position]:
        """Reconcile single position."""
        if exchange_amount < self.tolerance:
            return None

        if internal_pos is None:
            return Position(
                symbol="",
                amount=exchange_amount,
                avg_price=exchange_avg_price,
                realized_pnl=0.0,
            )

        diff = abs(exchange_amount - internal_pos.amount)
        if diff > self.tolerance:
            return Position(
                symbol=internal_pos.symbol,
                amount=exchange_amount,
                avg_price=exchange_avg_price,
                unrealized_pnl=0.0,
                realized_pnl=internal_pos.realized_pnl,
            )

        return internal_pos
