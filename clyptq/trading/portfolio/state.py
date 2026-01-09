import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

from clyptq.core.types import Fill, OrderSide, Position, Snapshot


MarketType = Literal["spot", "futures", "margin"]


class PortfolioState:
    """Portfolio state tracking with cash constraint and overselling prevention."""

    def __init__(self, initial_cash: float = 10000.0):
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0

    def apply_fill(self, fill: Fill) -> None:
        trade_value = fill.amount * fill.price

        if fill.side == OrderSide.BUY:
            required_cash = trade_value + fill.fee
            if self.cash < required_cash:
                raise ValueError(
                    f"Insufficient cash for {fill.symbol}: "
                    f"need {required_cash:.2f}, have {self.cash:.2f}"
                )

            self.cash -= required_cash

            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]
                total_cost = pos.amount * pos.avg_price + trade_value
                total_amount = pos.amount + fill.amount
                pos.avg_price = total_cost / total_amount if total_amount > 0 else fill.price
                pos.amount = total_amount
            else:
                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    amount=fill.amount,
                    avg_price=fill.price,
                )

        else:
            current_amount = (
                self.positions[fill.symbol].amount
                if fill.symbol in self.positions
                else 0.0
            )

            if fill.amount > current_amount + 1e-8:
                raise ValueError(
                    f"Overselling {fill.symbol}: "
                    f"trying to sell {fill.amount:.4f}, only have {current_amount:.4f}"
                )

            self.cash += trade_value - fill.fee

            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]

                realized_pnl = (fill.price - pos.avg_price) * fill.amount
                pos.realized_pnl += realized_pnl
                self._cumulative_realized_pnl += realized_pnl

                pos.amount -= fill.amount

                if abs(pos.amount) < 1e-8:
                    del self.positions[fill.symbol]

        self._trades_count += 1

    def get_snapshot(
        self, timestamp: datetime, prices: Dict[str, float]
    ) -> Snapshot:
        positions_value = 0.0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                market_value = pos.amount * current_price
                positions_value += market_value
                pos.unrealized_pnl = (current_price - pos.avg_price) * pos.amount

        equity = self.cash + positions_value

        return Snapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            positions=self.positions.copy(),
            positions_value=positions_value,
        )

    def equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (cash + positions value).

        Args:
            prices: Current prices {symbol: price}

        Returns:
            Total equity value
        """
        positions_value = sum(
            pos.amount * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)

        if eq < 1e-8:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            if symbol in prices:
                market_value = pos.amount * prices[symbol]
                weights[symbol] = market_value / eq

        return weights

    def save_state(self, filepath: Path) -> None:
        state = {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "amount": pos.amount,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "trades_count": self._trades_count,
            "cumulative_realized_pnl": self._cumulative_realized_pnl,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: Path) -> "PortfolioState":
        with open(filepath, "r") as f:
            state = json.load(f)

        portfolio = cls(initial_cash=state["initial_cash"])
        portfolio.cash = state["cash"]
        portfolio._trades_count = state.get("trades_count", 0)
        portfolio._cumulative_realized_pnl = state.get("cumulative_realized_pnl", 0.0)

        # Restore positions
        for symbol, pos_data in state["positions"].items():
            portfolio.positions[symbol] = Position(
                symbol=pos_data["symbol"],
                amount=pos_data["amount"],
                avg_price=pos_data["avg_price"],
                unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                realized_pnl=pos_data.get("realized_pnl", 0.0),
            )

        return portfolio

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.positions.clear()
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def total_realized_pnl(self) -> float:
        return self._cumulative_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def __repr__(self) -> str:
        return (
            f"PortfolioState(cash={self.cash:.2f}, "
            f"positions={self.num_positions}, "
            f"trades={self._trades_count})"
        )


class FuturesPortfolioState:
    """Futures portfolio with margin-based leverage trading.

    Key differences from spot:
    - Positions use margin (collateral) instead of full value
    - Supports both long and short positions
    - Unrealized PnL affects available margin
    - Liquidation risk when margin ratio drops

    Example:
        portfolio = FuturesPortfolioState(initial_cash=10000, leverage=10)
        # With 10x leverage, $10000 can control $100000 worth of positions
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        leverage: float = 1.0,
    ):
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")
        if leverage < 1.0:
            raise ValueError(f"leverage must be >= 1.0, got {leverage}")

        self.initial_cash = initial_cash
        self.cash = initial_cash  # Available margin (collateral)
        self.leverage = leverage
        self.positions: Dict[str, Position] = {}
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0
        self._used_margin = 0.0  # Margin locked in positions

    def apply_fill(self, fill: Fill) -> None:
        """Apply fill to portfolio (futures margin logic).

        For futures:
        - BUY (long): margin = notional / leverage
        - SELL (short): margin = notional / leverage (same)
        - Close position: release margin + realize PnL
        """
        notional_value = fill.amount * fill.price
        required_margin = notional_value / self.leverage

        if fill.side == OrderSide.BUY:
            # Opening or increasing long / Closing short
            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]

                if pos.amount < 0:
                    # Closing short position
                    close_amount = min(fill.amount, abs(pos.amount))
                    # Release margin based on ENTRY price, not current price
                    entry_notional = close_amount * pos.avg_price
                    released_margin = entry_notional / self.leverage

                    # Realize PnL (short: profit when price drops)
                    realized_pnl = (pos.avg_price - fill.price) * close_amount
                    pos.realized_pnl += realized_pnl
                    self._cumulative_realized_pnl += realized_pnl

                    # Release margin and add PnL
                    self._used_margin -= released_margin
                    self.cash += realized_pnl - fill.fee

                    pos.amount += close_amount

                    # If fully closed, remove position
                    if abs(pos.amount) < 1e-8:
                        del self.positions[fill.symbol]
                        self._trades_count += 1
                        return

                    # Remaining is opening new long
                    remaining = fill.amount - close_amount
                    if remaining > 1e-8:
                        new_margin = (remaining * fill.price) / self.leverage
                        if self.cash < new_margin + fill.fee:
                            raise ValueError(
                                f"Insufficient margin for {fill.symbol}: "
                                f"need {new_margin + fill.fee:.2f}, have {self.cash:.2f}"
                            )
                        self.cash -= new_margin + fill.fee
                        self._used_margin += new_margin
                        pos.amount = remaining
                        pos.avg_price = fill.price
                else:
                    # Increasing long position
                    total_margin = required_margin + fill.fee
                    if self.cash < total_margin:
                        raise ValueError(
                            f"Insufficient margin for {fill.symbol}: "
                            f"need {total_margin:.2f}, have {self.cash:.2f}"
                        )
                    self.cash -= total_margin
                    self._used_margin += required_margin

                    # Update avg price
                    old_notional = pos.amount * pos.avg_price
                    new_notional = old_notional + notional_value
                    pos.amount += fill.amount
                    pos.avg_price = new_notional / pos.amount if pos.amount > 0 else fill.price
            else:
                # New long position
                total_margin = required_margin + fill.fee
                if self.cash < total_margin:
                    raise ValueError(
                        f"Insufficient margin for {fill.symbol}: "
                        f"need {total_margin:.2f}, have {self.cash:.2f}"
                    )
                self.cash -= total_margin
                self._used_margin += required_margin

                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    amount=fill.amount,
                    avg_price=fill.price,
                )

        else:  # SELL
            # Opening or increasing short / Closing long
            if fill.symbol in self.positions:
                pos = self.positions[fill.symbol]

                if pos.amount > 0:
                    # Closing long position
                    close_amount = min(fill.amount, pos.amount)
                    # Release margin based on ENTRY price, not current price
                    entry_notional = close_amount * pos.avg_price
                    released_margin = entry_notional / self.leverage

                    # Realize PnL (long: profit when price rises)
                    realized_pnl = (fill.price - pos.avg_price) * close_amount
                    pos.realized_pnl += realized_pnl
                    self._cumulative_realized_pnl += realized_pnl

                    # Release margin and add PnL
                    self._used_margin -= released_margin
                    self.cash += realized_pnl - fill.fee

                    pos.amount -= close_amount

                    if abs(pos.amount) < 1e-8:
                        del self.positions[fill.symbol]
                        self._trades_count += 1
                        return

                    # Remaining is opening new short
                    remaining = fill.amount - close_amount
                    if remaining > 1e-8:
                        new_margin = (remaining * fill.price) / self.leverage
                        if self.cash < new_margin + fill.fee:
                            raise ValueError(
                                f"Insufficient margin for {fill.symbol}: "
                                f"need {new_margin + fill.fee:.2f}, have {self.cash:.2f}"
                            )
                        self.cash -= new_margin + fill.fee
                        self._used_margin += new_margin
                        pos.amount = -remaining
                        pos.avg_price = fill.price
                else:
                    # Increasing short position
                    total_margin = required_margin + fill.fee
                    if self.cash < total_margin:
                        raise ValueError(
                            f"Insufficient margin for {fill.symbol}: "
                            f"need {total_margin:.2f}, have {self.cash:.2f}"
                        )
                    self.cash -= total_margin
                    self._used_margin += required_margin

                    # Update avg price (negative amount for short)
                    old_notional = abs(pos.amount) * pos.avg_price
                    new_notional = old_notional + notional_value
                    pos.amount -= fill.amount  # More negative
                    pos.avg_price = new_notional / abs(pos.amount)
            else:
                # New short position
                total_margin = required_margin + fill.fee
                if self.cash < total_margin:
                    raise ValueError(
                        f"Insufficient margin for {fill.symbol}: "
                        f"need {total_margin:.2f}, have {self.cash:.2f}"
                    )
                self.cash -= total_margin
                self._used_margin += required_margin

                self.positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    amount=-fill.amount,  # Negative for short
                    avg_price=fill.price,
                )

        self._trades_count += 1

    def equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (margin + unrealized PnL).

        Equity = available_margin + used_margin + unrealized_pnl
        """
        unrealized_pnl = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                if pos.amount > 0:  # Long
                    unrealized_pnl += (current_price - pos.avg_price) * pos.amount
                else:  # Short
                    unrealized_pnl += (pos.avg_price - current_price) * abs(pos.amount)

        return self.cash + self._used_margin + unrealized_pnl

    def buying_power(self, prices: Dict[str, float]) -> float:
        """Calculate available buying power with leverage."""
        return self.cash * self.leverage

    def get_snapshot(
        self, timestamp: datetime, prices: Dict[str, float]
    ) -> Snapshot:
        """Get portfolio snapshot."""
        positions_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                # For futures, position value is the notional
                notional = abs(pos.amount) * current_price
                positions_value += notional

                # Update unrealized PnL
                if pos.amount > 0:  # Long
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.amount
                else:  # Short
                    pos.unrealized_pnl = (pos.avg_price - current_price) * abs(pos.amount)

        eq = self.equity(prices)

        return Snapshot(
            timestamp=timestamp,
            equity=eq,
            cash=self.cash,
            positions=self.positions.copy(),
            positions_value=positions_value,
            leverage=positions_value / eq if eq > 0 else 0.0,
        )

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get position weights (signed: negative for short)."""
        eq = self.equity(prices)
        if eq < 1e-8:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            if symbol in prices:
                # Signed notional (negative for short)
                signed_notional = pos.amount * prices[symbol]
                weights[symbol] = signed_notional / eq

        return weights

    def margin_ratio(self, prices: Dict[str, float]) -> float:
        """Calculate margin ratio (equity / positions_value).

        Lower ratio = higher risk of liquidation.
        Typically liquidated at ~0.5% (varies by exchange).
        """
        positions_value = sum(
            abs(pos.amount) * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        if positions_value < 1e-8:
            return float('inf')

        return self.equity(prices) / positions_value

    def reset(self) -> None:
        """Reset portfolio state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self._trades_count = 0
        self._cumulative_realized_pnl = 0.0
        self._used_margin = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def total_realized_pnl(self) -> float:
        return self._cumulative_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def used_margin(self) -> float:
        return self._used_margin

    def __repr__(self) -> str:
        return (
            f"FuturesPortfolioState(cash={self.cash:.2f}, "
            f"leverage={self.leverage}x, "
            f"positions={self.num_positions}, "
            f"used_margin={self._used_margin:.2f})"
        )
