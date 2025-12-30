"""
Unit tests for PortfolioState - Critical Tests 2 & 3

Test 2: Cash Constraint Enforcement
Test 3: Overselling Prevention
"""

from datetime import datetime

import pytest

from clyptq.portfolio.state import PortfolioState
from clyptq.core.types import Fill, OrderSide, FillStatus


def test_insufficient_cash_rejected():
    """
    CRITICAL TEST 2: Cash constraint enforcement.

    portfolio_state.py:55-76 - MUST reject buy with insufficient cash.

    Tests that buying with insufficient cash raises ValueError.
    """
    # Create portfolio with limited cash
    portfolio = PortfolioState(initial_cash=1000.0)

    # Try to buy BTC worth $2000 (more than we have)
    expensive_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=10.0,  # 10 BTC
        price=200.0,  # @ $200 = $2000 trade value
        fee=10.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.apply_fill(expensive_fill)

    # Cash should be unchanged
    assert portfolio.cash == 1000.0


def test_sufficient_cash_accepted():
    """Test that buying with sufficient cash succeeds."""
    portfolio = PortfolioState(initial_cash=1000.0)

    # Buy within budget
    affordable_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=2.0,  # 2 BTC
        price=100.0,  # @ $100 = $200 trade value
        fee=2.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )

    # Should succeed
    portfolio.apply_fill(affordable_fill)

    # Check state
    assert portfolio.cash == 1000.0 - 200.0 - 2.0  # initial - trade_value - fee = 798
    assert "BTC/USDT" in portfolio.positions
    assert portfolio.positions["BTC/USDT"].amount == 2.0


def test_exact_cash_amount():
    """Test buying with exactly the available cash."""
    portfolio = PortfolioState(initial_cash=100.0)

    # Buy using all cash
    fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=99.0,  # $99 trade value
        fee=1.0,     # $1 fee, total = $100
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )

    # Should succeed (exactly enough)
    portfolio.apply_fill(fill)
    assert portfolio.cash == 0.0


def test_overselling_prevented():
    """
    CRITICAL TEST 3: Overselling prevention.

    portfolio_state.py:78-96 - MUST reject selling more than owned.

    Tests that selling more than the current position raises ValueError.
    """
    portfolio = PortfolioState(initial_cash=10000.0)

    # First, buy 1 BTC
    buy_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=1000.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(buy_fill)

    # Verify we have 1 BTC
    assert portfolio.positions["BTC/USDT"].amount == 1.0

    # Try to sell 2 BTC (more than we own)
    oversell_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=2.0,  # Trying to sell 2, but only have 1
        price=1100.0,
        fee=2.0,
        timestamp=datetime(2023, 1, 2),
        status=FillStatus.FILLED,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Overselling"):
        portfolio.apply_fill(oversell_fill)

    # Position should be unchanged
    assert portfolio.positions["BTC/USDT"].amount == 1.0


def test_selling_exact_amount():
    """Test selling exactly the owned amount."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy 5 BTC
    buy_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=5.0,
        price=1000.0,
        fee=5.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(buy_fill)

    # Sell exactly 5 BTC
    sell_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=5.0,
        price=1100.0,
        fee=5.0,
        timestamp=datetime(2023, 1, 2),
        status=FillStatus.FILLED,
    )

    # Should succeed
    portfolio.apply_fill(sell_fill)

    # Position should be closed
    assert "BTC/USDT" not in portfolio.positions


def test_selling_without_position():
    """Test that selling without owning the asset fails."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Try to sell BTC without owning any
    sell_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=1000.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )

    # Should raise ValueError (trying to sell 1, have 0)
    with pytest.raises(ValueError, match="Overselling"):
        portfolio.apply_fill(sell_fill)


def test_partial_sell():
    """Test selling part of a position."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy 10 BTC @ $999 + $10 fee = $9,990 (within budget)
    buy_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=10.0,
        price=999.0,  # Reduced to fit within $10,000 budget
        fee=10.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(buy_fill)

    # Sell 3 BTC (partial)
    sell_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=3.0,
        price=1100.0,
        fee=3.0,
        timestamp=datetime(2023, 1, 2),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(sell_fill)

    # Should have 7 BTC remaining
    assert portfolio.positions["BTC/USDT"].amount == 7.0


def test_average_price_calculation():
    """Test that average price is calculated correctly."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy 1 BTC @ $1000
    fill1 = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=1000.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill1)

    # Buy another 1 BTC @ $1200
    fill2 = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=1200.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 2),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill2)

    # Average price should be (1000 + 1200) / 2 = $1100
    avg_price = portfolio.positions["BTC/USDT"].avg_price
    assert abs(avg_price - 1100.0) < 1e-6


def test_realized_pnl_tracking():
    """Test that realized P&L is tracked correctly."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy 1 BTC @ $1000
    buy_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=1000.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(buy_fill)

    # Sell 1 BTC @ $1200 (profit = $200)
    sell_fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        amount=1.0,
        price=1200.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 2),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(sell_fill)

    # Position closed, but check total realized P&L
    assert portfolio.total_realized_pnl == 200.0


def test_snapshot_creation():
    """Test portfolio snapshot creation."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy some BTC
    fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=2.0,
        price=1000.0,
        fee=2.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill)

    # Create snapshot with current prices
    prices = {"BTC/USDT": 1100.0}
    snapshot = portfolio.get_snapshot(datetime(2023, 1, 2), prices)

    # Verify snapshot
    assert snapshot.cash == 10000.0 - 2000.0 - 2.0  # 7998
    assert snapshot.positions_value == 2.0 * 1100.0  # 2200
    assert snapshot.equity == snapshot.cash + snapshot.positions_value  # 10198
    assert snapshot.num_positions == 1


def test_get_weights():
    """Test portfolio weight calculation."""
    portfolio = PortfolioState(initial_cash=10000.0)

    # Buy 2 BTC @ $1000
    fill1 = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=2.0,
        price=1000.0,
        fee=2.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill1)

    # Buy 4 ETH @ $500
    fill2 = Fill(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        amount=4.0,
        price=500.0,
        fee=2.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill2)

    # Get weights with current prices
    prices = {"BTC/USDT": 1100.0, "ETH/USDT": 550.0}
    weights = portfolio.get_weights(prices)

    # Total equity = cash + positions_value
    # Cash = 10000 - 2000 - 2 - 2000 - 2 = 5996
    # BTC value = 2 * 1100 = 2200
    # ETH value = 4 * 550 = 2200
    # Total equity = 5996 + 2200 + 2200 = 10396

    # BTC weight = 2200 / 10396 ≈ 0.2116
    # ETH weight = 2200 / 10396 ≈ 0.2116

    assert abs(weights["BTC/USDT"] - 0.2116) < 0.01
    assert abs(weights["ETH/USDT"] - 0.2116) < 0.01


def test_state_persistence():
    """Test saving and loading portfolio state."""
    import tempfile
    from pathlib import Path

    portfolio = PortfolioState(initial_cash=10000.0)

    # Make some trades
    fill = Fill(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=1.0,
        price=1000.0,
        fee=1.0,
        timestamp=datetime(2023, 1, 1),
        status=FillStatus.FILLED,
    )
    portfolio.apply_fill(fill)

    # Save state
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "state.json"
        portfolio.save_state(filepath)

        # Load state
        loaded = PortfolioState.load_state(filepath)

        # Verify loaded state matches original
        assert loaded.initial_cash == portfolio.initial_cash
        assert loaded.cash == portfolio.cash
        assert len(loaded.positions) == len(portfolio.positions)
        assert "BTC/USDT" in loaded.positions
        assert loaded.positions["BTC/USDT"].amount == 1.0
