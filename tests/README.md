# Tests

Comprehensive test suite for the Clypt Trading Engine.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_data_store.py

# Run with verbose output
pytest -v
```

## Test Structure

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **test_data_store.py** - Look-ahead bias prevention, data availability
- **test_portfolio_state.py** - Cash constraints, overselling prevention
- **test_engine_core.py** - Rebalancing frequency, order generation
- **test_risk_manager.py** - Position limits, stop loss, take profit
- **test_order_tracker.py** - Order state tracking
- **test_position_sync.py** - Position synchronization
- **test_live_factors.py** - Live data factor computation
- **test_saas_export.py** - Result export functionality

### Integration Tests (`tests/integration/`)

Test system-wide behavior:

- **test_parity.py** - Backtest-Paper parity verification (CRITICAL)

## Critical Tests

These tests validate core engine correctness:

1. **Look-ahead Bias Prevention** (`test_data_store.py`)
   - Ensures strategies cannot access future information

2. **Cash Constraint Enforcement** (`test_portfolio_state.py`)
   - Prevents buying with insufficient funds

3. **Overselling Prevention** (`test_portfolio_state.py`)
   - Blocks selling more than owned

4. **Rebalancing Frequency** (`test_engine_core.py`)
   - Ensures rebalancing happens only once per period

5. **Backtest-Paper Parity** (`test_parity.py`)
   - Verifies identical results across execution modes

## Writing Tests

Follow these guidelines:

```python
def test_feature_behavior():
    """Short description of what this tests."""
    # Setup
    obj = create_test_object()

    # Execute
    result = obj.do_something()

    # Verify
    assert result == expected
```

**Best Practices:**
- Use descriptive test names
- Keep tests focused and independent
- Minimize comments (code should be self-documenting)
- Use fixtures for common setup
- Test edge cases and failure modes

## Test Data

Create deterministic test data for reproducible results:

```python
dates = pd.date_range(start=datetime(2023, 1, 1), periods=30, freq="D")
data = pd.DataFrame({
    "open": [100.0 + i for i in range(30)],
    "high": [101.0 + i for i in range(30)],
    "low": [99.0 + i for i in range(30)],
    "close": [100.5 + i for i in range(30)],
    "volume": [1000.0] * 30,
}, index=dates)
```

## Coverage

Target coverage levels:
- Unit tests: >80%
- Integration tests: >70%
- Critical components: 100%
