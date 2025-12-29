# Clypt Trading Engine - Examples

Real-world usage examples for the Clypt trading engine.

## Backtesting

### 01_simple_backtest.py
Basic momentum + volatility strategy backtest.

```bash
python examples/01_simple_backtest.py
```

**Features**:
- Historical CCXT data loading
- Multi-factor strategy
- Performance metrics
- Top N portfolio construction

### 02_universe_data.py
Custom universe with data validation.

```bash
python examples/02_universe_data.py
```

**Features**:
- Fixed universe selection
- Data validation
- Constraint enforcement

### 03_dynamic_universe.py
Dynamic universe based on volume.

```bash
python examples/03_dynamic_universe.py
```

**Features**:
- Volume-based universe selection
- Rebalancing on universe changes
- Look-ahead bias prevention

## Risk Analysis

### 05_monte_carlo.py
Monte Carlo simulation for backtest validation.

```bash
python examples/05_monte_carlo.py
```

**Features**:
- Bootstrap sampling of returns
- Confidence intervals (5%, 50%, 95%)
- Risk metrics (CVaR, max drawdown distribution)
- Probability of loss calculation
- Sharpe ratio confidence bands

## Live Trading

### 04_streaming_live.py
Real-time trading with streaming data.

```bash
# Set your API keys
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

python examples/04_streaming_live.py
```

**Features**:
- Async streaming (WebSocket-style)
- Paper/Live mode support
- Risk management integration
- Real-time factor computation
- Position synchronization

**Modes**:
- `paper_mode=True`: Simulation (no real orders)
- `paper_mode=False`: Live trading (requires API keys)
