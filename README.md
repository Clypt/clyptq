# ClyptQ

Production-ready quantitative cryptocurrency trading engine with realistic backtesting and live execution capabilities.

## Overview

Quantitative trading system for cryptocurrency markets featuring alpha factor computation, portfolio optimization, and event-driven backtesting with proper look-ahead bias prevention.

## Features

- Alpha factor framework with extensible factor library
- Multiple portfolio construction strategies (Top-N, Score-Weighted, Risk Parity)
- Event-driven backtesting engine with look-ahead bias prevention
- Live trading via CCXT integration
- Comprehensive performance analytics with auto-detecting time frequencies
- Cash constraint enforcement and overselling prevention
- Proper rebalancing timing control

## Installation

```bash
pip install clyptq
```

Or from source:

```bash
git clone https://github.com/Clypt/clyptq.git
cd clyptq

python -m venv venv
source venv/bin/activate

pip install -e .
```

## CLI Usage

```bash
# Download top 60 symbols by 24h volume (90 days of data)
clyptq data download --exchange binance --days 90 --limit 60

# List downloaded data
clyptq data list

# Download specific symbols
clyptq data download --symbols BTC/USDT ETH/USDT SOL/USDT

# Run backtest
clyptq backtest --strategy MyStrategy --start 2024-01-01 --end 2024-03-01

# Run live trading
clyptq live --strategy MyStrategy --mode paper
```

## Quick Start

```python
from clyptq.data.loaders.ccxt import load_crypto_data
from clyptq.strategy.base import Strategy
from clyptq.factors.library.momentum import MomentumFactor
from clyptq.portfolio.construction import TopNConstructor
from clyptq.engine import Engine, BacktestExecutor
from clyptq import Constraints, CostModel, EngineMode
from datetime import datetime, timedelta

# Load data
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
store = load_crypto_data(symbols, days=180)

# Define strategy
class MyStrategy(Strategy):
    def factors(self):
        return [MomentumFactor(lookback=20)]

    def portfolio_constructor(self):
        return TopNConstructor(top_n=5)

    def constraints(self):
        return Constraints(
            max_position_size=0.3,
            max_gross_exposure=1.0
        )

# Run backtest
cost_model = CostModel(maker_fee=0.001, taker_fee=0.001)
executor = BacktestExecutor(cost_model)

engine = Engine(
    strategy=MyStrategy(),
    data_store=store,
    mode=EngineMode.BACKTEST,
    executor=executor,
    initial_capital=10000.0
)

end = datetime.now()
start = end - timedelta(days=90)
result = engine.run_backtest(start, end, verbose=True)

from clyptq.analytics.metrics import print_metrics
print_metrics(result.metrics)
```

## Architecture

```
clyptq/
├── types.py              # Core type definitions
├── config.py             # Configuration
├── cli/                  # Command-line tools
│   ├── main.py           # CLI entry point
│   └── commands/         # CLI commands
├── data/
│   ├── store.py          # Data storage
│   ├── validation.py     # Data quality
│   ├── streaming/        # Real-time data
│   ├── live/             # Live trading data
│   └── loaders/
│       └── ccxt.py       # CCXT loader
├── factors/
│   ├── base.py           # Factor base
│   ├── cache.py          # Caching
│   └── library/
│       ├── momentum.py   # Momentum
│       └── volatility.py # Volatility
├── portfolio/
│   ├── construction.py   # Constructors
│   ├── constraints.py    # Constraints
│   └── state.py          # Portfolio state
├── execution/            # Order execution
│   ├── base.py           # Base executor
│   ├── backtest.py       # Backtest executor
│   ├── live.py           # Live/Paper executor
│   ├── orders/           # Order management
│   └── positions/        # Position management
├── risk/                 # Risk management
│   ├── costs.py          # Trading costs
│   └── manager.py        # Risk manager
├── engine/
│   └── core.py           # Main orchestrator
├── analytics/
│   └── metrics.py        # Performance metrics
└── strategy/
    └── base.py           # Strategy base
```

## Engine Modes

**Backtest**: Deterministic execution with historical data, no real money

```python
engine = Engine(..., mode=EngineMode.BACKTEST)
```

**Paper**: Real-time execution with real market data, no real money

```python
engine = Engine(..., mode=EngineMode.PAPER)
```

**Live**: Real-time execution with real money (use with caution)

```python
from clyptq.execution.live import LiveExecutor

# Paper mode (simulated)
executor = LiveExecutor(
    exchange_id="binance",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    paper_mode=True
)

# Live mode (real money)
executor = LiveExecutor(
    exchange_id="binance",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    paper_mode=False,
    sandbox=True
)

engine = Engine(..., mode=EngineMode.LIVE, executor=executor)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=clyptq --cov-report=term-missing

# Run critical tests only
pytest tests/integration/test_parity.py -v
```

### Critical Tests

1. Look-ahead bias prevention in `available_symbols()`
2. Cash constraint enforcement
3. Overselling prevention
4. Rebalancing frequency control
5. Backtest-Paper parity verification

### CI/CD Notes

- Streaming tests are skipped in GitHub Actions (Binance API geo-restricted)
- Tests run fully in local development environment
- All core functionality tests pass in CI

## Performance Metrics

- Returns: Total, Annualized
- Risk: Volatility, Sharpe, Sortino, Max Drawdown
- Trading: Win Rate, Profit Factor, Average P&L
- Exposure: Leverage, Number of Positions

## License

MIT License

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk. Not financial advice. Test thoroughly before live trading.
