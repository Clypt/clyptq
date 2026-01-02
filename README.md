# ClyptQ

Production-ready quantitative cryptocurrency trading engine with realistic backtesting and live execution capabilities.

## Overview

Quantitative trading system for cryptocurrency markets featuring alpha factor computation, portfolio optimization, and event-driven backtesting with proper look-ahead bias prevention.

## Features

### Core Trading System
- Alpha factor framework with 15+ production-ready factors
- Multiple portfolio construction strategies (Top-N, Score-Weighted, Risk Parity, Blended)
- Event-driven backtesting engine with look-ahead bias prevention
- Paper/Live trading with real-time factor-based execution
- Multi-timeframe support (1h, 4h, 1d, 1w) with automatic alignment
- Walk-forward optimization for parameter tuning
- Multi-strategy blending with flexible allocation
- Cash constraint enforcement and overselling prevention
- Deterministic backtests for reproducibility

### Analytics & Reporting
- Monte Carlo simulation with confidence intervals and risk metrics
- Performance attribution (factor/asset contribution, cost breakdown)
- Rolling metrics (Sharpe, Sortino, volatility, drawdown)
- HTML report generation with equity curves and analytics
- Comprehensive drawdown analysis (duration, recovery, underwater periods)

### Infrastructure (v0.6.1)
- 5 domain groups architecture (core, infra, data, trading, analytics)
- SaaS-ready infrastructure (health monitoring, multi-tenancy, export utilities)
- Research tools (data exploration, factor analyzer, strategy backtester)
- Comprehensive testing suite (186 unit tests)
- Security audit tools (credential safety, secrets management, PII redaction)

### Integrations
- CCXT integration for 100+ cryptocurrency exchanges
- 61% test coverage with 186 passing tests
- Production-ready code quality and maintainability

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
from clyptq.core.base import Strategy
from clyptq.trading.factors.library.momentum import MomentumFactor
from clyptq.trading.portfolio.constructors import TopNConstructor
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq import Constraints, CostModel
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

engine = BacktestEngine(
    strategy=MyStrategy(),
    data_store=store,
    executor=executor,
    initial_capital=10000.0
)

end = datetime.now()
start = end - timedelta(days=90)
result = engine.run(start, end, verbose=True)

from clyptq.analytics.metrics import print_metrics
print_metrics(result.metrics)
```

## Paper Trading

```python
from clyptq.data.stores.live_store import LiveDataStore
from clyptq.trading.execution.live import LiveExecutor
from clyptq.trading.engine import LiveEngine
from clyptq.core.types import EngineMode
from datetime import datetime
import time

# Setup
strategy = MyStrategy()
universe = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

cost_model = CostModel(maker_fee=0.001, taker_fee=0.001, slippage_bps=5.0)
executor = LiveExecutor(
    exchange_id="binance",
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
    paper_mode=True,
)

# Fetch historical for warmup
store = LiveDataStore(lookback_days=strategy.warmup_periods() + 30)
for symbol in universe:
    df = executor.fetch_historical(symbol, days=strategy.warmup_periods() + 30)
    store.add_historical(symbol, df)

# Create engine
engine = LiveEngine(
    strategy=strategy,
    data_store=store,
    executor=executor,
    initial_capital=10000.0,
    mode=EngineMode.PAPER,
)

# Main loop
while True:
    now = datetime.now()
    prices = executor.fetch_prices(universe)

    result = engine.step(now, prices)

    if result.action == "rebalance":
        print(f"Rebalanced: {len(result.fills)} fills")
        for fill in result.fills:
            print(f"  {fill.side} {fill.symbol}: {fill.amount} @ {fill.price}")

    time.sleep(60)
```

## Architecture (v0.6.0 - 5 Domain Groups)

**Design**: 15 folders → 5 domain groups for clear separation of concerns

```
clyptq/
├── core/                       # Core ABCs & type definitions
│   ├── base.py                 # Factor, Executor, Strategy, PortfolioConstructor ABC
│   └── types.py                # Order, Fill, Position, Constraints, etc.
│
├── infra/                      # Infrastructure & utilities
│   ├── health/                 # Health monitoring (SaaS)
│   ├── security/               # Security & secrets management
│   ├── utils/                  # Utilities & logging
│   └── cli/                    # CLI interface
│
├── data/                       # Data management
│   ├── stores/                 # Data storage
│   │   ├── store.py            # DataStore (backtest)
│   │   ├── live_store.py       # LiveDataStore (rolling window)
│   │   └── mtf_store.py        # MultiTimeframeStore
│   ├── streams/                # Real-time data streaming
│   └── loaders/
│       └── ccxt.py             # CCXT loader
│
├── trading/                    # Trading system
│   ├── engine/                 # Trading engines
│   │   ├── backtest.py         # BacktestEngine
│   │   └── live.py             # LiveEngine (paper/live)
│   ├── execution/              # Order execution
│   │   ├── backtest.py         # BacktestExecutor
│   │   ├── live.py             # LiveExecutor (CCXT integration)
│   │   ├── order_tracker.py    # Order tracking
│   │   └── position_sync.py    # Position synchronization
│   ├── factors/                # Factor system
│   │   ├── ops/                # Factor operations
│   │   │   ├── time_series.py  # ts_mean, ts_std, correlation
│   │   │   └── cross_sectional.py  # rank, normalize, winsorize
│   │   ├── mtf_factor.py       # MultiTimeframeFactor base
│   │   └── library/
│   │       ├── momentum.py     # Momentum, MultiTimeframeMomentum
│   │       ├── volatility.py   # Volatility factors
│   │       ├── mean_reversion.py   # Bollinger, ZScore, Percentile
│   │       ├── volume.py       # Volume, VolumeRatio, DollarVolume
│   │       ├── liquidity.py    # Amihud, EffectiveSpread, VolOfVol
│   │       └── size.py         # DollarVolumeSize
│   ├── portfolio/              # Portfolio management
│   │   ├── constructors.py     # TopN, ScoreWeighted, RiskParity, BlendedConstructor
│   │   ├── constraints.py      # Position constraints
│   │   └── state.py            # Portfolio state
│   ├── strategy/               # Strategy implementation
│   │   ├── base.py             # SimpleStrategy
│   │   └── blender.py          # StrategyBlender (multi-strategy)
│   ├── optimization/           # Optimization
│   │   └── walk_forward.py     # Walk-forward optimization
│   └── risk/                   # Risk management
│       ├── costs.py            # Trading costs
│       └── manager.py          # Risk manager
│
└── analytics/                  # Analytics & reporting
    ├── metrics.py              # Performance metrics
    ├── monte_carlo.py          # Monte Carlo simulation
    ├── attribution.py          # Performance attribution
    ├── rolling.py              # Rolling metrics
    ├── drawdown.py             # Drawdown analysis
    └── report.py               # HTML report generation
```

## Engine Modes

**Backtest**: Deterministic execution with historical data, no real money

```python
from clyptq.trading.engine import BacktestEngine
from clyptq.trading.execution.backtest import BacktestExecutor

executor = BacktestExecutor(cost_model)
engine = BacktestEngine(strategy, data_store, executor, initial_capital)
result = engine.run(start, end)
```

**Paper**: Real-time execution with real market data, no real money

```python
from clyptq.trading.engine import LiveEngine
from clyptq.trading.execution.live import LiveExecutor
from clyptq.core.types import EngineMode

executor = LiveExecutor(
    exchange_id="binance",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    paper_mode=True
)

engine = LiveEngine(strategy, live_store, executor, initial_capital, mode=EngineMode.PAPER)
result = engine.step(timestamp, prices)
```

**Live**: Real-time execution with real money (use with caution)

```python
from clyptq.trading.engine import LiveEngine
from clyptq.trading.execution.live import LiveExecutor
from clyptq.core.types import EngineMode

executor = LiveExecutor(
    exchange_id="binance",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    paper_mode=False,
    sandbox=True  # Use sandbox for testing
)

engine = LiveEngine(strategy, live_store, executor, initial_capital, mode=EngineMode.LIVE)
result = engine.step(timestamp, prices)
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
6. Engine.step() vs run_backtest() consistency

### CI/CD Notes

- Streaming tests are skipped in GitHub Actions (Binance API geo-restricted)
- Tests run fully in local development environment
- All core functionality tests pass in CI

## Performance Metrics

- Returns: Total, Annualized, CAGR
- Risk: Volatility, Sharpe, Sortino, Calmar, Max Drawdown
- Trading: Win Rate, Profit Factor, Average P&L
- Exposure: Leverage, Number of Positions
- Attribution: Factor contribution, Asset contribution, Cost breakdown
- Rolling: Rolling Sharpe, Sortino, Volatility, Drawdown
- Drawdown: Duration, Recovery time, Underwater periods
- Monte Carlo: Confidence intervals, CVaR, Loss probability

## License

MIT License

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk. Not financial advice. Test thoroughly before live trading.
