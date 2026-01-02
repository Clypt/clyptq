# ClyptQ Examples

Production-ready examples for backtesting, paper trading, live trading, and quantitative research.

## Quick Start

### 1. Backtesting
Historical data backtest with performance metrics.

```bash
python examples/backtesting.py
```

No API keys required. Downloads 720 days of data and runs backtest.

### 2. Paper Trading
Real-time prices, simulated orders (no real money).

```bash
python examples/paper_trading.py
```

Requires API keys. Press Ctrl+C to stop and view results.

### 3. Live Trading
Real orders with real money (USE WITH EXTREME CAUTION).

```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
python examples/live_trading.py
```

Types "YES" to confirm before execution.

### 4. Quantitative Research
Comprehensive factor research framework.

```bash
python examples/research.py
```

**Features:**
- Tests 3 strategy variants (Baseline, Normalized, ShortLookback)
- Performance comparison table
- Monte Carlo simulation (1000 runs)
- Performance attribution analysis
- Factor contribution breakdown
- Generates HTML report: `quant_research_report.html`

**Output:**
- Strategy comparison metrics
- Detailed performance analysis
- Risk metrics (VaR, CVaR, probability of loss)
- Sharpe ratio confidence intervals
- Factor analysis
- HTML report with equity curve, attribution, drawdowns, rolling metrics

## Files

### strategy.py
Common momentum + volatility strategy (Top-3 positions, daily rebalance).

### backtesting.py
Simple backtest example (90-day test period).

### paper_trading.py
Paper trading loop with live prices (60-second intervals).

### live_trading.py
Live trading with confirmation prompt (REAL MONEY).

### research.py
**Quantitative research framework** with:
- Multiple strategy variants comparison
- Cross-sectional normalization testing
- Lookback period sensitivity analysis
- Performance attribution
- Monte Carlo risk assessment
- Comprehensive HTML reporting

## Requirements

- Python 3.10+
- Internet connection (for data download)
- API keys (for paper/live trading only)

## Notes

- Backtesting: No API keys, downloads data automatically
- Paper trading: API keys required, no real orders
- Live trading: API keys + "YES" confirmation, REAL orders
- Research: No API keys, generates comprehensive analysis
