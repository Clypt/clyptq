# ClyptQ Examples

Jupyter notebook examples for quantitative crypto trading research.

## Quick Start

```bash
jupyter notebook examples/01_quickstart.ipynb
```

## Notebook Overview

### 1. Quickstart (01_quickstart.ipynb)
**Goal**: Run your first backtest in 5 minutes

**What you'll learn**:
- Load crypto data from Binance
- Create a simple momentum strategy
- Run backtest and analyze results

**Time**: 5 minutes

---

### 2. Data Exploration (02_data_exploration.ipynb)
**Goal**: Understand your data before backtesting

**What you'll learn**:
- Data quality checks (missing values, outliers)
- Price distribution and correlation analysis
- Volume and liquidity patterns
- Identify data issues early

**Time**: 10 minutes

---

### 3. Factor Research (03_factor_research.ipynb)
**Goal**: Develop and validate alpha factors

**What you'll learn**:
- Signal quality analysis (IC, IR, hit rate)
- IC decay: how long factors stay predictive
- Turnover-performance tradeoff
- Factor orthogonalization (remove redundancy)

**Key concepts**:
- **IC (Information Coefficient)**: Correlation between factor and future returns
- **IR (Information Ratio)**: IC mean / IC std (consistency)
- **IC Decay**: How long does the signal last?
- **Orthogonalization**: Remove redundancy between factors

**Time**: 20 minutes

---

### 4. Strategy Comparison (04_strategy_comparison.ipynb)
**Goal**: Compare different strategy approaches

**What you'll learn**:
- Momentum vs Mean Reversion
- Multi-factor combination
- Long-short market neutral
- Performance attribution
- Monte Carlo risk analysis

**Strategies tested**:
1. **Momentum Only**: Trend following (works in bull markets)
2. **Mean Reversion**: Contrarian (works in ranging/bear)
3. **Multi-Factor**: Combined signals (robust)
4. **Long-Short**: Market neutral (hedged)

**Time**: 15 minutes

---

### 5. Parameter Optimization (05_parameter_optimization.ipynb)
**Goal**: Find optimal strategy parameters

**What you'll learn**:
- Grid search with cross-validation
- Walk-forward analysis
- Out-of-sample validation
- Overfitting detection
- Parameter stability analysis

**Methods**:
- **Grid Search**: Test all parameter combinations
- **Walk-Forward**: Rolling train/test windows
- **Out-of-Sample**: Train on first half, test on second half
- **Stability Analysis**: Are nearby parameters also good?

**Time**: 25 minutes

---

### 6. Portfolio Optimization (06_portfolio_optimization.ipynb)
**Goal**: Optimize portfolio construction methods

**What you'll learn**:
- TopN (simple selection)
- ScoreWeighted (factor-based weights)
- Mean-Variance (Markowitz)
- Risk Parity (equal risk contribution)
- Blended (combine multiple methods)

**Recommendation**:
- **Beginners**: TopN
- **Robustness**: RiskParity or Blended
- **Max Sharpe**: MeanVariance (watch overfitting)
- **Factor strategies**: ScoreWeighted

**Time**: 20 minutes

---

### 7. Risk Analysis (07_risk_analysis.ipynb)
**Goal**: Comprehensive risk assessment before production

**What you'll learn**:
- Monte Carlo simulation (bootstrap 1000 runs)
- Drawdown analysis
- Rolling risk metrics
- Stress testing
- Risk limits and monitoring

**Production checklist**:
- P(Loss) <20%
- Max Drawdown <30%
- Min Sharpe >0.5
- CVaR (5%) <15%

**Time**: 20 minutes

---

### 8. Paper Trading (08_paper_trading.ipynb)
**Goal**: Deploy strategy to paper trading

**What you'll learn**:
- Set up LiveEngine with LiveDataStore
- Paper mode (simulated orders)
- Real-time factor computation
- Position tracking and reconciliation
- Transition from backtest to live

**Key differences**: Backtest vs Paper vs Live
- **Backtest**: Historical data, fast, no risk
- **Paper**: Real-time data, simulated orders, no risk
- **Live**: Real-time data, real orders, real money

**Time**: 25 minutes

---

## Research Workflow

Follow notebooks in order for complete research workflow:

```
01_quickstart.ipynb            
    ↓
02_data_exploration.ipynb        
    ↓
03_factor_research.ipynb
    ↓
04_strategy_comparison.ipynb
    ↓
05_parameter_optimization.ipynb
    ↓
06_portfolio_optimization.ipynb
    ↓
07_risk_analysis.ipynb
    ↓
08_paper_trading.ipynb
```

**Total time**: ~2.5 hours for complete workflow

## Installation

```bash
pip install clyptq jupyter
```

## Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Requirements

- Python 3.10+
- Jupyter or JupyterLab
- Internet connection (for data download)

## Tips

**Data Download**:
- First run may take 30-60 seconds per notebook
- Data is cached by CCXT
- Use smaller universe for faster testing

**Performance**:
- Start with 5-10 symbols
- Use 365 days for quick tests
- 720 days for production backtests

**Customization**:
- All notebooks use same data structure
- Easy to swap factors, strategies, parameters
- Modify constraints, costs, universe as needed

## Support

- **Documentation**: `docs/`
- **Issues**: https://github.com/Clypt/clyptq/issues
- **Examples**: This directory

## Next Steps

After completing notebooks:

1. **Read QUANT_RESEARCH.md**: Comprehensive research guide
2. **Explore Factor Library**: `clyptq/trading/factors/library/`
3. **Custom Factors**: Implement your own in `clyptq/core/base.py`
4. **Production**: Deploy to paper trading, then live

## License

MIT
