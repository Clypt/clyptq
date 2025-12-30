# Changelog

## [0.5.0] - 2025-12-30

### Added

**Monte Carlo Simulation**
- MonteCarloSimulator with bootstrap resampling
- Confidence intervals (5%, 50%, 95%)
- Risk metrics (CVaR, loss probability)
- 6 tests with 69% coverage

**Multi-Timeframe Support**
- MultiTimeframeStore for 1h/4h/1d/1w data
- Automatic timeframe alignment and resampling
- MultiTimeframeFactor base class
- MultiTimeframeMomentum factor
- 18 tests with 90%+ coverage

**Performance Attribution**
- Factor-level contribution analysis
- Asset-level contribution tracking
- Transaction cost breakdown
- Cash drag measurement
- 8 tests with 84% coverage

**Rolling Metrics**
- Rolling Sharpe, Sortino, Volatility
- Rolling max drawdown tracking
- Configurable window sizes
- 9 tests with 99% coverage

**Drawdown Analysis**
- Drawdown period detection
- Duration and recovery tracking
- Depth analysis with sorting
- Underwater equity curves
- 9 tests with 100% coverage

**HTML Report Generation**
- Comprehensive backtest reports
- Metrics tables with formatting
- Equity curve visualization
- Attribution and drawdown sections
- CSS styling included
- 8 tests with 95% coverage

**Factor Library Expansion**
- Volume factors: VolumeFactor, DollarVolumeFactor, VolumeRatioFactor
- Liquidity factors: AmihudFactor, EffectiveSpreadFactor, VolatilityOfVolatilityFactor
- Size factors: DollarVolumeSizeFactor
- 16 tests with 89-100% coverage

### Changed
- Deterministic backtests with sorted symbol lists
- Organized tests/ into categorized subfolders
- Organized examples/ into basic/advanced/analytics/live

### Fixed
- Point-in-time consistency in multi-timeframe data access

### Test Coverage
- 185 tests passing (was 106)
- Overall coverage: 69.53% (was 60.61%)

## [0.4.0] - 2025-12-29

### Added

**Mean Reversion Factors**
- BollingerFactor with overbought/oversold signals
- ZScoreFactor for mean reversion
- PercentileFactor for extreme value detection
- 9 tests with 95% coverage

**Operations Library**
- time_series: ts_mean, ts_std, ts_sum, correlation, covariance
- cross_sectional: rank, normalize, winsorize, demean

**Multi-Strategy Framework**
- StrategyBlender for portfolio allocation
- BlendedConstructor for strategy weight blending
- 9 tests with 90%+ coverage

**Walk-forward Optimization**
- Rolling train/test windows
- Parameter grid search
- Combined result aggregation
- 8 tests with 94% coverage

**OPERATIONS.md**
- Factor building blocks guide
- Usage examples for all operations

### Changed
- Simplified Factor base class (39 lines)
- Removed factor cache/registry (433 lines deleted)
- Folder structure cleanup: moved types.py to core/types.py
- Deleted config.py, cleaned __init__.py

### Examples
- 06_professional_backtest.py with operations library

### Test Coverage
- 106 tests passing (was 85)
- Overall coverage: 60.61% (was 51.20%)

## [0.3.0] - 2025-12-29

### Added

**Paper/Live Trading**
- Engine.step() for synchronous trading
- LiveDataStore with rolling window management
- ExecutionResult type
- CCXTExecutor.fetch_historical() for warmup

**Integration Test**
- test_parity.py for backtest-paper verification

**Examples**
- 05_paper_trading.py

### Test Coverage
- 85 tests passing
- Overall coverage: 51.20%

## [0.2.1] - 2025-12-29

### Fixed
- DNS resolution in hotspot environments (aiohttp.ThreadedResolver)
- Async resource cleanup (CCXT exchange/session)
- datetime.utcnow() deprecation warnings
- Git tracking for clyptq/data/ files

### Changed
- Skip streaming tests in CI (Binance geo-restriction)
- Increased load_markets wait time for stability

## [0.2.0] - 2025-12-29

### Added

**SaaS Export**
- BacktestResult.to_dict()
- BacktestResult.to_json()

**Async Streaming**
- run_live_stream() for real-time trading

**Delisting Detection**
- Automatic liquidation of delisted symbols

**Risk Management**
- Stop-loss and max drawdown kill switch

**CLI**
- clyptq command with data/backtest/live subcommands

**CI/CD**
- GitHub Actions workflow
- PyPI packaging setup

## [0.1.0] - 2025-12-28

### Added
- Event-driven backtesting engine
- Look-ahead bias prevention
- CCXT data loader
- Factor/Portfolio system
- Performance metrics
