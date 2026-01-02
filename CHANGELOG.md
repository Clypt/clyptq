# Changelog

All notable changes to ClyptQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-01-02

### Added
- **Signal Quality Analysis**: `SignalQuality` class for IC, IR, and hit rate analysis
- **IC Decay Analysis**: `FactorAnalyzer.analyze_ic_decay()` for factor predictive power over time
- **Turnover-Performance Analysis**: `turnover_performance_frontier()` and `optimal_rebalance_frequency()`
- **Factor Orthogonalization**: Three new operations in `clyptq.trading.factors.ops.factor_combination`
  - `orthogonalize_factors()`: Gram-Schmidt orthogonalization
  - `pca_factors()`: PCA-based dimensionality reduction
  - `remove_correlation()`: Linear regression residualization
- **Analytics Folder Restructuring**: Organized into 4 categories (performance, factors, risk, reporting)
- **Documentation**: Updated OPERATIONS.md with factor combination operations
- **Example**: `professional_factor_research.py` demonstrating v0.8.0 features

### Changed
- Reorganized `clyptq/analytics/` into subfolders:
  - `performance/`: metrics, attribution, rolling, drawdown
  - `factors/`: analyzer, turnover, signal_quality
  - `risk/`: monte_carlo
  - `reporting/`: report, data_explorer

### Fixed
- Import paths updated across codebase (27 files) after analytics restructuring

## [0.7.0] - 2025-12-31

### Added
- Comprehensive testing suite:
  - 180 unit tests
  - 4 integration tests
  - 5 performance tests (load testing, memory leak detection)
  - 5 security tests (credential safety, PII redaction)
- SaaS infrastructure: Health monitoring, multi-tenancy support
- Research tools: Data exploration, factor analyzer, strategy backtester
- Coverage: 61.69% (exceeds 60% target)

### Changed
- Major folder reorganization: 15 folders → 5 domain groups
  - `core/`: Base classes and types
  - `infra/`: Health, security, utils, CLI
  - `data/`: Stores, streams, loaders
  - `trading/`: Engine, execution, strategy, factors, portfolio, risk, optimization
  - `analytics/`: Performance analytics and reporting
- Updated 339 import statements across entire codebase
- Clear separation of concerns and improved maintainability

### Fixed
- API Compatibility: CostModel (taker_fee_bps → taker_fee), Strategy.name, BacktestResult access
- CCXTLoader parameters (start/end → since/limit)
- LiveDataStore timezone handling (DatetimeIndex + timezone-aware datetime support)

## [0.6.0] - 2025-12-31

See v0.7.0 above (v0.6.0 and v0.7.0 released together).

## [0.5.0] - 2025-12-30

### Added
- **Monte Carlo Simulation**: Bootstrap resampling, confidence intervals, CVaR, risk metrics
- **Multi-Timeframe Support**: `MultiTimeframeStore` (1h/4h/1d/1w), point-in-time consistency
- **MultiTimeframeFactor**: Base class for multi-timeframe factors, `MultiTimeframeMomentum`
- **Performance Attribution**: Factor/asset-level contribution, transaction costs, cash drag
- **Rolling Metrics**: Rolling Sharpe/Sortino/volatility/drawdown with configurable windows
- **Drawdown Analysis**: Period detection, duration/recovery tracking, depth analysis
- **HTML Report**: Comprehensive backtest reports with metrics and equity curve chart
- Examples: `07_multi_timeframe.py`, `08_html_report.py`, `09_real_strategy_report.py`

### Changed
- Deterministic backtests: Sorted symbol lists for consistent execution order
- Organized tests/ and examples/ into categorized subfolders
- 169 tests passing, coverage 67.95%

## [0.4.0] - 2025-12-29

### Added
- **Mean Reversion Factors**: `BollingerFactor`, `ZScoreFactor`, `PercentileFactor`
- **Operations Library**: Reusable factor building blocks (`time_series`, `cross_sectional`)
- **Multi-Strategy Framework**: `StrategyBlender` for portfolio allocation across strategies
- **Walk-forward Optimization**: Rolling train/test windows for robust parameter tuning
- **BlendedConstructor**: Portfolio constructor for multi-strategy allocation
- Documentation: OPERATIONS.md for factor operations guide
- Example: `06_professional_backtest.py` with operations library

### Changed
- Factor refactoring: Simplified base.py (39 lines), removed cache/registry (433 lines)
- Folder structure cleanup: Moved types.py → core/types.py, deleted config.py, cleaned __init__.py
- 106 tests passing, coverage 60.61%

## [0.3.0] - 2025-12-29

### Added
- **Paper/Live Trading**: Real-time factor-based execution with `LiveEngine`
- **Engine.step()**: Synchronous trading method for live/paper modes
- **LiveDataStore**: Rolling window data management for real-time trading
- **ExecutionResult**: Type for step execution results
- **CCXTExecutor.fetch_historical()**: Warmup data loading for live trading
- Integration test: Backtest-paper parity verification
- Example: `05_paper_trading.py`

### Changed
- 85 tests passing, coverage 51.20%

## [0.2.1] - 2025-12-29

### Fixed
- DNS issues in hotspot environments: aiohttp.ThreadedResolver
- Async resource cleanup: CCXT exchange/session explicit close
- Streaming tests: CI skip for Binance geo-restrictions
- datetime.utcnow() deprecation warnings: datetime.now(timezone.utc)
- Test stability: Increased load_markets wait time
- Git tracking: Added clyptq/data/ files

## [0.2.0] - 2025-12-29

### Added
- **SaaS-Ready Export**: `BacktestResult.to_dict()`, `BacktestResult.to_json()`
- **Async Streaming**: `run_live_stream()` for real-time data
- **Delisting Auto-Liquidation**: Automatic position closure for delisted symbols
- **Risk Management**: Stop-loss, max drawdown kill switch integration
- **Paper/Live Executor**: Unified executor with `paper_mode` flag
- **CLI Commands**: `clyptq` command-line interface
- **PyPI Packaging**: Setup for PyPI distribution
- **GitHub Actions CI/CD**: Automated testing and deployment

## [0.1.0] - 2025-12-28

### Added
- Event-driven backtesting engine
- Look-ahead bias prevention
- CCXT data loader (100+ exchanges)
- Factor system (Momentum, Volatility, Volume, Liquidity)
- Portfolio construction (TopN, ScoreWeighted, RiskParity)
- Performance metrics (Sharpe, Sortino, Calmar, drawdown)
- Cash constraint validation
- Rebalancing frequency control
- Order generation (sells-first logic)

[Unreleased]: https://github.com/Clypt/clypt-trading-engine/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.5.0...v0.7.0
[0.6.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.5.0...v0.7.0
[0.5.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Clypt/clypt-trading-engine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Clypt/clypt-trading-engine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Clypt/clypt-trading-engine/releases/tag/v0.1.0
