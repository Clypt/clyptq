# Changelog

All notable changes to ClyptQ will be documented in this file.

## [0.3.0] - 2025-12-29

### Added
- Paper/Live trading with real-time factor-based execution
- `Engine.step()` method for synchronous live/paper trading
- `LiveDataStore` with rolling window data management
- `ExecutionResult` type for step execution results
- `CCXTExecutor.fetch_historical()` for warmup data fetching
- Integration test for backtest-paper parity verification
- Example: `examples/05_paper_trading.py`

### Changed
- Paper/Live CLI commands now execute full strategy pipeline
- CLI commands fetch historical data for warmup period
- Improved real-time rebalancing display with fills and P&L

### Testing
- Added 10 new unit tests for LiveDataStore and Engine.step()
- Added integration test: `test_backtest_vs_livestore_step_parity()`
- Total: 85 tests passing, coverage 51.20%

## [0.2.3] - 2025-12-28

### Added
- SaaS-ready data export (`BacktestResult.to_dict()`, `.to_json()`)
- Async streaming data support (WebSocket-style)
- Auto-liquidation of delisted symbols
- Risk management with stop-loss and max drawdown kill switch

### Changed
- Repackaged CLI commands for cleaner structure
- Removed verbose comments and documentation

## [0.2.0] - 2025-12-28

### Added
- Event-driven backtesting engine
- Look-ahead bias prevention
- CCXT data loader integration
- Factor and portfolio construction system
- Performance metrics calculation
- CLI commands for data download and backtesting

### Fixed
- Cash constraint enforcement
- Overselling prevention
- Rebalancing frequency control

## [0.1.0] - 2025-12-27

### Added
- Initial release
- Core engine architecture
- Basic backtesting functionality
- Factor framework
- Portfolio construction strategies
