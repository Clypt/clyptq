"""
Unified Engine for backtest/paper/live trading.

Usage:
    ```python
    class MyStrategy(Strategy):
        universe = DynamicUniverse(symbols=["BTC", "ETH"])
        data = {"ohlcv": OHLCVSpec(timeframe="1h")}
        rebalance_freq = "1d"

        def compute_signal(self):
            return operator.rank(self.provider["close"])

        def warmup_periods(self) -> int:
            return 50

    # Run backtest
    engine = Engine()
    result = engine.run(
        MyStrategy(),
        mode="backtest",
        data_path="data/crypto/",
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1),
    )

    # Run paper trading
    engine.run(MyStrategy(), mode="paper")

    # Run live trading
    engine.run(MyStrategy(), mode="live")
    ```
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from clyptq.core.types import (
    BacktestResult,
    CostModel,
    EngineMode,
    ExecutionResult,
    Fill,
    Order,
    OrderSide,
    Snapshot,
)
from clyptq.data.provider import DataProvider
from clyptq.data.spec import OHLCVSpec
from clyptq.infra.utils import get_logger
from clyptq.strategy.base import Strategy
from clyptq.trading.execution.base import Executor
from clyptq.trading.execution.backtest import BacktestExecutor
from clyptq.trading.portfolio.state import PortfolioState, FuturesPortfolioState


class Engine:
    """Unified trading engine for backtest/paper/live modes.

    The engine:
    1. Reads strategy's data/universe specs
    2. Builds appropriate DataProvider based on mode
    3. Binds provider to strategy
    4. Runs the trading loop
    5. Returns results

    Attributes:
        logger: Logger instance
    """

    def __init__(self):
        """Initialize Engine."""
        self.logger = get_logger(__name__)

    def run(
        self,
        strategy: Strategy,
        mode: Literal["backtest", "paper", "live"] = "backtest",
        # Backtest options
        data_path: Optional[Union[str, Path]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        # Execution options
        executor: Optional[Executor] = None,
        cost_model: Optional[CostModel] = None,
        initial_capital: float = 10000.0,
        # Market type and leverage
        market_type: Literal["spot", "futures"] = "spot",
        leverage: float = 1.0,
        # Live options
        exchange: Optional[str] = None,
        interval_seconds: int = 60,
        verbose: bool = False,
    ) -> BacktestResult:
        """Run strategy in specified mode.

        Args:
            strategy: Strategy instance (with universe/data specs)
            mode: "backtest", "paper", or "live"

            # Backtest options
            data_path: Path to historical data (Parquet files)
            start: Backtest start date
            end: Backtest end date

            # Execution options
            executor: Custom executor (auto-created if None)
            cost_model: Cost model for backtest executor
            initial_capital: Starting capital

            # Market type and leverage
            market_type: "spot" or "futures" (spot: no shorts, futures: long/short)
            leverage: Leverage multiplier (1.0 for spot, 1.0-125.0 for futures)

            # Live options
            exchange: Exchange name for live/paper mode
            interval_seconds: Tick interval for live mode
            verbose: Print progress

        Returns:
            BacktestResult with performance metrics
        """
        # Validate market_type and leverage
        if market_type == "spot" and leverage != 1.0:
            raise ValueError(
                f"Spot market does not support leverage (must be 1.0), got {leverage}"
            )
        if leverage < 1.0:
            raise ValueError(f"Leverage must be >= 1.0, got {leverage}")

        self.logger.info(
            f"Engine starting",
            extra={
                "mode": mode,
                "strategy": strategy.__class__.__name__,
                "market_type": market_type,
                "leverage": leverage,
            },
        )

        # 1. Build provider from strategy specs + mode
        provider = self._build_provider(
            strategy=strategy,
            mode=mode,
            data_path=data_path,
            start=start,
            end=end,
            exchange=exchange,
        )

        # 2. Bind provider to strategy
        strategy.bind_provider(provider)

        # 3. Create executor if not provided
        if executor is None:
            executor = self._create_executor(mode, cost_model, exchange)

        # 4. Run based on mode
        if mode == "backtest":
            return self._run_backtest(
                strategy=strategy,
                executor=executor,
                initial_capital=initial_capital,
                market_type=market_type,
                leverage=leverage,
                verbose=verbose,
            )
        else:
            return self._run_live(
                strategy=strategy,
                executor=executor,
                initial_capital=initial_capital,
                mode=mode,
                interval_seconds=interval_seconds,
                verbose=verbose,
            )

    def _build_provider(
        self,
        strategy: Strategy,
        mode: str,
        data_path: Optional[Union[str, Path]],
        start: Optional[datetime],
        end: Optional[datetime],
        exchange: Optional[str],
    ) -> DataProvider:
        """Build DataProvider from strategy specs and mode.

        Builds multi-source DataProvider based on strategy's data specs.
        System clock = OHLCV timeframe (all other data aligned via ffill).

        If strategy already has a declarative DataProvider (with data= parameter),
        the Engine binds sources to it based on mode.

        Args:
            strategy: Strategy with universe/data specs
            mode: "backtest", "paper", or "live"
            data_path: Path for backtest data
            start: Start date
            end: End date
            exchange: Exchange for live mode

        Returns:
            Configured DataProvider
        """
        from clyptq.data.sources.parquet import ParquetSource

        # Check if strategy already has a declarative DataProvider
        existing_provider = getattr(strategy, '_provider', None)

        if existing_provider is not None and hasattr(existing_provider, '_is_declarative'):
            if existing_provider._is_declarative:
                # Use existing provider, bind sources based on mode
                return self._bind_sources_to_provider(
                    provider=existing_provider,
                    mode=mode,
                    data_path=data_path,
                    start=start,
                    end=end,
                    exchange=exchange,
                    warmup=strategy.warmup_periods(),
                )

        # Legacy path: build provider from strategy class attributes
        universe = strategy.universe
        data_specs = strategy.data or {"ohlcv": OHLCVSpec()}
        rebalance_freq = strategy.rebalance_freq

        # Get symbols from universe
        # For dynamic universes (like CryptoLiquid), symbols are determined from data
        symbols = None
        is_dynamic_universe = False

        if universe is not None:
            if hasattr(universe, "symbols") and universe.symbols:
                symbols = universe.symbols
            elif hasattr(universe, "_symbols") and universe._symbols:
                symbols = universe._symbols
            else:
                # Dynamic universe - get symbols from data source
                is_dynamic_universe = True
        else:
            raise ValueError("Strategy must define universe")

        # Get OHLCV timeframe (system clock)
        ohlcv_spec = data_specs.get("ohlcv", OHLCVSpec())
        ohlcv_timeframe = ohlcv_spec.timeframe if hasattr(ohlcv_spec, "timeframe") else "1d"

        # Build provider based on mode
        if mode == "backtest":
            # data_path is optional if OHLCVSpec has exchange/market_type
            # (auto-resolved in _build_sources_from_specs)

            # Build sources from specs
            sources = self._build_sources_from_specs(
                data_specs=data_specs,
                data_path=data_path,  # Can be None, will auto-resolve from spec
                mode="backtest",
            )

            provider = DataProvider(
                universe=universe,
                sources=sources,
                rebalance_freq=rebalance_freq,
                mode="backtest",
            )

            # For dynamic universe, get symbols from source
            if is_dynamic_universe:
                # Get all available symbols from first source
                first_source = list(sources.values())[0]
                symbols = first_source.available_symbols()
                self.logger.info(
                    f"Dynamic universe: loading {len(symbols)} symbols from source"
                )

            provider.load(symbols=symbols, start=start, end=end)

            # For dynamic universe, compute in_universe mask after load
            if is_dynamic_universe:
                universe.compute_in_universe(provider)
                self.logger.info(
                    f"Dynamic universe: {len(universe.get_in_universe_symbols(provider))} symbols in universe"
                )

        else:  # paper or live
            if exchange is None:
                exchange = self._infer_exchange(universe)

            from clyptq.data.collectors.ccxt import CCXTCollector

            collector = CCXTCollector(exchange=exchange)
            max_bars = strategy.warmup_periods() + 100

            # Build sources for live mode
            sources = self._build_sources_from_specs(
                data_specs=data_specs,
                collector=collector,
                mode="live",
            )

            provider = DataProvider(
                universe=universe,
                sources=sources,
                rebalance_freq=rebalance_freq,
                mode="live",
                max_bars=max_bars,
            )
            provider.start_live(symbols=symbols)

        return provider

    def _bind_sources_to_provider(
        self,
        provider: DataProvider,
        mode: str,
        data_path: Optional[Union[str, Path]],
        start: Optional[datetime],
        end: Optional[datetime],
        exchange: Optional[str],
        warmup: int,
    ) -> DataProvider:
        """Bind sources to a declarative DataProvider.

        Args:
            provider: Declarative DataProvider (with data= specs)
            mode: "backtest", "paper", or "live"
            data_path: Path for backtest data
            start: Start date
            end: End date
            exchange: Exchange for live mode
            warmup: Strategy warmup periods

        Returns:
            Provider with sources bound and data loaded
        """
        # Get data specs from provider
        data_specs = provider.data_specs or {"ohlcv": OHLCVSpec()}

        # Get symbols from universe
        # For dynamic universes, symbols are determined from data source
        universe = provider.universe
        symbols = None
        is_dynamic_universe = False

        if universe is not None:
            if hasattr(universe, "symbols") and universe.symbols:
                symbols = universe.symbols
            elif hasattr(universe, "_symbols") and universe._symbols:
                symbols = universe._symbols
            else:
                # Dynamic universe - get symbols from data source
                is_dynamic_universe = True
        else:
            raise ValueError("Provider must define universe")

        if mode == "backtest":
            if data_path is None:
                raise ValueError("data_path required for backtest mode")

            # Build sources from specs
            sources = self._build_sources_from_specs(
                data_specs=data_specs,
                data_path=data_path,
                mode="backtest",
            )

            # For dynamic universe, get symbols from source
            if is_dynamic_universe:
                first_source = list(sources.values())[0]
                symbols = first_source.available_symbols()

            # Bind sources to provider
            provider.bind_sources(sources, mode="backtest")
            provider.load(symbols=symbols, start=start, end=end)

            # For dynamic universe, compute in_universe mask after load
            if is_dynamic_universe:
                universe.compute_in_universe(provider)

        else:  # paper or live
            if exchange is None:
                exchange = self._infer_exchange(universe)

            from clyptq.data.collectors.ccxt import CCXTCollector

            collector = CCXTCollector(exchange=exchange)
            max_bars = warmup + 100

            # Build sources for live mode
            sources = self._build_sources_from_specs(
                data_specs=data_specs,
                collector=collector,
                mode="live",
            )

            # Update max_bars before binding
            provider.max_bars = max_bars

            # Bind sources to provider
            provider.bind_sources(sources, mode="live")
            provider.start_live(symbols=symbols)

        return provider

    def _build_sources_from_specs(
        self,
        data_specs: Dict,
        data_path: Optional[Union[str, Path]] = None,
        collector: Optional[Any] = None,
        mode: str = "backtest",
    ) -> Dict:
        """Build DataSource dict from strategy data specs.

        Automatically resolves data path from OHLCVSpec attributes:
        - exchange: gateio, binance, okx, bybit, upbit
        - market_type: spot, futures, margin
        - timeframe: 1m, 1h, 1d, etc.

        Path resolution:
            {DATA_ROOT}/{market_type}/{exchange}/{timeframe}/
            Example: data/spot/gateio/1d/

        Args:
            data_specs: Dict of {name: DataSpec}
            data_path: Override path (optional, for legacy compatibility)
            collector: Data collector for live mode
            mode: "backtest" or "live"

        Returns:
            Dict of {name: DataSource}
        """
        from clyptq.data.sources.parquet import ParquetSource
        from clyptq.data.sources.live import LiveSource
        from clyptq.data.spec import OHLCVSpec

        sources = {}

        for spec_name, spec in data_specs.items():
            timeframe = getattr(spec, "timeframe", "1d")

            if mode == "backtest":
                # Check if spec has exchange/market_type (OHLCVSpec)
                if isinstance(spec, OHLCVSpec) and data_path is None:
                    # Auto-resolve path from spec attributes
                    exchange = getattr(spec, "exchange", "gateio")
                    market_type = getattr(spec, "market_type", "spot")

                    # Build path: {PROJECT_ROOT}/data/{market_type}/{exchange}/{timeframe}/
                    # PROJECT_ROOT is clyptq package's parent directory
                    import clyptq
                    project_root = Path(clyptq.__file__).parent.parent
                    resolved_path = project_root / "data" / market_type / exchange / timeframe

                    self.logger.debug(
                        f"Auto-resolved data path: {resolved_path}",
                        extra={
                            "spec": spec_name,
                            "exchange": exchange,
                            "market_type": market_type,
                            "timeframe": timeframe,
                        },
                    )

                    sources[spec_name] = ParquetSource(
                        path=resolved_path,
                        timeframe=timeframe,
                    )
                else:
                    # Legacy: use provided data_path
                    sources[spec_name] = ParquetSource(
                        path=data_path,
                        timeframe=timeframe,
                    )
            else:  # live
                # Get exchange from spec for live mode
                exchange = getattr(spec, "exchange", None)
                if exchange and collector is None:
                    from clyptq.data.collectors.ccxt import CCXTCollector
                    collector = CCXTCollector(exchange=exchange)

                sources[spec_name] = LiveSource(
                    collector=collector,
                    timeframe=timeframe,
                )

        return sources

    def _infer_exchange(self, universe) -> str:
        """Infer exchange from universe type."""
        # Check if universe has exchange info
        if hasattr(universe, "exchange"):
            return universe.exchange

        # Check universe name/type for hints
        universe_name = universe.__class__.__name__.lower()
        if "crypto" in universe_name:
            return "binance"
        elif "stock" in universe_name or "equity" in universe_name:
            return "alpaca"

        # Default
        return "binance"

    def _create_executor(
        self,
        mode: str,
        cost_model: Optional[CostModel],
        exchange: Optional[str],
    ) -> Executor:
        """Create appropriate executor for mode."""
        if mode == "backtest":
            if cost_model is None:
                cost_model = CostModel(
                    slippage_bps=5.0,
                    taker_fee=0.001,  # 0.1%
                    maker_fee=0.0005,  # 0.05%
                )
            return BacktestExecutor(cost_model=cost_model)

        elif mode == "paper":
            # Paper mode uses backtest executor with live prices
            if cost_model is None:
                cost_model = CostModel(
                    slippage_bps=5.0,
                    taker_fee=0.001,  # 0.1%
                    maker_fee=0.0005,  # 0.05%
                )
            return BacktestExecutor(cost_model=cost_model)

        else:  # live
            from clyptq.trading.execution.live import LiveExecutor

            if exchange is None:
                exchange = "binance"
            return LiveExecutor(exchange=exchange)

    def _run_backtest(
        self,
        strategy: Strategy,
        executor: Executor,
        initial_capital: float,
        market_type: str = "spot",
        leverage: float = 1.0,
        verbose: bool = False,
    ) -> BacktestResult:
        """Run backtest mode."""
        from clyptq.analytics.performance.metrics import compute_metrics

        provider = strategy.provider
        universe = strategy.universe

        # Check if dynamic universe
        is_dynamic_universe = (
            universe is not None
            and not (hasattr(universe, "symbols") and universe.symbols)
            and not (hasattr(universe, "_symbols") and universe._symbols)
        )

        # Create appropriate portfolio based on market type
        if market_type == "futures":
            portfolio = FuturesPortfolioState(initial_capital, leverage)
        else:
            portfolio = PortfolioState(initial_capital)

        warmup = strategy.warmup_periods()

        snapshots: List[Snapshot] = []
        trades: List[Fill] = []

        self.logger.info(
            "Backtest started",
            extra={
                "symbols": len(provider.symbols),
                "warmup": warmup,
                "rebalance_freq": provider.rebalance_freq,
                "market_type": market_type,
                "leverage": leverage,
                "dynamic_universe": is_dynamic_universe,
            },
        )

        bar_count = 0
        warmup_done = False

        # Track symbols with open positions (for dynamic universe)
        # When a symbol exits universe but has position, we need to close it
        active_symbols: set = set()

        # Main tick loop
        while provider.tick():
            bar_count += 1

            # Skip warmup
            if not warmup_done:
                if bar_count < warmup:
                    continue
                warmup_done = True

            timestamp = provider.current_timestamp
            prices = provider.current_prices()

            if not prices:
                continue

            # Record snapshot
            snapshot = portfolio.get_snapshot(timestamp, prices)
            snapshots.append(snapshot)

            # Check rebalance
            if not provider.should_rebalance():
                continue

            provider.mark_rebalanced()

            # Get weights from strategy
            weights = strategy.on_bar(timestamp=timestamp)

            if not weights:
                weights = {}

            # For dynamic universe, handle symbols entering/exiting universe
            if is_dynamic_universe and universe is not None:
                in_universe_symbols = set(universe.get_in_universe(timestamp))

                # Symbols to consider = union of (current universe + open positions)
                symbols_to_trade = in_universe_symbols | active_symbols

                # Build final weights:
                # - In universe: use strategy weight
                # - Not in universe but has position: weight=0 (close position)
                final_weights = {}
                for symbol in symbols_to_trade:
                    if symbol in in_universe_symbols:
                        # In universe: use strategy weight (default 0 if not computed)
                        final_weights[symbol] = weights.get(symbol, 0.0)
                    else:
                        # Exited universe: close position
                        final_weights[symbol] = 0.0

                weights = final_weights

            if not weights:
                continue

            # Validate weights based on market_type
            self._validate_weights(weights, market_type, timestamp)

            # Convert to orders
            orders = self._weights_to_orders(weights, prices, portfolio)

            if not orders:
                continue

            # Sort orders: SELL first (to free up cash/margin), then BUY
            sell_orders = [o for o in orders if o.side == OrderSide.SELL]
            buy_orders = [o for o in orders if o.side == OrderSide.BUY]
            sorted_orders = sell_orders + buy_orders

            # Execute
            fills = executor.execute(sorted_orders, timestamp, prices)

            # Apply fills
            for fill in fills:
                try:
                    portfolio.apply_fill(fill)
                    trades.append(fill)
                except ValueError as e:
                    self.logger.warning(f"Fill rejected: {e}")

            # Update active_symbols based on current positions
            if is_dynamic_universe:
                active_symbols = {
                    s for s, pos in portfolio.positions.items()
                    if abs(pos.amount) > 1e-10
                }

            if verbose and bar_count % 100 == 0:
                self.logger.info(
                    f"Progress: bar {bar_count}, equity {snapshot.equity:.2f}"
                )

        # Compute metrics
        if snapshots:
            metrics = compute_metrics(snapshots, trades)
        else:
            metrics = self._empty_metrics()

        self.logger.info(
            "Backtest completed",
            extra={
                "total_bars": bar_count,
                "total_trades": len(trades),
                "final_equity": snapshots[-1].equity if snapshots else 0,
            },
        )

        return BacktestResult(
            snapshots=snapshots,
            trades=trades,
            metrics=metrics,
            strategy_name=strategy._name,
            mode=EngineMode.BACKTEST,
        )

    def _run_live(
        self,
        strategy: Strategy,
        executor: Executor,
        initial_capital: float,
        mode: str,
        interval_seconds: int,
        verbose: bool,
    ) -> BacktestResult:
        """Run live/paper mode."""
        import time

        from clyptq.infra.utils import GracefulShutdown

        provider = strategy.provider
        portfolio = PortfolioState(initial_capital)
        shutdown = GracefulShutdown()

        snapshots: List[Snapshot] = []
        trades: List[Fill] = []

        engine_mode = EngineMode.LIVE if mode == "live" else EngineMode.PAPER

        self.logger.info(
            f"{mode.upper()} trading started",
            extra={
                "symbols": provider.symbols,
                "interval_seconds": interval_seconds,
            },
        )

        try:
            while not shutdown.is_shutdown_requested():
                timestamp = datetime.utcnow()

                prices = provider.current_prices()
                if not prices:
                    time.sleep(interval_seconds)
                    continue

                # Record snapshot
                snapshot = portfolio.get_snapshot(timestamp, prices)
                snapshots.append(snapshot)

                # Check rebalance
                if not provider.should_rebalance():
                    time.sleep(interval_seconds)
                    continue

                provider.mark_rebalanced()

                # Get weights
                weights = strategy.on_bar(timestamp=timestamp)

                if not weights:
                    time.sleep(interval_seconds)
                    continue

                # Convert to orders
                orders = self._weights_to_orders(weights, prices, portfolio)

                if orders:
                    fills = executor.execute(orders, timestamp, prices)

                    for fill in fills:
                        try:
                            portfolio.apply_fill(fill)
                            trades.append(fill)
                            if verbose:
                                self.logger.info(
                                    f"{fill.side.name} {fill.symbol}: "
                                    f"{fill.amount:.6f} @ ${fill.price:,.2f}"
                                )
                        except ValueError as e:
                            self.logger.warning(f"Fill rejected: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")

        finally:
            provider.stop_live()

        # Return results
        from clyptq.analytics.performance.metrics import compute_metrics

        metrics = compute_metrics(snapshots, trades) if snapshots else self._empty_metrics()

        return BacktestResult(
            snapshots=snapshots,
            trades=trades,
            metrics=metrics,
            strategy_name=strategy._name,
            mode=engine_mode,
        )

    def _validate_weights(
        self,
        weights: Dict[str, float],
        market_type: str,
        timestamp: datetime,
    ) -> None:
        """Validate weights based on market type.

        Args:
            weights: Target weights {symbol: weight}
            market_type: "spot" or "futures"
            timestamp: Current timestamp (for error message)

        Raises:
            ValueError: If spot weights contain negative values (short positions)
        """
        if market_type == "spot":
            negative_weights = {s: w for s, w in weights.items() if w < -1e-8}
            if negative_weights:
                raise ValueError(
                    f"Spot market does not allow short positions (negative weights). "
                    f"At {timestamp}: {negative_weights}. "
                    f"Use rank() + l1_norm() instead of demean() for spot strategies."
                )

        # Validate total exposure
        abs_sum = sum(abs(w) for w in weights.values())
        if abs_sum > 1.0 + 1e-6:
            self.logger.warning(
                f"Weights abs sum ({abs_sum:.4f}) exceeds 1.0 at {timestamp}. "
                f"Consider using l1_norm() to normalize."
            )

    def _weights_to_orders(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
        portfolio: Union[PortfolioState, FuturesPortfolioState],
    ) -> List[Order]:
        """Convert target weights to orders."""
        import math

        orders = []
        equity = portfolio.equity(prices)

        for symbol, target_weight in weights.items():
            if symbol not in prices:
                continue

            price = prices[symbol]

            # Skip NaN weight or price (can happen during warmup or missing data)
            if math.isnan(target_weight) or math.isnan(price):
                continue

            target_value = equity * target_weight
            target_amount = target_value / price

            current_amount = 0.0
            if symbol in portfolio.positions:
                current_amount = portfolio.positions[symbol].amount

            delta = target_amount - current_amount

            if abs(delta) < 1e-8:
                continue

            if delta > 0:
                orders.append(Order(symbol=symbol, side=OrderSide.BUY, amount=delta))
            else:
                orders.append(Order(symbol=symbol, side=OrderSide.SELL, amount=abs(delta)))

        return orders

    def _empty_metrics(self):
        """Create empty metrics."""
        from clyptq.core.types import PerformanceMetrics

        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_leverage=0.0,
            max_leverage=0.0,
            avg_num_positions=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            duration_days=0,
        )
