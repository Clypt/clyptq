"""CCXT-based data collector for crypto exchanges.

Handles both historical and live data collection from CCXT-compatible exchanges.
Combines functionality from CCXTLoader (historical) and CCXTStreamingSource (live).
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import ccxt
import pandas as pd

from clyptq.data.collectors.base import DataCollector, Subscription
from clyptq.infra.utils import get_logger


class CCXTCollector(DataCollector):
    """CCXT-based collector for OHLCV data.

    Supports both:
    - Historical: Bulk download via REST API
    - Live: Real-time streaming via polling (or websocket with ccxt.pro)

    Example:
        ```python
        collector = CCXTCollector("binance")

        # Historical
        data = collector.collect_historical(
            symbols=["BTC/USDT", "ETH/USDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 1),
            timeframe="1h"
        )

        # Live
        def on_data(symbol, candle):
            print(f"{symbol}: {candle}")

        sub = collector.subscribe(["BTC/USDT"], on_data, timeframe="1m")
        # ... later
        sub.unsubscribe()
        ```
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
    ):
        """Initialize CCXT collector.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: API key for authenticated endpoints
            api_secret: API secret
            sandbox: Use sandbox/testnet mode
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.logger = get_logger(
            __name__, context={"exchange": exchange_id, "sandbox": sandbox}
        )

        self._exchange: Optional[ccxt.Exchange] = None
        self._async_exchange = None  # For live streaming
        self._markets_loaded = False

    @property
    def name(self) -> str:
        """Collector name."""
        return f"ccxt:{self.exchange_id}"

    @property
    def supported_fields(self) -> List[str]:
        """Supported OHLCV fields."""
        return ["open", "high", "low", "close", "volume"]

    def _ensure_exchange(self) -> ccxt.Exchange:
        """Ensure exchange is initialized."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                "enableRateLimit": True,
                "sandbox": self.sandbox,
            }
            if self.api_key:
                config["apiKey"] = self.api_key
            if self.api_secret:
                config["secret"] = self.api_secret

            self._exchange = exchange_class(config)

        if not self._markets_loaded:
            try:
                self._exchange.load_markets()
                self._markets_loaded = True
                self.logger.info(
                    "Markets loaded",
                    extra={"market_count": len(self._exchange.markets)},
                )
            except ccxt.NetworkError as e:
                self.logger.error("Network error loading markets", extra={"error": str(e)})
                raise
            except ccxt.ExchangeError as e:
                self.logger.error("Exchange error loading markets", extra={"error": str(e)})
                raise

        return self._exchange

    def collect_historical(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1h",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Collect historical OHLCV data.

        Args:
            symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe (e.g., "1h", "4h", "1d")
            **kwargs: Additional options (limit per request)

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data
        """
        exchange = self._ensure_exchange()
        limit = kwargs.get("limit", 1000)
        result = {}

        for symbol in symbols:
            try:
                df = self._fetch_ohlcv(exchange, symbol, timeframe, start, end, limit)
                if df is not None and not df.empty:
                    result[symbol] = df
                    self.logger.info(
                        "Historical data collected",
                        extra={"symbol": symbol, "bars": len(df)},
                    )
            except Exception as e:
                self.logger.error(
                    "Failed to collect historical data",
                    extra={"symbol": symbol, "error": str(e)},
                )
                continue

        return result

    def _fetch_ohlcv(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with pagination."""
        all_ohlcv = []
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while since_ms < end_ms:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                )

                if not ohlcv:
                    break

                # Filter to end date
                ohlcv = [candle for candle in ohlcv if candle[0] <= end_ms]
                all_ohlcv.extend(ohlcv)

                if len(ohlcv) < limit:
                    break

                since_ms = ohlcv[-1][0] + 1

            except ccxt.NetworkError as e:
                self.logger.error(
                    "Network error fetching OHLCV",
                    extra={"symbol": symbol, "error": str(e)},
                )
                raise
            except ccxt.ExchangeError as e:
                self.logger.error(
                    "Exchange error fetching OHLCV",
                    extra={"symbol": symbol, "error": str(e)},
                )
                raise

        if not all_ohlcv:
            return None

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def subscribe(
        self,
        symbols: List[str],
        on_data: Callable[[str, Dict[str, Any]], None],
        timeframe: str = "1h",
        **kwargs,
    ) -> Subscription:
        """Subscribe to live OHLCV stream.

        Uses async polling. For production, consider ccxt.pro for websockets.

        Args:
            symbols: Symbols to subscribe
            on_data: Callback (symbol, ohlcv_dict)
            timeframe: Candle timeframe
            **kwargs: poll_interval (default 1.0s)

        Returns:
            Subscription handle
        """
        poll_interval = kwargs.get("poll_interval", 1.0)

        # Create streaming task
        loop = asyncio.new_event_loop()
        cancel_event = asyncio.Event()

        def run_stream():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._stream_loop(symbols, on_data, timeframe, poll_interval, cancel_event)
            )

        import threading

        thread = threading.Thread(target=run_stream, daemon=True)
        thread.start()

        def cancel():
            loop.call_soon_threadsafe(cancel_event.set)

        return Subscription(self, symbols, cancel)

    async def _stream_loop(
        self,
        symbols: List[str],
        on_data: Callable[[str, Dict[str, Any]], None],
        timeframe: str,
        poll_interval: float,
        cancel_event: asyncio.Event,
    ) -> None:
        """Async streaming loop."""
        import aiohttp
        import ccxt.async_support as ccxt_async

        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)

        exchange_class = getattr(ccxt_async, self.exchange_id)
        config = {
            "enableRateLimit": True,
            "session": session,
        }
        if self.api_key:
            config["apiKey"] = self.api_key
        if self.api_secret:
            config["secret"] = self.api_secret
        if self.sandbox:
            config["sandbox"] = True

        exchange = exchange_class(config)

        try:
            await exchange.load_markets()

            while not cancel_event.is_set():
                for symbol in symbols:
                    try:
                        ohlcv = await exchange.fetch_ohlcv(
                            symbol, timeframe, limit=1
                        )
                        if ohlcv:
                            candle = ohlcv[-1]
                            on_data(
                                symbol,
                                {
                                    "timestamp": datetime.utcfromtimestamp(
                                        candle[0] / 1000
                                    ),
                                    "open": candle[1],
                                    "high": candle[2],
                                    "low": candle[3],
                                    "close": candle[4],
                                    "volume": candle[5],
                                },
                            )
                    except Exception as e:
                        self.logger.warning(
                            "Stream fetch error",
                            extra={"symbol": symbol, "error": str(e)},
                        )

                await asyncio.sleep(poll_interval)

        finally:
            await exchange.close()
            await session.close()

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols against exchange markets."""
        exchange = self._ensure_exchange()
        valid = []
        for symbol in symbols:
            if symbol in exchange.markets:
                valid.append(symbol)
            else:
                self.logger.warning(
                    "Invalid symbol skipped", extra={"symbol": symbol}
                )
        return valid

    def get_available_symbols(self, quote: str = "USDT") -> List[str]:
        """Get available trading pairs for a quote currency."""
        exchange = self._ensure_exchange()
        return sorted(
            symbol
            for symbol in exchange.markets
            if symbol.endswith(f"/{quote}")
        )

    def close(self) -> None:
        """Close exchange connection."""
        if self._exchange and hasattr(self._exchange, "close"):
            self._exchange.close()
            self._exchange = None
            self._markets_loaded = False
