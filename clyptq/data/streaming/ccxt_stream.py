"""CCXT-based streaming data source (async polling)."""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import aiohttp
import ccxt.async_support as ccxt

from clyptq.data.streaming.base import StreamingDataSource


class CCXTStreamingSource(StreamingDataSource):
    """
    Async streaming via fast polling.

    Uses CCXT's async API for concurrent price fetching.
    Future: swap to ccxt.pro for real WebSocket.
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
        poll_interval: float = 1.0,
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.poll_interval = poll_interval

        self._exchange: Optional[ccxt.Exchange] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def __del__(self):
        """Cleanup on deletion - warn if not properly closed."""
        if self._exchange is not None:
            import warnings
            warnings.warn(
                f"CCXTStreamingSource for {self.exchange_id} was not properly closed. "
                "Call await stream.stop() before discarding.",
                ResourceWarning
            )

    async def start(
        self, symbols: List[str], on_tick: Callable[[datetime, Dict[str, float]], None]
    ) -> None:
        """Start streaming."""
        if self._running:
            raise RuntimeError("Already running")

        # Create custom aiohttp session with ThreadedResolver
        # This fixes DNS issues with hotspot/tethering IPv6 DNS servers
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        self._session = aiohttp.ClientSession(connector=connector)

        exchange_class = getattr(ccxt, self.exchange_id)
        config = {
            "enableRateLimit": True,
            "session": self._session,  # Use custom session with system DNS
        }

        if self.api_key and self.api_secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret

        if self.sandbox:
            config["sandbox"] = True

        self._exchange = exchange_class(config)
        self._running = True  # Set running before async operations

        try:
            await self._exchange.load_markets()
        except Exception:
            # Cleanup on initialization failure
            self._running = False
            await self._exchange.close()
            self._exchange = None
            await self._session.close()
            self._session = None
            raise

        self._task = asyncio.create_task(self._stream_loop(symbols, on_tick))

    async def stop(self) -> None:
        """Stop streaming."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._exchange:
            await self._exchange.close()
            self._exchange = None

        if self._session:
            await self._session.close()
            self._session = None

    def is_running(self) -> bool:
        """Check if running."""
        return self._running

    async def _stream_loop(
        self, symbols: List[str], on_tick: Callable[[datetime, Dict[str, float]], None]
    ) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                timestamp = datetime.now(timezone.utc)
                prices = await self._fetch_prices_async(symbols)

                if prices:
                    on_tick(timestamp, prices)

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                await asyncio.sleep(self.poll_interval * 2)

    async def _fetch_prices_async(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch prices concurrently."""
        if not self._exchange:
            return {}

        tasks = [self._fetch_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                continue
            if result and "last" in result and result["last"]:
                prices[symbol] = float(result["last"])

        return prices

    async def _fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch single ticker."""
        if not self._exchange:
            return None

        try:
            return await self._exchange.fetch_ticker(symbol)
        except Exception:
            return None
