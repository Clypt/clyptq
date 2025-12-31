import asyncio
import time
from typing import Optional


class RateLimiter:
    def __init__(self, max_requests: int, window: float = 60.0):
        self.max_requests = max_requests
        self.window = window
        self.tokens = max_requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_update

        if elapsed > 0:
            refill_amount = (elapsed / self.window) * self.max_requests
            self.tokens = min(self.max_requests, self.tokens + refill_amount)
            self.last_update = now

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                sleep_time = (tokens - self.tokens) / (self.max_requests / self.window)
                await asyncio.sleep(sleep_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def available_tokens(self) -> float:
        self._refill()
        return self.tokens
