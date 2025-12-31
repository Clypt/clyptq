import asyncio
import signal
from typing import Callable, Optional

from clyptq.infra.utils import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    def __init__(self):
        self._shutdown_requested = False
        self._shutdown_callbacks: list[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        self._shutdown_callbacks.append(callback)

    def request_shutdown(self) -> None:
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info("Shutdown requested, executing callbacks")

        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error("Error during shutdown callback", extra={"error": str(e)})

        logger.info("Graceful shutdown completed")

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested


_global_shutdown_handler: Optional[GracefulShutdown] = None


def get_shutdown_handler() -> GracefulShutdown:
    global _global_shutdown_handler
    if _global_shutdown_handler is None:
        _global_shutdown_handler = GracefulShutdown()
    return _global_shutdown_handler


def setup_signal_handlers(shutdown_handler: Optional[GracefulShutdown] = None) -> None:
    if shutdown_handler is None:
        shutdown_handler = get_shutdown_handler()

    def signal_handler(signum, frame):
        logger.info("Signal received", extra={"signal": signum})
        shutdown_handler.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered")
