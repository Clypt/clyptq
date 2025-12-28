"""
Factor computation caching for performance optimization.

Provides caching layer to avoid recomputing expensive factors when
market data hasn't changed significantly.
"""

import hashlib
from datetime import datetime
from typing import Dict, Optional

from clypt.data.store import DataView
from clypt.factors.base import Factor
from clypt.types import CacheEntry, CacheStats


class FactorCache:
    """
    LRU cache for factor computation results.

    Caches factor scores to avoid recomputation when market data
    is similar to previous computations.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize factor cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    def _compute_hash(self, factor: Factor, data: DataView) -> str:
        """
        Compute hash key for cache lookup.

        Args:
            factor: Factor being computed
            data: DataView for computation

        Returns:
            Hash key string
        """
        # Create hash from factor name, timestamp, and symbols
        hash_input = f"{factor.name}:{data.timestamp.isoformat()}:{sorted(data.symbols)}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def get(
        self, factor: Factor, data: DataView
    ) -> Optional[Dict[str, float]]:
        """
        Get cached factor scores if available.

        Args:
            factor: Factor to lookup
            data: DataView for computation

        Returns:
            Cached scores or None if not found/expired
        """
        cache_key = self._compute_hash(factor, data)

        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Check TTL
            age = (datetime.now() - entry.timestamp).total_seconds()
            if age < self.ttl_seconds:
                # Cache hit
                self._stats.hits += 1
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                return entry.data

            # Expired, remove
            del self._cache[cache_key]
            self._stats.evictions += 1

        # Cache miss
        self._stats.misses += 1
        self._stats.update_hit_rate()
        return None

    def put(
        self, factor: Factor, data: DataView, scores: Dict[str, float]
    ) -> None:
        """
        Store factor scores in cache.

        Args:
            factor: Factor being cached
            data: DataView used for computation
            scores: Computed scores to cache
        """
        cache_key = self._compute_hash(factor, data)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Store entry
        self._cache[cache_key] = CacheEntry(
            timestamp=datetime.now(),
            data=scores.copy(),
            hash_key=cache_key,
            last_accessed=datetime.now(),
        )

        self._stats.total_size = len(self._cache)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or datetime.min,
        )

        # Remove it
        del self._cache[lru_key]
        self._stats.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        evicted = len(self._cache)
        self._cache.clear()
        self._stats.evictions += evicted
        self._stats.total_size = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_size = len(self._cache)
        self._stats.update_hit_rate()
        return self._stats

    def __len__(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)


class CachedFactor(Factor):
    """
    Wrapper that adds caching to any factor.

    Automatically caches factor computation results to improve performance.
    """

    def __init__(
        self,
        base_factor: Factor,
        cache: Optional[FactorCache] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize cached factor.

        Args:
            base_factor: Underlying factor to cache
            cache: Shared cache instance (creates new if None)
            cache_enabled: Enable/disable caching
        """
        super().__init__(name=f"Cached_{base_factor.name}")
        self.base_factor = base_factor
        self.cache = cache or FactorCache()
        self.cache_enabled = cache_enabled

    def compute(self, data: DataView) -> Dict[str, float]:
        """
        Compute factor scores with caching.

        Args:
            data: DataView at current timestamp

        Returns:
            Dictionary of {symbol: score}
        """
        if not self.cache_enabled:
            return self.base_factor.compute(data)

        # Try cache first
        cached_scores = self.cache.get(self.base_factor, data)
        if cached_scores is not None:
            return cached_scores

        # Cache miss - compute
        scores = self.base_factor.compute(data)

        # Store in cache
        self.cache.put(self.base_factor, data, scores)

        return scores

    def clear_cache(self) -> None:
        """Clear this factor's cache."""
        self.cache.clear()

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()


def cached(factor: Factor, cache: Optional[FactorCache] = None) -> CachedFactor:
    """
    Convenience function to wrap a factor with caching.

    Args:
        factor: Factor to cache
        cache: Shared cache instance

    Returns:
        CachedFactor instance

    Example:
        >>> momentum = MomentumFactor(lookback=20)
        >>> cached_momentum = cached(momentum)
    """
    return CachedFactor(factor, cache)
