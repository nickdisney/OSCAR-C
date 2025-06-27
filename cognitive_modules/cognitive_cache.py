
import asyncio
import time
import sys # For calculating approximate size
import logging
from typing import Dict, Any, Optional, Callable, Hashable

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("CognitiveCache: Relative import failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    CognitiveComponent = globals().get('CognitiveComponent')


logger_cognitive_cache = logging.getLogger(__name__) # Use standard module logger name

class CognitiveCache(CognitiveComponent): # Correctly inherit
    """TTL-based cache for storing results of expensive computations."""

    def __init__(self):
        self._cache: Dict[Hashable, Any] = {}
        self._timestamps: Dict[Hashable, float] = {}
        self._ttl: float = 1.0 # Default TTL, can be overridden by config
        self._hits: int = 0
        self._misses: int = 0
        self._lock = asyncio.Lock() # Protect cache access
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize cache with configuration."""
        self._controller = controller
        self._config = config.get("cognitive_cache", {}) # Get cache specific config
        self._ttl = self._config.get("default_ttl", 1.0) # Get TTL from config

        if self._ttl <= 0:
            logger_cognitive_cache.warning("CognitiveCache: default_ttl must be positive. Using default 1.0s.")
            self._ttl = 1.0

        logger_cognitive_cache.info(f"CognitiveCache initialized with default TTL: {self._ttl}s")
        return True

    async def get_or_compute(self, key: Hashable, compute_func: Callable[[], Any], ttl_override: Optional[float] = None) -> Any:
        """
        Get a value from the cache if it's fresh, otherwise compute it, cache it, and return it.
        Allows overriding the default TTL for specific keys.
        The compute_func should be a callable (sync or async) that takes no arguments.
        """
        current_time = time.time()
        effective_ttl = ttl_override if ttl_override is not None and ttl_override > 0 else self._ttl

        async with self._lock:
            # Check if value exists and is fresh
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if (current_time - timestamp) < effective_ttl:
                    self._hits += 1
                    logger_cognitive_cache.debug(f"Cache hit for key: {key}")
                    return self._cache[key]
                else:
                    # Cache entry expired
                    logger_cognitive_cache.debug(f"Cache expired for key: {key}")
                    # Remove expired entry (optional, or let it be overwritten)
                    # del self._cache[key]
                    # del self._timestamps[key]

            # If key not in cache or expired
            self._misses += 1
            logger_cognitive_cache.debug(f"Cache miss for key: {key}")

        # --- Computation happens outside the lock ---
        # This prevents blocking other cache accesses during potentially long computations.
        try:
            logger_cognitive_cache.debug(f"Computing value for key: {key}")
            start_compute_time = time.monotonic()
            if asyncio.iscoroutinefunction(compute_func):
                result = await compute_func()
            else:
                # Run synchronous function in executor to avoid blocking event loop
                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, compute_func)
                except RuntimeError as e: # Handle cases where loop might not be running (e.g., during shutdown tests)
                    logger_cognitive_cache.warning(f"Could not get running loop for sync func '{key}', running directly: {e}")
                    result = compute_func() # Run directly as fallback

            compute_duration = time.monotonic() - start_compute_time
            logger_cognitive_cache.debug(f"Computed value for key {key} in {compute_duration:.4f}s")

        except Exception as e:
            logger_cognitive_cache.exception(f"Error computing value for cache key '{key}': {e}")
            raise # Re-raise the exception so the caller knows computation failed

        # --- Store the result back in the cache, under lock ---
        async with self._lock:
            self._cache[key] = result
            self._timestamps[key] = time.time() # Use current time for the new entry timestamp

        return result

    async def get(self, key: Hashable) -> Optional[Any]:
         """Get value if present and not expired, otherwise return None."""
         async with self._lock:
             current_time = time.time()
             if key in self._cache and (current_time - self._timestamps.get(key, 0)) < self._ttl:
                 return self._cache[key]
         return None

    async def put(self, key: Hashable, value: Any, ttl_override: Optional[float] = None):
        """Explicitly put a value into the cache."""
        effective_ttl = ttl_override if ttl_override is not None and ttl_override > 0 else self._ttl
        # Ensure TTL is positive
        if effective_ttl <= 0:
             logger_cognitive_cache.warning(f"Ignoring put operation for key {key} due to non-positive TTL: {effective_ttl}")
             return

        async with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
            logger_cognitive_cache.debug(f"Explicitly added/updated cache key: {key}")


    async def clear(self):
        """Clear the entire cache."""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0 # Reset stats as well
            self._misses = 0
        logger_cognitive_cache.info("CognitiveCache cleared.")

    # --- CognitiveComponent Implementation ---

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Cache itself doesn't process, it's used by other components."""
        return None

    async def reset(self) -> None:
        """Reset the cache by clearing it."""
        await self.clear()

    async def get_status(self) -> Dict[str, Any]:
        """Get cache performance statistics and status."""
        async with self._lock:
            total_calls = self._hits + self._misses
            hit_rate = (self._hits / total_calls * 100) if total_calls > 0 else 0.0
            cached_items_count = len(self._cache)

            # Estimate cache size (can be inaccurate for complex objects)
            size_bytes = -1
            size_mb = -1.0
            try:
                # Sum sizes of keys and values separately
                size_bytes = sum(sys.getsizeof(k) for k in self._cache.keys()) + \
                             sum(sys.getsizeof(v) for v in self._cache.values())
                size_mb = round(size_bytes / (1024 * 1024), 3)
            except Exception as e:
                logger_cognitive_cache.warning(f"Could not estimate cache size: {e}")

            status = {
                "component": "CognitiveCache",
                "status": "operational",
                "default_ttl_s": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "total_calls": total_calls,
                "hit_rate_percent": round(hit_rate, 2),
                "cached_items_count": cached_items_count,
                "estimated_size_bytes": size_bytes, # Added bytes
                "estimated_size_mb": size_mb
            }
        return status

    async def shutdown(self) -> None:
        """Perform cleanup (optional for this component)."""
        logger_cognitive_cache.info("CognitiveCache shutting down (clearing cache).")
        await self.clear()