"""Caching utilities for performance optimization."""

import hashlib
from functools import wraps
from typing import Any, Callable, Dict, Optional


class QueryCache:
    """Simple in-memory cache for query results."""

    def __init__(self, max_size: int = 100):
        """Initialize cache with maximum size."""
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value

    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def cache_query_result(cache: QueryCache):
    """Decorator to cache query results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function args
            key_str = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper

    return decorator


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_str = f"{prefix}:{str(args)}:{str(kwargs)}"
    return hashlib.md5(key_str.encode()).hexdigest()
