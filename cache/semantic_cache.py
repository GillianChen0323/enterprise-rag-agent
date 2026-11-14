"""Redis-backed query caching to avoid repeated LLM calls. Keys include use case, tenant
ID, and normalized query hash; TTL varies by use case. This implementation uses exact
matching, NOT semantic similarity. Embedding-based cache lookup would require a separate
implementation."""

import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class SemanticCache:
    """Redis-backed query cache. Key namespace:
    rag:cache:{use_case}:{tenant_id}:{query_hash}."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "rag:cache",
        default_ttl: int = 300,
    ):
        self.redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            # Connection pool settings.
            max_connections=50,
        )
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

        # Cache hit-rate counters.
        self._hits = 0
        self._misses = 0

    def _build_key(self, query: str, use_case: str, tenant_id: str) -> str:
        """Normalize query whitespace and case, hash with MD5, and build
        prefix:use_case:tenant_id:hash."""
        normalized = query.strip().lower()
        # Normalize repeated whitespace.
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        query_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{use_case}:{tenant_id}:{query_hash}"

    async def get(
        self,
        query: str,
        use_case: str,
        tenant_id: str,
    ) -> Optional[dict]:
        """Return the cached dictionary on a hit, or None on a miss."""
        key = self._build_key(query, use_case, tenant_id)
        try:
            value = await self.redis.get(key)
            if value:
                self._hits += 1
                hit_rate = self._hits / (self._hits + self._misses) * 100
                logger.debug(f"Cache HIT | key={key[-16:]} | "
                             f"hit_rate={hit_rate:.1f}%")
                return json.loads(value)
            else:
                self._misses += 1
                return None
        except Exception as e:
            # Fall back to retrieval if the cache fails.
            logger.warning(f"Cache GET error: {e}")
            return None

    async def set(
        self,
        query: str,
        use_case: str,
        tenant_id: str,
        result: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Write a cache entry. An explicit TTL overrides the default. Write failures do
        not interrupt the main pipeline."""
        key = self._build_key(query, use_case, tenant_id)
        effective_ttl = ttl or self.default_ttl

        # Skip empty or error results.
        if not result.get("answer"):
            return False

        # Exclude from_cache from stored results.
        cache_data = {k: v for k, v in result.items() if k != "from_cache"}

        try:
            serialized = json.dumps(cache_data, ensure_ascii=False)
            await self.redis.setex(key, effective_ttl, serialized)
            logger.debug(f"Cache SET | key={key[-16:]} | ttl={effective_ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")
            return False

    async def invalidate(
        self,
        use_case: str,
        tenant_id: str,
    ) -> int:
        """Invalidate all cache entries for a tenant after document updates. Return the
        number of deleted keys."""
        pattern = f"{self.key_prefix}:{use_case}:{tenant_id}:*"
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cache invalidated | pattern={pattern} | "
                            f"deleted={deleted} keys")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0

    async def invalidate_all(self) -> int:
        """Clear all RAG cache entries. Use with care."""
        pattern = f"{self.key_prefix}:*"
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Return cache hit-rate statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0,
        }

    async def close(self):
        """Close the Redis connection during shutdown."""
        await self.redis.aclose()
