import redis.asyncio as redis
import hashlib
import json
from app.config import get_settings

settings = get_settings()
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

DEFAULT_TTL = settings.cache_ttl_seconds


def _cache_key(query: str, doc_ids: list[str] | None = None) -> str:
    raw = query.strip().lower() + "|" + ",".join(sorted(doc_ids or []))
    return "cache:" + hashlib.md5(raw.encode()).hexdigest()


async def get_cached_response(query: str, doc_ids: list[str] | None = None) -> dict | None:
    key = _cache_key(query, doc_ids)
    cached = await redis_client.get(key)
    return json.loads(cached) if cached else None


async def set_cached_response(query: str, response: dict, doc_ids: list[str] | None = None, ttl: int = DEFAULT_TTL):
    key = _cache_key(query, doc_ids)
    await redis_client.set(key, json.dumps(response), ex=ttl)


async def get_cache_stats() -> dict:
    info = await redis_client.info("stats")
    return {
        "keyspace_hits": info.get("keyspace_hits", 0),
        "keyspace_misses": info.get("keyspace_misses", 0),
    }