from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from urllib.parse import urlparse, urlunparse
from app.config import get_settings

_s = get_settings()
db_url = _s.database_url

# --- Normalize Neon URL for asyncpg ---
# Neon gives URLs like: postgresql://user:pass@host/db?sslmode=require&channel_binding=...
# asyncpg only understands ?ssl=require — everything else causes crashes.
# So: fix the scheme, strip ALL query params, add only ?ssl=require.
parsed = urlparse(db_url)
clean_scheme = "postgresql+asyncpg"
clean_url = urlunparse(parsed._replace(scheme=clean_scheme, query="ssl=require"))

engine = create_async_engine(clean_url, pool_pre_ping=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session