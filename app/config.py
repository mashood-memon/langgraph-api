"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

class Settings(BaseSettings):
    
    # LLM Configuration
    openai_api_key: str
    gemini_api_key: str
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "gemini-2.5-flash"
    
    # Vector DB + Embeddings
    qdrant_url: str
    qdrant_api_key: str 
    voyage_api_key: str
    
    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "prod-app"
    
    
    # Application
    app_env: str = "development" 
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    relevance_threshold: float = 0.5
    top_k_chunks: int = 10
    
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()