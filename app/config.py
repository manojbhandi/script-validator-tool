from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM provider
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.6-flash"

    # Embeddings provider
    embedding_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768

    # Database
    supabase_db_url: str = ""

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.68

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
