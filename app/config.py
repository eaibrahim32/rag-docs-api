"""Centralised configuration. All values overridable via environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- app ---
    app_name: str = "rag-docs-api"
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- storage ---
    sql_url: str = "sqlite+aiosqlite:///./data/app.db"
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "ragdocs"
    redis_url: str = "redis://localhost:6379/0"
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "documents"
    upload_dir: str = "./data/uploads"

    # --- rag ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    # onnx = all-MiniLM-L6-v2 via ONNX Runtime (no torch). sentence_transformers
    # is an optional extra install for GPU hosts. hash is for tests.
    embedding_backend: Literal["onnx", "sentence_transformers", "hash"] = "onnx"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # ST backend only

    # --- llm ---
    llm_backend: Literal["ollama", "openai"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0

    # --- cache ---
    cache_ttl_s: int = 3600
    cache_enabled: bool = True

    # --- limits ---
    max_upload_mb: int = Field(default=25, ge=1, le=200)


@lru_cache
def get_settings() -> Settings:
    return Settings()
