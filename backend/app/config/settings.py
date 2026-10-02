"""Application settings loaded from environment variables / `.env`.

API keys are stored as `SecretStr` so they never appear in logs, reprs or
serialized output. All relative paths are resolved against the project root,
so the app behaves the same regardless of the current working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SQLITE_PREFIX = "sqlite:///"


class Settings(BaseSettings):
    """Typed, validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API keys (optional at startup so the app can report what's missing) ---
    gemini_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # --- Gemini (free-tier Flash by default) ---
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Free-tier quota protection ---
    gemini_max_rpm: int = Field(default=8, ge=1, le=60)
    max_revisions: int = Field(default=1, ge=0, le=3)
    max_research_tool_rounds: int = Field(default=2, ge=1, le=5)
    tavily_max_results: int = Field(default=5, ge=1, le=20)
    tavily_search_depth: Literal["basic", "advanced"] = "basic"  # basic = 1 credit/search
    max_web_searches: int = Field(default=3, ge=1, le=20)  # per research iteration
    writer_max_sources: int = Field(default=14, ge=5, le=30)  # best sources passed to the writer
    max_research_iterations: int = Field(default=2, ge=1, le=5)  # "Modify research" limit
    llm_timeout_seconds: int = Field(default=120, ge=10, le=600)
    rag_top_k: int = Field(default=4, ge=1, le=20)

    # --- Knowledge base / RAG ---
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    embedding_batch_size: int = Field(default=50, ge=1, le=100)
    embedding_max_rpm: int = Field(default=30, ge=1, le=1000)
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    # gemini-embedding-001 scores unrelated text ~0.45-0.55 and relevant text 0.6+
    rag_min_relevance: float = Field(default=0.6, ge=0.0, le=1.0)
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    # --- Storage ---
    database_url: str = f"{SQLITE_PREFIX}data/researchpilot.db"
    checkpoint_db: Path = Path("data/checkpoints.db")  # LangGraph run state (resumable)
    chroma_dir: Path = Path("data/chroma")
    upload_dir: Path = Path("data/uploads")
    reports_dir: Path = Path("reports")

    # --- Services ---
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    backend_url: str = "http://127.0.0.1:8000"

    @field_validator("gemini_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        """Treat empty values (e.g. `GEMINI_API_KEY=` in .env) as not set."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("gemini_model", "gemini_embedding_model")
    @classmethod
    def _strip_models_prefix(cls, value: str) -> str:
        """Accept both `gemini-2.5-flash` and `models/gemini-2.5-flash`."""
        value = value.strip()
        if not value:
            raise ValueError("model name must not be empty")
        return value.removeprefix("models/")

    @field_validator("checkpoint_db", "chroma_dir", "upload_dir", "reports_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("database_url")
    @classmethod
    def _resolve_sqlite_path(cls, value: str) -> str:
        """Make relative SQLite paths absolute; leave other URLs untouched."""
        if value.startswith(SQLITE_PREFIX) and value != f"{SQLITE_PREFIX}:memory:":
            raw_path = Path(value.removeprefix(SQLITE_PREFIX))
            if not raw_path.is_absolute():
                raw_path = (PROJECT_ROOT / raw_path).resolve()
            return f"{SQLITE_PREFIX}{raw_path.as_posix()}"
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def missing_required_keys(self) -> list[str]:
        """Names of required API keys that are not configured."""
        missing: list[str] = []
        if self.gemini_api_key is None:
            missing.append("GEMINI_API_KEY")
        if self.tavily_api_key is None:
            missing.append("TAVILY_API_KEY")
        return missing

    def ensure_directories(self) -> None:
        """Create runtime data directories if they do not exist."""
        for directory in (self.chroma_dir, self.upload_dir, self.reports_dir, self.checkpoint_db.parent):
            directory.mkdir(parents=True, exist_ok=True)
        if self.is_sqlite and ":memory:" not in self.database_url:
            Path(self.database_url.removeprefix(SQLITE_PREFIX)).parent.mkdir(
                parents=True, exist_ok=True
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
