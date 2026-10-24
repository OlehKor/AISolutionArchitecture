from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="PARKING_", extra="ignore", frozen=True
    )

    llm_provider: Literal["openai", "azure"] = "openai"
    chat_model: str = "gpt-4o-mini"

    embeddings_provider: Literal["fastembed", "openai", "azure"] = "fastembed"
    embeddings_model: str = "BAAI/bge-small-en-v1.5"

    milvus_uri: str = str(PROJECT_ROOT / "var" / "milvus.db")
    milvus_collection: str = "parking_static_knowledge"
    sqlite_path: Path = PROJECT_ROOT / "var" / "parking.sqlite"

    static_data_dir: Path = PROJECT_ROOT / "data" / "static"
    schema_sql: Path = PROJECT_ROOT / "data" / "dynamic" / "schema.sql"
    seed_sql: Path = PROJECT_ROOT / "data" / "dynamic" / "seed.sql"
    eval_data_dir: Path = PROJECT_ROOT / "data" / "eval"
    reports_dir: Path = PROJECT_ROOT / "reports"

    retrieval_k: int = 4
    chunk_size: int = 700
    chunk_overlap: int = 80
    history_messages: int = 6

    telegram_bot_token: SecretStr = SecretStr("")
    telegram_admin_chat_id: int = 0
    telegram_poll_seconds: int = 30

    public_contacts: tuple[str, ...] = Field(
        default=("+370 5 210 4455", "info@skypark-central.com"),
        description="Contact details that are allowed to appear in answers.",
    )

    @property
    def telegram_token_set(self) -> bool:
        return bool(self.telegram_bot_token.get_secret_value())

    @property
    def telegram_configured(self) -> bool:
        return self.telegram_token_set and self.telegram_admin_chat_id != 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
