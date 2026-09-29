from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vibe to MIDI API"
    environment: str = "development"
    rate_limit_per_minute: str = "5/minute"
    cors_origins: list[str] = ["*"]
    
    # Reads automatically from .env or environment variables
    gemini_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Returns cached application settings."""
    return Settings()