"""Configuration management for CroweTrade."""
import os
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    run_mode: str
    paper_mode: bool
    log_level: str
    pythonpath: str
    port: int
    host: str


@lru_cache()
def get_settings() -> Settings:
    """Get application settings from environment variables."""
    return Settings(
        run_mode=os.getenv("RUN_MODE", "development"),
        paper_mode=os.getenv("PAPER_MODE", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
        pythonpath=os.getenv("PYTHONPATH", "/app/src"),
        port=int(os.getenv("PORT", "8080")),
        host=os.getenv("HOST", "0.0.0.0"),
    )


def is_production() -> bool:
    """Check if running in production mode."""
    return get_settings()["run_mode"] == "production"


def is_paper_mode() -> bool:
    """Check if running in paper trading mode."""
    return get_settings()["paper_mode"]
