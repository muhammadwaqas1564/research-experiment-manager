"""Application configuration loaded from environment variables and a local .env file.

The API key is deliberately NOT stored on the Settings object so it cannot be
accidentally printed, logged or rendered in the UI. Only ``llm/groq_client.py``
reads it, directly from the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}


def load_env(dotenv_path: Path | None = None) -> None:
    """Load the local .env file without overriding variables already set."""
    load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    results_dir: Path
    tasks_dir: Path
    cache_path: Path
    llm_enabled: bool
    llm_provider: str

    @classmethod
    def from_env(cls, data_dir: Path | str | None = None) -> "Settings":
        load_env()
        base = Path(data_dir or os.getenv("REM_DATA_DIR", PROJECT_ROOT / "data"))
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return cls(
            data_dir=base,
            db_path=base / "registry.sqlite",
            results_dir=base / "results",
            tasks_dir=base / "tasks",
            cache_path=base / "llm_cache.sqlite",
            llm_enabled=env_bool("LLM_ENABLED", True),
            llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.results_dir, self.tasks_dir):
            d.mkdir(parents=True, exist_ok=True)
