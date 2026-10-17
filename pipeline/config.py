"""Shared pipeline configuration from `.env` / environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

import dotenv

# Relative paths in the environment resolve against the repo root, not the cwd, so every process
# shares the same data directory (and therefore one Gemini cache, ledger and budget).
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Load the repo's .env into os.environ (existing variables win)."""
    dotenv.load_dotenv(REPO_ROOT / ".env")


def env_path(name: str, default: str) -> Path:
    return REPO_ROOT / Path(os.environ.get(name, default)).expanduser()


def data_dir() -> Path:
    return env_path("DATA_DIR", "data")


def raw_data_dir() -> Path:
    return env_path("RAW_DATA_DIR", "~/Desktop/vinhomes-crawler/detect_human")
