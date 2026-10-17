"""Gemini configuration from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dotenv

from pipeline.gemini.client import Pricing


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str
    budget_usd: float
    pricing: Pricing
    thinking_level: str
    cache_dir: Path
    usage_log: Path

    @classmethod
    def from_env(cls, *, load_dotenv: bool = True) -> GeminiSettings:
        if load_dotenv:
            dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        env = os.environ.get
        data_dir = Path(env("DATA_DIR", "data")).expanduser()
        return cls(
            api_key=env("GEMINI_API_KEY", ""),
            model=env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            budget_usd=float(env("GEMINI_BUDGET_USD", "5")),
            pricing=Pricing(
                input_per_m=float(env("GEMINI_PRICE_INPUT_PER_M", "0.30")),
                output_per_m=float(env("GEMINI_PRICE_OUTPUT_PER_M", "2.50")),
                batch_discount=float(env("GEMINI_BATCH_DISCOUNT", "0.5")),
            ),
            thinking_level=env("GEMINI_THINKING_LEVEL", "MINIMAL"),
            cache_dir=data_dir / "cache" / "gemini",
            usage_log=data_dir / "gemini_usage.parquet",
        )
