from pathlib import Path

import pytest

from pipeline.gemini import Pricing
from pipeline.gemini.settings import GeminiSettings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_BUDGET_USD",
        "GEMINI_PRICE_INPUT_PER_M",
        "GEMINI_PRICE_OUTPUT_PER_M",
        "GEMINI_BATCH_DISCOUNT",
        "DATA_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = GeminiSettings.from_env(load_dotenv=False)

    assert settings.api_key == ""
    assert settings.model == "gemini-3.5-flash-lite"
    assert settings.budget_usd == 5.0
    assert settings.pricing == Pricing(input_per_m=0.30, output_per_m=2.50, batch_discount=0.5)
    repo_root = Path(__file__).resolve().parents[1]
    assert settings.cache_dir == repo_root / "data" / "cache" / "gemini"
    assert settings.usage_log == repo_root / "data" / "gemini_usage.parquet"


def test_overrides_and_home_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_BUDGET_USD", "12.5")
    monkeypatch.setenv("GEMINI_PRICE_INPUT_PER_M", "0.15")
    monkeypatch.setenv("DATA_DIR", "~/vsf-data")

    settings = GeminiSettings.from_env(load_dotenv=False)

    assert settings.api_key == "k"
    assert settings.budget_usd == 12.5
    assert settings.pricing.input_per_m == 0.15
    assert settings.usage_log == Path.home() / "vsf-data" / "gemini_usage.parquet"


def test_relative_data_dir_resolves_against_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.chdir("/tmp")

    settings = GeminiSettings.from_env(load_dotenv=False)

    repo_root = Path(__file__).resolve().parents[1]
    assert settings.usage_log == repo_root / "data" / "gemini_usage.parquet"
