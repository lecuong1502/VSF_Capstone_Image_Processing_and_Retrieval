"""Guards the pre-commit hook that keeps data, weights and secrets out of the public repo."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml"


def _hook(hook_id: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] == hook_id:
                return dict(hook)
    raise LookupError(hook_id)


FORBID_HOOK = _hook("forbid-data-weights-secrets")


def _is_blocked(path: str) -> bool:
    hook = FORBID_HOOK
    included = re.search(hook["files"], path) is not None
    excluded = "exclude" in hook and re.search(hook["exclude"], path) is not None
    return included and not excluded


@pytest.mark.parametrize(
    "path",
    [
        "data/foo.csv",
        "data/raw/images.parquet",
        "data/qdrant/snapshot.tar",
        "models/jina-clip-v2.onnx",
        "services/search/model.onnx",
        "weights/model.pt",
        "checkpoints/head.pth",
        "heads/nsfw.safetensors",
        "eval/results.parquet",
        "pipeline/meta.duckdb",
        "services/search/kg.sqlite",
        "reports/k6.html",
        ".env",
        "services/search/.env",
        ".env.local",
    ],
)
def test_blocks_data_weights_and_secrets(path: str) -> None:
    assert _is_blocked(path)


@pytest.mark.parametrize(
    "path",
    [
        "data/sample/.gitkeep",
        "data/sample/images.csv",
        ".env.example",
        "pipeline/ingest.py",
        "docs/data.md",
        "tests/data_fixture.py",
        "docs/reports.md",
    ],
)
def test_allows_sample_and_code(path: str) -> None:
    assert not _is_blocked(path)
