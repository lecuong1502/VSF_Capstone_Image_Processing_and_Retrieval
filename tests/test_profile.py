"""Data profiling: verify rates by group (with Wilson intervals) and the HTML report."""

from pathlib import Path

import polars as pl
import pytest

from pipeline.ingest import ingest
from pipeline.profile import build_profile, render_html, size_bucket, verify_rates, wilson


def test_wilson_interval() -> None:
    low, high = wilson(8, 10)

    assert low == pytest.approx(0.490, abs=1e-3)
    assert high == pytest.approx(0.943, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("w", "h", "bucket"),
    [
        (1080, 900, "800–1199"),
        (200, 1000, "<300"),
        (1600, 1200, "≥1200"),
        (500, 640, "500–799"),
        (None, 720, "unknown"),
    ],
)
def test_size_bucket_uses_the_short_side(w: int | None, h: int | None, bucket: str) -> None:
    assert size_bucket(w, h) == bucket


def test_verify_rates_skip_unlabeled_and_conflicts() -> None:
    labeled = pl.DataFrame(
        {
            "group": ["a", "a", "a", "a", "b", "b"],
            "value": ["yes", "yes", "no", "unlabeled", "no", "yes"],
            "label_conflict": [False, False, False, False, False, True],
        }
    )

    rates = {r["group"]: r for r in verify_rates(labeled, "group").iter_rows(named=True)}

    assert (rates["a"]["n"], rates["a"]["yes"]) == (3, 2)
    assert rates["a"]["yes_rate"] == pytest.approx(2 / 3)
    assert (rates["b"]["n"], rates["b"]["yes"]) == (1, 0)


def test_profile_on_the_committed_sample() -> None:
    sample = ingest(Path(__file__).resolve().parents[1] / "data" / "sample")

    profile = build_profile(sample.images, sample.labels, sample.entities)

    assert profile.overview["images"] == sample.images.height
    assert profile.overview["labeled_pairs"] == 35  # 41 pairs - 6 unlabeled
    for table in ("size", "source", "query_type", "branch", "priority"):
        rates = profile.rates[table]
        assert rates["n"].sum() == 32  # labeled pairs without the 3 conflicts
    assert profile.few_yes["entity_id"].n_unique() == profile.few_yes.height


def test_html_report_has_every_section() -> None:
    sample = ingest(Path(__file__).resolve().parents[1] / "data" / "sample")

    html = render_html(build_profile(sample.images, sample.labels, sample.entities))

    assert html.startswith("<!doctype html>")
    for heading in (
        "Declared size",
        "Image source",
        "Lowest yes rate sources",
        "Query type",
        "Branch",
        "Priority",
        "Entities with fewer than 5",
    ):
        assert heading in html
    assert "prefers-color-scheme: dark" in html
