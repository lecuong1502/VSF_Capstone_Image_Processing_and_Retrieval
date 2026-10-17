"""Moderation heads: thresholds and CIs, linear heads, relevance features, end-to-end training."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.embed import embedding_path
from pipeline.heads.linear import (
    LinearHead,
    SoftmaxHead,
    fit_linear_head,
    fit_softmax_head,
    load_head,
)
from pipeline.heads.metrics import bootstrap_prf, expected_calibration_error, pick_thresholds
from pipeline.heads.relevance import relevance_features
from pipeline.heads.train import scores_path, train_heads

MODEL = "org/tiny-clip"

# --- metrics -----------------------------------------------------------------------------------


def test_thresholds_meet_precision_and_recall_targets() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    p = np.clip(np.where(y, rng.normal(0.75, 0.15, 2000), rng.normal(0.25, 0.15, 2000)), 0, 1)

    t = pick_thresholds(p, y, precision=0.95, recall=0.98)

    assert t.low <= t.high
    reject = p >= t.high
    assert y[reject].mean() >= 0.95
    assert (p[y] > t.low).mean() >= 0.98
    # the loosest reject threshold: the next score below it breaks the precision target
    next_lower = p[p < t.high].max()
    assert y[p >= next_lower].mean() < 0.95


def test_thresholds_without_a_precise_region_reject_nothing() -> None:
    y = np.array([True, False] * 50)
    p = np.full(100, 0.5)

    t = pick_thresholds(p, y, precision=0.95, recall=0.98)

    assert t.high == np.inf  # never auto-reject; everything above low escalates
    assert t.low < 0.5


def test_bootstrap_prf_contains_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.3
    pred = y ^ (rng.random(500) < 0.1)

    m = bootstrap_prf(y, pred, n_boot=200, seed=0)

    for name in ("precision", "recall", "f1"):
        point, lo, hi = m[name]
        assert lo <= point <= hi
        assert 0 < hi - lo < 0.2


def test_ece_is_zero_for_perfect_calibration() -> None:
    p = np.repeat([0.1, 0.9], 1000)
    y = np.concatenate([np.arange(1000) < 100, np.arange(1000) < 900])
    assert expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(np.full(10, 0.9), np.zeros(10, bool)) == pytest.approx(0.9)


# --- linear heads ------------------------------------------------------------------------------


def _blobs(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[:, 0] += np.where(y, 2.0, -1.0)
    return x, y


def test_linear_head_is_calibrated_and_round_trips(tmp_path: Path) -> None:
    x, y = _blobs(3000, 6, 0)
    head = fit_linear_head(
        "watermark", x[:2000], y[:2000], x[2000:], y[2000:], precision=0.95, recall=0.98
    )

    p = head.predict_proba(x[2000:])
    assert ((p >= 0) & (p <= 1)).all()
    assert expected_calibration_error(p, y[2000:]) < 0.05
    assert head.thresholds.low <= head.thresholds.high

    path = head.save(tmp_path)
    again = load_head(path)
    assert isinstance(again, LinearHead)
    assert np.allclose(again.predict_proba(x), head.predict_proba(x))
    assert again.thresholds == head.thresholds
    assert path.suffix == ".npz"  # plain arrays, no pickle


def test_softmax_head_predicts_classes(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    classes = np.array(["A", "B", "C"])
    y = classes[rng.integers(0, 3, 900)]
    x = rng.normal(size=(900, 4)).astype(np.float32)
    x[np.arange(900), np.searchsorted(classes, y)] += 3

    head = fit_softmax_head("shot_type", x[:600], y[:600])

    assert (head.predict(x[600:]) == y[600:]).mean() > 0.9
    again = load_head(head.save(tmp_path))
    assert isinstance(again, SoftmaxHead)
    assert (again.predict(x) == head.predict(x)).all()


# --- relevance features ------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


def test_relevance_features_leave_one_out_and_fallback() -> None:
    # entity e1: four yes images along x; entity e2: one yes image along y
    vectors = np.stack(
        [_unit([1, 0.1 * i, 0]) for i in range(4)] + [_unit([0, 1, 0]), _unit([0.1, 1, 0])]
    )
    ids = list(range(1, 7))
    pairs = pl.DataFrame(
        {
            "image_id": ids,
            "entity_id": ["e1"] * 4 + ["e2", "e1"],
            "value": ["yes"] * 5 + ["no"],
        }
    )
    text = {"e1": _unit([1, 0, 0]), "e2": _unit([0, 1, 0])}

    f = relevance_features(vectors, dict(zip(ids, range(6), strict=True)), pairs, text, min_proto=3)

    row = {r["image_id"]: r for r in f.iter_rows(named=True)}
    # image 1's prototype excludes itself: mean of images 2-4
    loo = vectors[1:4].mean(0)
    assert row[1]["s_proto"] == pytest.approx(
        float(vectors[0] @ (loo / np.linalg.norm(loo))), abs=1e-5
    )
    assert not row[1]["proto_fallback"]
    # e2 has a single yes image: without itself nothing is left → falls back to s_text
    assert row[5]["proto_fallback"] and row[5]["s_proto"] == pytest.approx(row[5]["s_text"])
    # image 6 is labeled e1 but looks like e2 → e1 ranks second, suggested e2 (possible mislabel)
    assert (row[6]["neighbor_rank"], row[6]["suggested_entity_id"]) == (2, "e2")
    assert row[1]["neighbor_rank"] == 1


# --- end to end --------------------------------------------------------------------------------


def _vec(seed: int, d: int, shift: dict[int, float]) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=d).astype(np.float32)
    for k, s in shift.items():
        v[k] += s
    return v / np.linalg.norm(v)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """600 images of 6 entities (entity k shifts dim k); watermark shifts dim 7 in 30%."""
    data = tmp_path / "data"
    d, n = 16, 600
    rng = np.random.default_rng(0)
    entity_of = rng.integers(0, 6, n)
    watermark = rng.random(n) < 0.3
    political = rng.random(n) < 0.05
    ids = list(range(1, n + 1))
    vecs = [
        _vec(
            i, d, {int(entity_of[i - 1]): 4.0, 7: 3.0 * watermark[i - 1], 8: 3.0 * political[i - 1]}
        )
        for i in ids
    ]
    out = embedding_path(data, MODEL)
    out.mkdir(parents=True)
    pl.DataFrame(
        {
            "image_id": ids,
            "sha256": [f"s{i}" for i in ids],
            "embedding": np.stack(vecs),
            "embedded_at": [datetime(2026, 10, 17, tzinfo=UTC)] * n,
        },
        schema={
            "image_id": pl.Int64,
            "sha256": pl.String,
            "embedding": pl.Array(pl.Float32, d),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(out / "part-1.parquet")
    (data / "ingest").mkdir(parents=True)
    pl.DataFrame(
        {
            "entity_id": [f"e{k}" for k in range(6)],
            "entity_name": [f"Thực thể {k}" for k in range(6)],
            "leaf_label": ["Chùa"] * 6,
        }
    ).write_parquet(data / "ingest" / "entities.parquet")
    # verify: the right entity is `yes`; every 4th image is also paired (`no`) with the next one
    rows = []
    for i in ids:
        k = int(entity_of[i - 1])
        rows.append((i, f"e{k}", "yes", False, ["TOAN_CANH", "NOI_THAT"][i % 2]))
        if i % 4 == 0:
            rows.append((i, f"e{(k + 1) % 6}", "no", False, "TOAN_CANH"))
    pl.DataFrame(
        rows,
        schema=["image_id", "entity_id", "value", "label_conflict", "query_type"],
        orient="row",
    ).with_columns(criterion=pl.lit("relevance")).write_parquet(data / "ingest" / "labels.parquet")
    (data / "download").mkdir()
    pl.DataFrame(
        {"image_id": ids, "download_status": ["ok"] * n, "real_w": [800] * n, "real_h": [600] * n}
    ).write_parquet(data / "download" / "image_assets.parquet")
    # Gemini labeled the first 400 images
    (data / "gemini_labels").mkdir()
    g = ids[:400]
    pl.DataFrame(
        {
            "image_id": g,
            "entity_id": [f"e{int(entity_of[i - 1])}" for i in g],
            "watermark": [bool(watermark[i - 1]) for i in g],
            "text_overlay": [bool(watermark[i - 1]) for i in g],
            "nsfw": [False] * 400,
            "political": [bool(political[i - 1]) for i in g],
            "violence": [False] * 400,
            "relevance": ["yes"] * 400,
            "shot_type": ["KHAC"] * 400,
            "confidence": [0.9] * 400,
            "reason": [""] * 400,
            "model": ["g"] * 400,
            "input_tokens": [1] * 400,
            "output_tokens": [1] * 400,
            "error": [None] * 400,
        },
        schema_overrides={"error": pl.String},
    ).write_parquet(data / "gemini_labels" / "labels.parquet")
    # human relevance labels for 100 of the paired images
    (data / "review" / "export").mkdir(parents=True)
    human = [r for r in rows if r[0] > 500]
    pl.DataFrame(
        {
            "image_id": [r[0] for r in human],
            "entity_id": [r[1] for r in human],
            "criterion": ["relevance"] * len(human),
            "value": [r[2] for r in human],
            "source": ["human"] * len(human),
        }
    ).write_parquet(data / "review" / "export" / "labels_human.parquet")
    return data


def fake_text_encoder(texts: list[str]) -> np.ndarray:
    """'... Thực thể k ...' → a unit vector on dim k (like a perfect text tower); other prompts
    hash to a random direction."""
    out = []
    for t in texts:
        digits = [c for c in t if c.isdigit()]
        if "Thực thể" in t and digits:
            out.append(_vec(99, 16, {int(digits[0]): 20.0}))
        else:
            out.append(_vec(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16), 16, {}))
    return np.stack(out)


def test_train_heads_end_to_end(data_dir: Path) -> None:
    report = train_heads(data_dir, MODEL, fake_text_encoder, seed=0, n_boot=50)

    heads = report["heads"]
    assert set(heads) >= {
        "watermark",
        "text_overlay",
        "political",
        "relevance",
        "shot_type",
        "nsfw",
    }
    wm = heads["watermark"]
    assert wm["truth"] == "gemini"
    assert wm["test"]["f1"][0] > 0.8  # dim 7 separates it
    assert len(wm["test"]["precision"]) == 3  # point, CI low, CI high
    assert heads["relevance"]["truth"] == "human"
    assert heads["relevance"]["test"]["f1"][0] > 0.7
    assert heads["nsfw"]["truth"] == "none"  # no positives anywhere: zero-shot, unevaluated
    assert 0 <= report["escalation_rate"] <= 1

    version = report["model_version"]
    assert version.startswith("org--tiny-clip.")
    saved = data_dir / "heads" / version
    assert (saved / "watermark.npz").exists() and (saved / "report.json").exists()
    assert json.loads((saved / "report.json").read_text())["model_version"] == version

    scores = pl.read_parquet(scores_path(data_dir))
    assert scores.height == 600
    assert {"p_watermark", "p_text", "p_political", "p_nsfw", "p_violence", "quality"} <= set(
        scores.columns
    )
    assert scores["model_version"].unique().to_list() == [version]
    flags = pl.read_parquet(data_dir / "dedup" / "representative_flags.parquet")
    assert flags.columns == ["image_id", "has_brand_or_phone"]
    assert flags["has_brand_or_phone"].mean() == pytest.approx(0.3, abs=0.1)
    pairs = pl.read_parquet(scores_path(data_dir).with_name("relevance_scores.parquet"))
    assert {"image_id", "entity_id", "p_relevant", "s_text", "s_proto", "neighbor_rank"} <= set(
        pairs.columns
    )


def test_training_is_deterministic(data_dir: Path) -> None:
    a = train_heads(data_dir, MODEL, fake_text_encoder, seed=0, n_boot=20)
    b = train_heads(data_dir, MODEL, fake_text_encoder, seed=0, n_boot=20)
    assert a["model_version"] == b["model_version"]
    assert a["heads"]["watermark"]["test"] == b["heads"]["watermark"]["test"]
