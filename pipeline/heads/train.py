"""Train, evaluate and apply every moderation head on the shared embedding (spec §4.1, §4.2).

- watermark, text_overlay, political, violence: logistic regression + isotonic calibration on
  Gemini labels (#13).
- nsfw (and any flag with fewer than MIN_POSITIVES Gemini positives): zero-shot text prompts,
  unevaluated.
- relevance (per image–entity pair): logistic regression on s_text, s_proto, neighbor_rank and
  quality, trained on `verify` and evaluated on human /review labels.
- shot_type: multinomial logistic regression on the `query_type` of accepted pairs.
- quality: a resolution heuristic (ranking feature, no labels).

Images are split 60/20/20 (train / calibration+thresholds / test) by a hash of image_id, so every
head sees the same split. Binary heads report P/R/F1 with bootstrap 95% CIs on the test split at
p >= 0.5 and at the auto-reject threshold, PR-AUC and ECE, plus the share of all images each head
escalates. Outputs:
- `$DATA_DIR/heads/<model_version>/`: `<head>.npz`, `report.json`
- `$DATA_DIR/moderation/moderation_scores.parquet` (spec §7 `moderation_scores`, one row per image)
- `$DATA_DIR/moderation/relevance_scores.parquet` (one row per image–entity pair)
- `$DATA_DIR/dedup/representative_flags.parquet` (`has_brand_or_phone` = p_watermark >= 0.5) for
  the dedup representative rule – re-run `make dedup` afterwards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score

from pipeline.embed import embedding_path, load_embeddings
from pipeline.embedder import model_dir_name
from pipeline.files import atomic_write, atomic_write_text
from pipeline.heads.linear import LinearHead, fit_linear_head, fit_softmax_head
from pipeline.heads.metrics import Thresholds, bootstrap_prf, expected_calibration_error
from pipeline.heads.relevance import relevance_features

log = logging.getLogger(__name__)

TextEncoder = Callable[[list[str]], np.ndarray]

CODE_VERSION = "1"  # bump when training changes, so model_version changes too
PRECISION = 0.95  # precision(reject) at the auto-reject threshold (spec §4.2)
RECALL = 0.98  # recall(reject) at the auto-pass threshold (spec §4.2: 0.98–0.99)
MIN_POSITIVES = 10  # fewer Gemini positives than this → zero-shot instead of a trained head
SPLITS = (("train", 0.6), ("cal", 0.8), ("test", 1.0))
FLAGS = ("watermark", "text_overlay", "political", "violence", "nsfw")
ALWAYS_ZERO_SHOT = ("nsfw",)  # spec: start from a detector, not distilled labels
SCORE_COLUMNS = {"text_overlay": "p_text"}  # spec §7 names; others are p_<head>
# Zero-shot: score = cos(image, positive prompts) - cos(image, neutral prompts).
PROMPTS = {
    "nsfw": ["ảnh khỏa thân, nội dung người lớn", "nude explicit photo"],
    "violence": ["ảnh bạo lực, máu, đánh nhau", "violent bloody photo"],
    "political": ["ảnh biểu tình, cờ và khẩu hiệu chính trị", "political protest flags slogans"],
    "watermark": ["ảnh có logo và watermark đè lên", "photo with a watermark logo overlay"],
    "text_overlay": ["ảnh quảng cáo có chữ và khung", "advertisement with text overlay"],
}
NEUTRAL = ["ảnh du lịch Việt Nam", "a travel photo of Vietnam"]
ZERO_SHOT_REJECT, ZERO_SHOT_ESCALATE = 0.999, 0.995  # population quantiles of the score
# query_type values that name the crawl query, not what the photo shows
NOT_SHOT_TYPES = frozenset({"CHINH", "GOOGLE", "BING"})
MIN_CLASS = 50
QUALITY_MIN_SIDE, QUALITY_FULL_SIDE = 200, 1024


def heads_dir(data_dir: Path) -> Path:
    return data_dir / "heads"


def scores_path(data_dir: Path) -> Path:
    return data_dir / "moderation" / "moderation_scores.parquet"


def split_of(image_ids: Sequence[int], seed: int) -> np.ndarray:
    """'train' / 'cal' / 'test' from a hash of (seed, image_id): stable across heads and runs."""
    u = np.array(
        [
            int.from_bytes(hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8], "big") / 2**64
            for i in image_ids
        ]
    )
    out = np.full(len(u), "test", dtype=object)
    for name, upper in reversed(SPLITS):
        out[u < upper] = name
    return out


def quality_score(real_w: np.ndarray, real_h: np.ndarray) -> np.ndarray:
    """0 at a 200 px short side, 1 from 1024 px (log scale) – a ranking feature, no labels."""
    side = np.minimum(real_w, real_h).astype(np.float64)
    lo, hi = math.log(QUALITY_MIN_SIDE), math.log(QUALITY_FULL_SIDE)
    return np.asarray(np.clip((np.log(np.maximum(side, 1)) - lo) / (hi - lo), 0, 1))


def _binary_report(
    head: LinearHead, x_test: np.ndarray, y_test: np.ndarray, population: np.ndarray, n_boot: int
) -> dict[str, Any]:
    p = head.predict_proba(x_test)
    has_both = 0 < y_test.sum() < y_test.size
    return {
        "n_test": int(y_test.size),
        "positives_test": int(y_test.sum()),
        "test": bootstrap_prf(y_test, p >= 0.5, n_boot=n_boot),
        "reject": bootstrap_prf(y_test, p >= head.thresholds.high, n_boot=n_boot),
        "pr_auc": round(float(average_precision_score(y_test, p)), 4) if has_both else None,
        "ece": round(expected_calibration_error(p, y_test), 4),
        "tau_low": head.thresholds.low,
        "tau_high": head.thresholds.high,
        "escalation_rate": round(float(np.mean(head.thresholds.zone(population) == 1)), 4),
    }


def _zero_shot_head(name: str, encode_text: TextEncoder, vectors: np.ndarray) -> LinearHead:
    """A linear head whose direction is (positive prompts − neutral prompts); thresholds are
    population quantiles, so it escalates a fixed small share and rejects the extreme tail."""
    pos = encode_text(PROMPTS[name]).mean(axis=0)
    neg = encode_text(NEUTRAL).mean(axis=0)
    direction = (pos - neg).astype(np.float64)
    head = LinearHead(
        name, direction, 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0]), Thresholds(1, 1)
    )
    p = head.predict_proba(vectors)
    reject, escalate = np.quantile(p, [ZERO_SHOT_REJECT, ZERO_SHOT_ESCALATE])
    return LinearHead(
        name, direction, 0.0, head.iso_x, head.iso_y, Thresholds(float(escalate), float(reject))
    )


def train_heads(
    data_dir: Path, model: str, encode_text: TextEncoder, *, seed: int = 0, n_boot: int = 1000
) -> dict[str, Any]:
    emb = load_embeddings(embedding_path(data_dir, model))
    image_ids = emb["image_id"].to_list()
    vectors = np.asarray(emb["embedding"].to_list(), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    index = {image_id: row for row, image_id in enumerate(image_ids)}
    split = split_of(image_ids, seed)
    assets = pl.read_parquet(data_dir / "download" / "image_assets.parquet")
    size = dict(
        zip(assets["image_id"], zip(assets["real_w"], assets["real_h"], strict=True), strict=True)
    )
    wh = np.array([size.get(i, (0, 0)) for i in image_ids], dtype=np.float64)
    quality = quality_score(wh[:, 0], wh[:, 1])

    heads: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}
    gemini = pl.read_parquet(data_dir / "gemini_labels" / "labels.parquet").filter(
        pl.col("error").is_null() & pl.col("image_id").is_in(image_ids)
    )
    g_rows = np.array([index[i] for i in gemini["image_id"]], dtype=np.int64)
    g_split = split[g_rows] if g_rows.size else np.array([], dtype=object)
    for name in FLAGS:
        y = gemini[name].to_numpy().astype(bool) if gemini.height else np.zeros(0, bool)
        if name in ALWAYS_ZERO_SHOT or y.sum() < MIN_POSITIVES:
            head = _zero_shot_head(name, encode_text, vectors)
            p = head.predict_proba(vectors)
            reports[name] = {
                "truth": "none",
                "gemini_positives": int(y.sum()),
                "tau_low": head.thresholds.low,
                "tau_high": head.thresholds.high,
                "escalation_rate": round(float(np.mean(head.thresholds.zone(p) == 1)), 4),
            }
        else:
            x = vectors[g_rows]
            tr, cal, te = (g_split == s for s in ("train", "cal", "test"))
            head = fit_linear_head(
                name, x[tr], y[tr], x[cal], y[cal], precision=PRECISION, recall=RECALL
            )
            reports[name] = {
                "truth": "gemini",
                "n_train": int(tr.sum()),
                "n_cal": int(cal.sum()),
                **_binary_report(head, x[te], y[te], head.predict_proba(vectors), n_boot),
            }
        heads[name] = head
        log.info("%s: %s", name, json.dumps(reports[name]))

    relevance, pair_scores = _train_relevance(
        data_dir, encode_text, vectors, index, split, quality, n_boot
    )
    heads["relevance"] = relevance["head"]
    reports["relevance"] = relevance["report"]
    shot, reports["shot_type"] = _train_shot_type(data_dir, vectors, index, split, n_boot, seed)
    if shot is not None:
        heads["shot_type"] = shot
    reports["quality"] = {"truth": "none", "note": "resolution heuristic, ranking only"}

    version = _model_version(model, seed, data_dir)
    out = heads_dir(data_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    for head in heads.values():
        head.save(out)

    image_scores = pl.DataFrame(
        {
            "image_id": image_ids,
            "model_version": version,
            **{SCORE_COLUMNS.get(n, f"p_{n}"): heads[n].predict_proba(vectors) for n in FLAGS},
            "quality": quality,
            "shot_type_pred": heads["shot_type"].predict(vectors) if shot is not None else None,
        }
    )
    pair_scores = pair_scores.with_columns(model_version=pl.lit(version))
    escalated = np.zeros(len(image_ids), bool)
    for n in FLAGS:
        escalated |= (
            heads[n].thresholds.zone(image_scores[SCORE_COLUMNS.get(n, f"p_{n}")].to_numpy()) == 1
        )
    pair_zone = relevance["head"].thresholds.zone(1 - pair_scores["p_relevant"].to_numpy())
    for image_id in pair_scores.filter(pl.Series(pair_zone == 1))["image_id"].unique():
        escalated[index[image_id]] = True
    image_scores = image_scores.with_columns(escalated=pl.Series(escalated))

    moderation = scores_path(data_dir).parent
    atomic_write(scores_path(data_dir), image_scores.write_parquet)
    atomic_write(moderation / "relevance_scores.parquet", pair_scores.write_parquet)
    flags = image_scores.select(
        "image_id", has_brand_or_phone=pl.col(SCORE_COLUMNS.get("watermark", "p_watermark")) >= 0.5
    )
    atomic_write(data_dir / "dedup" / "representative_flags.parquet", flags.write_parquet)

    report = {
        "model_version": version,
        "embed_model": model,
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
        "targets": {"precision_reject": PRECISION, "recall_reject": RECALL},
        "escalation_rate": round(float(escalated.mean()), 4),
        "heads": reports,
    }
    atomic_write_text(out / "report.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _train_relevance(
    data_dir: Path,
    encode_text: TextEncoder,
    vectors: np.ndarray,
    index: dict[int, int],
    split: np.ndarray,
    quality: np.ndarray,
    n_boot: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Head on per-pair features; trained on `verify`, evaluated on human /review labels (the
    reject class is "not relevant"). Images with human labels never train or calibrate."""
    labels = pl.read_parquet(data_dir / "ingest" / "labels.parquet").filter(
        pl.col("criterion") == "relevance"
    )
    pairs = labels.select(
        "image_id",
        "entity_id",
        value=pl.when(pl.col("label_conflict")).then(pl.lit("conflict")).otherwise("value"),
    ).unique(["image_id", "entity_id"], keep="first", maintain_order=True)
    entities = pl.read_parquet(data_dir / "ingest" / "entities.parquet")
    prompts = [
        f"ảnh {name}, {leaf}" if leaf else f"ảnh {name}"
        for name, leaf in zip(entities["entity_name"], entities["leaf_label"], strict=True)
    ]
    text_vectors = encode_text(prompts)
    text_vectors = text_vectors / np.maximum(
        np.linalg.norm(text_vectors, axis=1, keepdims=True), 1e-12
    )
    text = dict(zip(entities["entity_id"], text_vectors, strict=True))
    features = relevance_features(vectors, index, pairs, text).join(
        pairs, on=["image_id", "entity_id"]
    )
    rows = np.array([index[i] for i in features["image_id"]], dtype=np.int64)
    x = _relevance_matrix(features, quality[rows])

    human_path = data_dir / "review" / "export" / "labels_human.parquet"
    human = (
        pl.read_parquet(human_path)
        .filter((pl.col("criterion") == "relevance") & pl.col("value").is_in(["yes", "no"]))
        .unique(["image_id", "entity_id"], keep="last")
        .select("image_id", "entity_id", human=pl.col("value"))
        if human_path.exists()
        else pl.DataFrame(schema={"image_id": pl.Int64, "entity_id": pl.String, "human": pl.String})
    )
    held_out = features["image_id"].is_in(human["image_id"].implode()).to_numpy()
    labeled = features["value"].is_in(["yes", "no"]).to_numpy() & ~held_out
    y = (features["value"] == "no").to_numpy()
    pair_split = split[rows]
    train = labeled & (pair_split != "cal")
    cal = labeled & (pair_split == "cal")
    head = fit_linear_head(
        "relevance", x[train], y[train], x[cal], y[cal], precision=PRECISION, recall=RECALL
    )
    p_reject = head.predict_proba(x)

    test = features.with_row_index("row").join(human, on=["image_id", "entity_id"])
    t_rows = test["row"].to_numpy()
    y_test = (test["human"] == "no").to_numpy()
    report = {
        "truth": "human",
        "train_truth": "verify",
        "n_train": int(train.sum()),
        "n_cal": int(cal.sum()),
        "features": RELEVANCE_FEATURES,
        **_binary_report(head, x[t_rows], y_test, p_reject, n_boot),
        "verify_vs_human_agreement": round(
            float(np.mean((test["value"] == test["human"]).to_numpy())), 4
        )
        if test.height
        else None,
    }
    scores = features.select(
        "image_id",
        "entity_id",
        "s_text",
        "s_proto",
        "proto_fallback",
        "neighbor_rank",
        "suggested_entity_id",
        p_relevant=pl.Series(1 - p_reject),
    )
    return {"head": head, "report": report}, scores


RELEVANCE_FEATURES = ["s_text", "s_proto", "proto_fallback", "log_neighbor_rank", "quality"]


def _relevance_matrix(features: pl.DataFrame, quality: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            features["s_text"].fill_nan(0).to_numpy(),
            features["s_proto"].fill_nan(0).to_numpy(),
            features["proto_fallback"].to_numpy().astype(np.float64),
            np.log1p(features["neighbor_rank"].to_numpy()),
            quality,
        ]
    )


def _train_shot_type(
    data_dir: Path,
    vectors: np.ndarray,
    index: dict[int, int],
    split: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Multi-class head on the most common `query_type` of each image's accepted pairs."""
    labels = (
        pl.read_parquet(data_dir / "ingest" / "labels.parquet")
        .filter(
            (pl.col("value") == "yes")
            & ~pl.col("label_conflict")
            & pl.col("query_type").is_not_null()
            & ~pl.col("query_type").is_in(list(NOT_SHOT_TYPES))
            & pl.col("image_id").is_in(list(index))
        )
        .group_by("image_id")
        .agg(pl.col("query_type").mode().sort().first())
    )
    counts = labels["query_type"].value_counts()
    keep = counts.filter(pl.col("count") >= MIN_CLASS)["query_type"]
    labels = labels.filter(pl.col("query_type").is_in(keep.implode())).sort("image_id")
    if keep.len() < 2:
        return None, {"truth": "none", "note": "fewer than two query_type classes"}
    rows = np.array([index[i] for i in labels["image_id"]], dtype=np.int64)
    y = labels["query_type"].to_numpy().astype(str)
    s = split[rows]
    head = fit_softmax_head("shot_type", vectors[rows][s != "test"], y[s != "test"])
    pred = head.predict(vectors[rows][s == "test"])
    truth = y[s == "test"]
    rng = np.random.default_rng(seed)
    boot = [
        f1_score(truth[idx], pred[idx], average="macro", zero_division=0)
        for idx in rng.integers(0, truth.size, (n_boot, truth.size))
    ]
    point = f1_score(truth, pred, average="macro", zero_division=0)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if n_boot else (point, point)
    return head, {
        "truth": "query_type",
        "classes": head.classes.tolist(),
        "n_test": int(truth.size),
        "accuracy": round(float(np.mean(pred == truth)), 4),
        "macro_f1": (round(float(point), 4), round(float(lo), 4), round(float(hi), 4)),
    }


def _model_version(model: str, seed: int, data_dir: Path) -> str:
    """<embedding model>.<hash of code version, seed and the label files>: same inputs, same
    version; new labels or code, new version."""
    digest = hashlib.sha256(f"{CODE_VERSION}:{seed}".encode())
    for path in (
        data_dir / "gemini_labels" / "labels.parquet",
        data_dir / "ingest" / "labels.parquet",
        data_dir / "review" / "export" / "labels_human.parquet",
    ):
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"{model_dir_name(model)}.{digest.hexdigest()[:10]}"
