"""Train, evaluate and apply every moderation head on the shared embedding (spec §4.1, §4.2).

- watermark, text_overlay, political, violence: logistic regression + isotonic calibration on
  Gemini labels (#13), evaluated against Gemini labels of held-out images (the human audit only
  validated relevance).
- nsfw (and any flag with fewer than MIN_POSITIVES Gemini positives): zero-shot text prompts. It
  never auto-rejects: it only escalates the top ZERO_SHOT_ESCALATE share of images.
- shot_type: multinomial logistic regression on the `query_type` of accepted pairs.
- relevance (per image–entity pair): logistic regression on s_text, s_proto, the neighbor rank,
  quality and the shot-type probabilities; trained on `verify`, evaluated on human /review
  labels. Images with human labels never train or calibrate it; the other test-split images do,
  since its test set is the human one.
- quality: a resolution heuristic (ranking feature, no labels).

Images are split 60/20/20 (train / calibration / test) by a hash of image_id, so every head sees
the same split. Isotonic calibration and the auto-reject threshold (precision >= PRECISION) are
fit on the calibration split; the pass thresholds split ESCALATION_BUDGET across heads where it
buys the most calibration recall (zero-shot heads get a fixed share);
binary heads report on the test split P/R/F1 with bootstrap 95% CIs at p >= 0.5 and at the
auto-reject threshold, the recall kept by the auto-pass threshold, a confusion matrix, PR-AUC,
ECE with reliability bins, and the share of all images each head escalates. Outputs:
- `$DATA_DIR/heads/<model_version>/`: `<head>.npz`, `report.json`
- `$DATA_DIR/moderation/moderation_scores.parquet` (spec §7 `moderation_scores`, one row per image)
- `$DATA_DIR/moderation/relevance_scores.parquet` (the per-pair columns of §7, one row per pair)
- `$DATA_DIR/dedup/representative_flags.parquet`: the `watermark_head` column (p_watermark >=
  0.5, trained head only), OR-ed with other sources' columns (OCR, #16) into
  `has_brand_or_phone` – re-run `make dedup` afterwards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score

from pipeline.download import assets_path
from pipeline.embed import embedding_matrix, embedding_path, load_embeddings
from pipeline.embedder import model_dir_name
from pipeline.files import atomic_write, atomic_write_text
from pipeline.gemini_label import labels_path as gemini_labels_path
from pipeline.heads.linear import LinearHead, SoftmaxHead, fit_linear_head, fit_softmax_head
from pipeline.heads.metrics import (
    BudgetCurve,
    Thresholds,
    allocate_escalation,
    bootstrap_prf,
    expected_calibration_error,
)
from pipeline.heads.relevance import relevance_features
from pipeline.review.items import review_dir
from pipeline.review.store import labels_human_path

log = logging.getLogger(__name__)

TextEncoder = Callable[[list[str]], np.ndarray]

CODE_VERSION = "4"  # bump when training changes, so model_version changes too
PRECISION = 0.95  # precision(reject) at the auto-reject threshold (spec §4.2)
RECALL = 0.98  # recall(reject) target (spec §4.2: 0.98–0.99), reached only within the budget
# Share of all images the heads may escalate together (#14: ≤ 5%). Pass thresholds are not set
# per head from RECALL – the heads can't meet it under this budget – but by allocating the
# budget where it buys the most recall (`allocate_escalation`); the PR reports the recall reached.
ESCALATION_BUDGET = 0.05
BUDGET_STEP = 0.0025
# Fewer Gemini positives than this leave ~6 per calibration/test split → zero-shot instead.
MIN_POSITIVES = 30
SPLITS = (("train", 0.6), ("cal", 0.8), ("test", 1.0))
FLAGS = ("watermark", "text_overlay", "political", "violence", "nsfw")
ALWAYS_ZERO_SHOT = ("nsfw",)  # Gemini found no NSFW image in 5k (#13)
SCORE_COLUMNS = {"text_overlay": "p_text"}  # spec §7 names; others are p_<head>
# Zero-shot: score = sigmoid(SCALE · (cos(image, positive prompts) − cos(image, neutral prompts))).
PROMPTS = {
    "nsfw": ["ảnh khỏa thân, nội dung người lớn", "nude explicit photo"],
    "violence": ["ảnh bạo lực, máu, đánh nhau", "violent bloody photo"],
    "political": ["ảnh biểu tình, cờ và khẩu hiệu chính trị", "political protest flags slogans"],
    "watermark": ["ảnh có logo và watermark đè lên", "photo with a watermark logo overlay"],
    "text_overlay": ["ảnh quảng cáo có chữ và khung", "advertisement with text overlay"],
}
NEUTRAL = ["ảnh du lịch Việt Nam", "a travel photo of Vietnam"]
ZERO_SHOT_SCALE = 100.0  # CLIP-style logit scale, so scores spread over (0, 1)
ZERO_SHOT_SHARE = 0.005  # share of images a zero-shot head escalates (reserved from the budget)
# query_type values that name the crawl query, not what the photo shows
NOT_SHOT_TYPES = frozenset({"CHINH", "GOOGLE", "BING"})
MIN_CLASS = 50
FLAG_C = 10.0  # weaker regularization: +0.02–0.04 CV PR-AUC on the Gemini flags vs C=1
QUALITY_MIN_SIDE, QUALITY_FULL_SIDE = 200, 1024
RELIABILITY_BINS = 10


def heads_dir(data_dir: Path) -> Path:
    return data_dir / "heads"


def scores_path(data_dir: Path) -> Path:
    return data_dir / "moderation" / "moderation_scores.parquet"


def score_column(head: str) -> str:
    return SCORE_COLUMNS.get(head, f"p_{head}")


@dataclass(frozen=True)
class Corpus:
    """Every embedded image: L2-normalized vectors, row lookup, split and quality."""

    image_ids: list[int]
    vectors: np.ndarray
    index: dict[int, int]
    split: np.ndarray  # "train" / "cal" / "test" per row
    quality: np.ndarray

    def rows(self, image_ids: Sequence[int]) -> np.ndarray:
        return np.array([self.index[i] for i in image_ids], dtype=np.int64)


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


def load_corpus(data_dir: Path, model: str, seed: int) -> Corpus:
    emb = load_embeddings(embedding_path(data_dir, model))
    image_ids = emb["image_id"].to_list()
    vectors = embedding_matrix(emb)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sizes = (
        pl.DataFrame({"image_id": image_ids})
        .join(pl.read_parquet(assets_path(data_dir)), on="image_id", how="left")
        .select(pl.col("real_w").fill_null(0), pl.col("real_h").fill_null(0))
    )
    return Corpus(
        image_ids=image_ids,
        vectors=vectors,
        index={image_id: row for row, image_id in enumerate(image_ids)},
        split=split_of(image_ids, seed),
        quality=quality_score(sizes["real_w"].to_numpy(), sizes["real_h"].to_numpy()),
    )


def _reliability(p: np.ndarray, y: np.ndarray) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, n) per probability bin – the reliability diagram."""
    which = np.minimum((p * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
    return [
        (round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum()))
        for m in (which == b for b in range(RELIABILITY_BINS))
        if m.any()
    ]


def _binary_report(
    head: LinearHead, x_test: np.ndarray, y_test: np.ndarray, n_boot: int
) -> dict[str, Any]:
    p = head.predict_proba(x_test)
    raw = head.raw(x_test)
    pred = p >= 0.5
    has_both = 0 < y_test.sum() < y_test.size
    t = head.thresholds
    return {
        "n_test": int(y_test.size),
        "positives_test": int(y_test.sum()),
        "test": bootstrap_prf(y_test, pred, n_boot=n_boot),
        "reject": bootstrap_prf(y_test, raw >= t.high, n_boot=n_boot),
        "recall_kept_by_low": round(float(np.mean(raw[y_test] > t.low)), 4)
        if y_test.any()
        else None,
        "confusion": {
            "tp": int(np.sum(pred & y_test)),
            "fp": int(np.sum(pred & ~y_test)),
            "fn": int(np.sum(~pred & y_test)),
            "tn": int(np.sum(~pred & ~y_test)),
        },
        "pr_auc": round(float(average_precision_score(y_test, p)), 4) if has_both else None,
        "ece": round(expected_calibration_error(p, y_test), 4),
        "reliability": _reliability(p, y_test),
        "tau_low": t.low,
        "tau_high": t.high,
    }


def _zero_shot_head(name: str, encode_text: TextEncoder, vectors: np.ndarray) -> LinearHead:
    """Direction (positive − neutral prompts); never rejects – the budget sets how much of the top
    it escalates."""
    pos = encode_text(PROMPTS[name]).mean(axis=0)
    neg = encode_text(NEUTRAL).mean(axis=0)
    direction = (pos - neg).astype(np.float64) * ZERO_SHOT_SCALE
    identity = np.array([0.0, 1.0])
    return LinearHead(name, direction, 0.0, identity, identity, Thresholds(math.inf, math.inf))


def _load_gemini(data_dir: Path, corpus: Corpus) -> pl.DataFrame:
    path = gemini_labels_path(data_dir)
    if not path.exists():
        log.warning("No Gemini labels at %s – flag heads fall back to zero-shot", path)
        return pl.DataFrame(schema={"image_id": pl.Int64, **dict.fromkeys(FLAGS, pl.Boolean)})
    return pl.read_parquet(path).filter(
        pl.col("error").is_null() & pl.col("image_id").is_in(corpus.image_ids)
    )


@dataclass
class Trained:
    """A binary head before its pass threshold comes from the escalation budget."""

    head: LinearHead
    report: dict[str, Any]
    x_test: np.ndarray | None = None  # None: no labels to evaluate on
    y_test: np.ndarray | None = None
    cal_positive_score: np.ndarray = field(default_factory=lambda: np.zeros(0))  # raw scores


def _train_flags(data_dir: Path, corpus: Corpus, encode_text: TextEncoder) -> dict[str, Trained]:
    gemini = _load_gemini(data_dir, corpus)
    rows = corpus.rows(gemini["image_id"].to_list())
    split = corpus.split[rows]
    trained: dict[str, Trained] = {}
    for name in FLAGS:
        y = gemini[name].to_numpy().astype(bool)
        if name in ALWAYS_ZERO_SHOT or y.sum() < MIN_POSITIVES:
            report = {"truth": "none", "model": "zero-shot", "gemini_positives": int(y.sum())}
            trained[name] = Trained(_zero_shot_head(name, encode_text, corpus.vectors), report)
            continue
        x = corpus.vectors[rows]
        train, cal, test = (split == s for s in ("train", "cal", "test"))
        head = fit_linear_head(
            name, x[train], y[train], x[cal], y[cal], precision=PRECISION, recall=RECALL, c=FLAG_C
        )
        report = {
            "truth": "gemini",
            "model": "logistic+isotonic",
            "n_train": int(train.sum()),
            "n_cal": int(cal.sum()),
        }
        trained[name] = Trained(head, report, x[test], y[test], head.raw(x[cal])[y[cal]])
    return trained


def _train_shot_type(
    data_dir: Path, corpus: Corpus, n_boot: int, seed: int
) -> tuple[SoftmaxHead | None, dict[str, Any]]:
    """Multi-class head on the most common `query_type` of each image's accepted pairs."""
    labels = (
        pl.read_parquet(data_dir / "ingest" / "labels.parquet")
        .filter(
            (pl.col("value") == "yes")
            & ~pl.col("label_conflict")
            & pl.col("query_type").is_not_null()
            & ~pl.col("query_type").is_in(list(NOT_SHOT_TYPES))
            & pl.col("image_id").is_in(corpus.image_ids)
        )
        .group_by("image_id")
        .agg(pl.col("query_type").mode().sort().first())
    )
    keep = labels["query_type"].value_counts().filter(pl.col("count") >= MIN_CLASS)["query_type"]
    labels = labels.filter(pl.col("query_type").is_in(keep.to_list())).sort("image_id")
    if keep.len() < 2:
        return None, {"truth": "none", "note": "fewer than two query_type classes"}
    rows = corpus.rows(labels["image_id"].to_list())
    y = labels["query_type"].to_numpy().astype(str)
    is_test = corpus.split[rows] == "test"
    head = fit_softmax_head("shot_type", corpus.vectors[rows][~is_test], y[~is_test])
    truth = y[is_test]
    if truth.size == 0:
        return head, {"truth": "query_type", "classes": head.classes.tolist(), "n_test": 0}
    pred = head.predict(corpus.vectors[rows][is_test])
    rng = np.random.default_rng(seed)
    boot = [
        f1_score(truth[idx], pred[idx], average="macro", zero_division=0)
        for idx in rng.integers(0, truth.size, (n_boot, truth.size))
    ]
    point = float(f1_score(truth, pred, average="macro", zero_division=0))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if n_boot else (point, point)
    return head, {
        "truth": "query_type",
        "classes": head.classes.tolist(),
        "n_test": int(truth.size),
        "accuracy": round(float(np.mean(pred == truth)), 4),
        "macro_f1": (round(point, 4), round(float(lo), 4), round(float(hi), 4)),
    }


RELEVANCE_FEATURES = ["s_text", "s_proto", "proto_fallback", "reciprocal_rank", "quality"]


def _relevance_matrix(
    features: pl.DataFrame, quality: np.ndarray, shot_probs: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            features["s_text"].fill_nan(0).to_numpy(),
            features["s_proto"].fill_nan(0).to_numpy(),
            features["proto_fallback"].to_numpy().astype(np.float64),
            (1 / features["neighbor_rank"].cast(pl.Float64)).fill_null(0).to_numpy(),
            quality,
            shot_probs,
        ]
    )


def _train_relevance(
    data_dir: Path,
    corpus: Corpus,
    encode_text: TextEncoder,
    shot: SoftmaxHead | None,
) -> tuple[Trained, pl.DataFrame]:
    """Head on per-pair features; the reject class is "not relevant"."""
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
    features = relevance_features(corpus.vectors, corpus.index, pairs, text).join(
        pairs, on=["image_id", "entity_id"]
    )
    rows = corpus.rows(features["image_id"].to_list())
    shot_probs = (
        shot.predict_proba(corpus.vectors[rows]) if shot is not None else np.zeros((len(rows), 0))
    )
    x = _relevance_matrix(features, corpus.quality[rows], shot_probs)

    human_path = labels_human_path(review_dir(data_dir))
    human = (
        pl.read_parquet(human_path)
        .filter((pl.col("criterion") == "relevance") & pl.col("value").is_in(["yes", "no"]))
        .unique(["image_id", "entity_id"], keep="last")
        .select("image_id", "entity_id", human=pl.col("value"))
        if human_path.exists()
        else pl.DataFrame(schema={"image_id": pl.Int64, "entity_id": pl.String, "human": pl.String})
    )
    held_out = features["image_id"].is_in(human["image_id"].to_list()).to_numpy()
    labeled = features["value"].is_in(["yes", "no"]).to_numpy() & ~held_out
    y = (features["value"] == "no").to_numpy()
    is_cal = corpus.split[rows] == "cal"
    train, cal = labeled & ~is_cal, labeled & is_cal
    head = fit_linear_head(
        "relevance", x[train], y[train], x[cal], y[cal], precision=PRECISION, recall=RECALL
    )
    p_reject = head.predict_proba(x)
    raw_reject = head.raw(x)

    test = features.with_row_index("row").join(human, on=["image_id", "entity_id"])
    t_rows = test["row"].to_numpy()
    report = {
        "truth": "human",
        "train_truth": "verify",
        "model": "logistic+isotonic",
        "n_train": int(train.sum()),
        "n_cal": int(cal.sum()),
        "features": RELEVANCE_FEATURES + [f"p_shot_{c}" for c in (shot.classes if shot else [])],
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
        reject_score=pl.Series(raw_reject),
    )
    cal_positive = raw_reject[cal & y]
    trained = Trained(head, report, x[t_rows], (test["human"] == "no").to_numpy(), cal_positive)
    return trained, scores


def _write_representative_flags(data_dir: Path, image_scores: pl.DataFrame) -> None:
    """Set this head's `watermark_head` column and recompute `has_brand_or_phone` as the OR of
    every source column, keeping other sources (OCR, #16) intact."""
    path = data_dir / "dedup" / "representative_flags.parquet"
    mine = image_scores.select("image_id", watermark_head=pl.col(score_column("watermark")) >= 0.5)
    flags = (
        pl.read_parquet(path).drop("watermark_head", "has_brand_or_phone", strict=False)
        if path.exists()
        else pl.DataFrame(schema={"image_id": pl.Int64})
    )
    flags = flags.join(mine, on="image_id", how="full", coalesce=True)
    sources = [c for c in flags.columns if c != "image_id"]
    flags = flags.with_columns(
        has_brand_or_phone=pl.any_horizontal(pl.col(sources).fill_null(False))
    ).select("image_id", "has_brand_or_phone", *sources)
    atomic_write(path, flags.write_parquet)


def _escalation_keys(head: LinearHead, raw: np.ndarray) -> np.ndarray:
    """An image escalates when its key > low; images the head auto-rejects never escalate."""
    return np.where(raw < head.thresholds.high, raw, -np.inf)


def _relevance_keys(corpus: Corpus, head: LinearHead, pair_scores: pl.DataFrame) -> np.ndarray:
    """Per image, the highest raw reject score among its pairs that are not auto-rejected: the
    image escalates when any of its pairs falls in the band."""
    keyed = pair_scores.filter(pl.col("reject_score") < head.thresholds.high)
    best = keyed.group_by("image_id").agg(pl.col("reject_score").max())
    keys = np.full(len(corpus.image_ids), -np.inf)
    keys[corpus.rows(best["image_id"].to_list())] = best["reject_score"].to_numpy()
    return keys


def train_heads(
    data_dir: Path, model: str, encode_text: TextEncoder, *, seed: int = 0, n_boot: int = 1000
) -> dict[str, Any]:
    corpus = load_corpus(data_dir, model, seed)
    trained = _train_flags(data_dir, corpus, encode_text)
    shot, shot_report = _train_shot_type(data_dir, corpus, n_boot, seed)
    trained["relevance"], pair_scores = _train_relevance(data_dir, corpus, encode_text, shot)

    raw = {n: trained[n].head.raw(corpus.vectors) for n in FLAGS}
    curves = {
        n: BudgetCurve(
            keys=_escalation_keys(trained[n].head, raw[n]),
            cal_positive_p=trained[n].cal_positive_score,
            high=trained[n].head.thresholds.high,
            fixed_share=ZERO_SHOT_SHARE if trained[n].report["model"] == "zero-shot" else None,
        )
        for n in FLAGS
    }
    curves["relevance"] = BudgetCurve(
        keys=_relevance_keys(corpus, trained["relevance"].head, pair_scores),
        cal_positive_p=trained["relevance"].cal_positive_score,
        high=trained["relevance"].head.thresholds.high,
    )
    allocation = allocate_escalation(
        curves, budget=ESCALATION_BUDGET, step=BUDGET_STEP, target_recall=RECALL
    )
    heads: dict[str, LinearHead] = {}
    reports: dict[str, dict[str, Any]] = {}
    for name, t in trained.items():
        a = allocation[name]
        head = replace(t.head, thresholds=Thresholds(a.low, t.head.thresholds.high))
        heads[name] = head
        reports[name] = {
            **t.report,
            "escalation_share": a.share,
            "recall_cal_within_budget": round(a.recall, 4) if t.cal_positive_score.size else None,
        }
        if t.x_test is not None and t.y_test is not None:
            reports[name] |= _binary_report(head, t.x_test, t.y_test, n_boot)
        else:
            reports[name] |= {"tau_low": head.thresholds.low, "tau_high": head.thresholds.high}
        log.info("%s: %s", name, json.dumps(reports[name]))
    reports["shot_type"] = shot_report
    reports["quality"] = {"truth": "none", "model": "resolution heuristic, ranking only"}

    version = _model_version(model, seed, data_dir)
    out = heads_dir(data_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    saved: list[LinearHead | SoftmaxHead] = [*heads.values(), *([shot] if shot else [])]
    for saved_head in saved:
        saved_head.save(out)

    image_scores = pl.DataFrame(
        {
            "image_id": corpus.image_ids,
            "model_version": version,
            **{score_column(n): heads[n].predict_proba(corpus.vectors) for n in FLAGS},
            **{f"zone_{n}": heads[n].thresholds.zone(raw[n]) for n in FLAGS},
            "quality": corpus.quality,
            "shot_type_pred": shot.predict(corpus.vectors) if shot else None,
        }
    )
    escalated = np.zeros(len(corpus.image_ids), bool)
    for n in FLAGS:
        escalated |= heads[n].thresholds.zone(raw[n]) == 1
    escalated |= curves["relevance"].keys > heads["relevance"].thresholds.low
    image_scores = image_scores.with_columns(escalated=pl.Series(escalated))

    atomic_write(scores_path(data_dir), image_scores.write_parquet)
    atomic_write(
        scores_path(data_dir).with_name("relevance_scores.parquet"),
        pair_scores.with_columns(
            zone=pl.Series(
                heads["relevance"].thresholds.zone(pair_scores["reject_score"].to_numpy())
            ),
            model_version=pl.lit(version),
        ).write_parquet,
    )
    if reports["watermark"]["truth"] == "gemini":
        _write_representative_flags(data_dir, image_scores)
    else:
        log.warning("watermark head is zero-shot – representative_flags left unchanged")

    report = {
        "model_version": version,
        "embed_model": model,
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
        "targets": {"precision_reject": PRECISION, "recall_reject": RECALL},
        "escalation_budget": ESCALATION_BUDGET,
        "escalation_rate": round(float(escalated.mean()), 4),
        "heads": reports,
    }
    atomic_write_text(out / "report.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _model_version(model: str, seed: int, data_dir: Path) -> str:
    """<embedding model>.<hash of code version, seed, embeddings and every input table>: same
    inputs, same version; new labels, entities, images or code, a new version."""
    digest = hashlib.sha256(f"{CODE_VERSION}:{seed}".encode())
    parts = sorted(embedding_path(data_dir, model).glob("part-*.parquet"))
    digest.update("".join(f"{p.name}:{p.stat().st_size};" for p in parts).encode())
    for path in (
        gemini_labels_path(data_dir),
        data_dir / "ingest" / "labels.parquet",
        data_dir / "ingest" / "entities.parquet",
        assets_path(data_dir),
        labels_human_path(review_dir(data_dir)),
    ):
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"{model_dir_name(model)}.{digest.hexdigest()[:10]}"
