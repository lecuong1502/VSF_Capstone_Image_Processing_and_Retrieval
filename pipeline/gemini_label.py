"""Gemini moderation labels for a stratified image sample (spec §4.1 T4, §10 task 5).

Gemini is the teacher for the criteria the human `verify` column doesn't cover: each sampled image
(its ≤384 px thumbnail) is labeled against one of its entities with the spec prompt and JSON
schema through the Batch API. The labels train the heads (#14); a blind human audit of 500 of them
in /review gives Cohen's kappa per criterion.

Output in `$DATA_DIR/gemini_labels/`:
- `sample.parquet`: the stratified sample (by entity branch × verify), in labeling order.
- `labels.parquet`: one row per labeled image – criteria, confidence, reason, tokens, error.
- `labels_gemini.parquet`: the same in the `labels` shape (criterion/value, source='gemini').
- `kappa.parquet` / `kappa.json`: Gemini vs. the human audit.

Usage: python -m pipeline.gemini_label run [--n 5000] [--limit 100]   # --limit: paid pilot
       python -m pipeline.gemini_label audit-queue [--n 500]          # → /review queue
       python -m pipeline.gemini_label kappa                          # after the audit
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from pipeline import config
from pipeline.download import assets_path
from pipeline.files import atomic_write, atomic_write_text
from pipeline.gemini import GeminiClient, GeminiRequest, Image
from pipeline.review import builders
from pipeline.review.items import write_queue
from pipeline.review.store import export_queue

log = logging.getLogger(__name__)

STEP = "gemini_label"
AUDIT_QUEUE = "gemini-audit"
BOOLEAN_CRITERIA = ("watermark", "text_overlay", "nsfw", "political", "violence")
CRITERIA = (*BOOLEAN_CRITERIA, "relevance", "shot_type")
# JSON with a ≤120-char reason is ~100 tokens; the rest is headroom for MINIMAL thinking.
MAX_OUTPUT_TOKENS = 384

# Spec §4.1 prompt, verbatim (the backslash only wraps the source line).
PROMPT = """Bạn là bộ kiểm duyệt ảnh du lịch Việt Nam. Thực thể dự kiến: "{entity_name}" — \
{leaf_label}.
Đánh giá: watermark (logo/chữ website đè ảnh), text_overlay (chữ quảng cáo/khung), nsfw,
political (cờ/biểu tượng/khẩu hiệu chính trị, biểu tình, nội dung phản động, bản đồ sai chủ quyền),
violence, relevance (ảnh có đúng thực thể không: yes/no/unsure), shot_type.
Chỉ dựa vào nội dung nhìn thấy; không chắc thì đặt confidence thấp. Trả JSON đúng schema."""


class ModerationLabel(BaseModel):
    """Spec §4.1 response schema."""

    watermark: bool
    text_overlay: bool
    nsfw: bool
    political: bool
    violence: bool
    relevance: Literal["yes", "no", "unsure"]
    shot_type: Literal["PHONG_CANH", "CHECKIN", "FLYCAM", "KHAC"]
    confidence: float
    reason: str = Field(default="", max_length=120)


LABELS_SCHEMA = pl.Schema(
    {
        "image_id": pl.Int64(),
        "entity_id": pl.String(),
        **{c: pl.Boolean() for c in BOOLEAN_CRITERIA},
        "relevance": pl.String(),
        "shot_type": pl.String(),
        "confidence": pl.Float64(),
        "reason": pl.String(),
        "model": pl.String(),
        "input_tokens": pl.Int64(),
        "output_tokens": pl.Int64(),
        "error": pl.String(),
    }
)


def out_dir(data_dir: Path) -> Path:
    return data_dir / "gemini_labels"


def labels_path(data_dir: Path) -> Path:
    return out_dir(data_dir) / "labels.parquet"


def sample_images(data_dir: Path, *, n: int, seed: int) -> pl.DataFrame:
    """`n` reviewable images, each with its judging entity, stratified proportionally by entity
    branch × verify label (largest remainder), shuffled into a fixed labeling order."""
    one = builders.one_entity_per_image(builders.labeled_pairs(data_dir))
    entities = pl.read_parquet(data_dir / "ingest" / "entities.parquet").select(
        "entity_id", "entity_name", "leaf_label", "branch"
    )
    pool = (
        one.join(entities, on="entity_id")
        .select(
            "image_id",
            "entity_id",
            "entity_name",
            "leaf_label",
            "branch",
            pl.col("value").alias("verify"),
        )
        .sort("image_id")
    )
    n = min(n, pool.height)
    strata = pool.group_by("branch", "verify", maintain_order=True).len().sort("branch", "verify")
    exact = [count * n / pool.height for count in strata["len"]]
    quota = [int(x) for x in exact]
    for i in sorted(range(len(exact)), key=lambda i: exact[i] - quota[i], reverse=True)[
        : n - sum(quota)
    ]:
        quota[i] += 1
    parts = [
        pool.filter((pl.col("branch") == b) & (pl.col("verify") == v)).sample(
            n=q, seed=seed, shuffle=True
        )
        for (b, v), q in zip(strata.select("branch", "verify").rows(), quota, strict=True)
    ]
    return pl.concat(parts).sample(fraction=1.0, seed=seed, shuffle=True)


def build_request(row: dict[str, Any], image: bytes) -> GeminiRequest:
    prompt = PROMPT.format(entity_name=row["entity_name"], leaf_label=row["leaf_label"] or "")
    return GeminiRequest(
        prompt=prompt,
        images=(Image(data=image, mime_type="image/webp"),),
        schema=ModerationLabel,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


@dataclass
class Summary:
    requested: int
    labeled: int
    failed: int
    input_tokens: int
    output_tokens: int
    usd: float  # paid in this run (cache hits are free)
    usd_per_image: float  # tokens of every requested image at the batch price, paid or cached

    @property
    def input_tokens_per_image(self) -> float:
        return self.input_tokens / self.requested if self.requested else 0.0

    @property
    def output_tokens_per_image(self) -> float:
        return self.output_tokens / self.requested if self.requested else 0.0


def run_labeling(
    data_dir: Path, client: GeminiClient, *, n: int, seed: int, limit: int | None = None
) -> Summary:
    """Label the first `limit` (pilot) or all `n` sampled images. Repeats are served from the
    client's cache, so a pilot followed by the full run pays for each image once."""
    sample = sample_images(data_dir, n=n, seed=seed)
    out = out_dir(data_dir)
    atomic_write(out / "sample.parquet", sample.write_parquet)
    todo = sample.head(limit) if limit is not None else sample
    thumbs = dict(pl.read_parquet(assets_path(data_dir)).select("image_id", "thumb_path").rows())
    rows = todo.to_dicts()
    requests = [build_request(r, (data_dir / thumbs[r["image_id"]]).read_bytes()) for r in rows]
    log.info("Labeling %d images with %s (Batch API)", len(requests), client.model)
    results = client.batch(STEP, requests)

    records = []
    for row, result in zip(rows, results, strict=True):
        record: dict[str, Any] = {
            "image_id": row["image_id"],
            "entity_id": row["entity_id"],
            "model": client.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "error": result.error,
        }
        if result.ok:
            record |= result.parse(ModerationLabel).model_dump()
        records.append(record)
    labels = pl.DataFrame(records, schema=LABELS_SCHEMA)
    atomic_write(labels_path(data_dir), labels.write_parquet)
    atomic_write(out / "labels_gemini.parquet", to_labels_shape(labels).write_parquet)

    input_tokens = int(labels["input_tokens"].sum())
    output_tokens = int(labels["output_tokens"].sum())
    projected = client.pricing.cost(input_tokens, output_tokens, batch=True)
    return Summary(
        requested=len(rows),
        labeled=int(labels["error"].is_null().sum()),
        failed=int(labels["error"].is_not_null().sum()),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=sum(r.usd for r in results),
        usd_per_image=projected / len(rows) if rows else 0.0,
    )


def to_labels_shape(labels: pl.DataFrame) -> pl.DataFrame:
    """One row per (image, criterion) like ingest `labels`: booleans become yes/no."""
    ok = labels.filter(pl.col("error").is_null())
    values = [
        pl.when(pl.col(c)).then(pl.lit("yes")).otherwise(pl.lit("no")).alias(c)
        for c in BOOLEAN_CRITERIA
    ]
    return (
        ok.with_columns(values)
        .unpivot(
            on=list(CRITERIA),
            index=["image_id", "entity_id", "model", "confidence"],
            variable_name="criterion",
            value_name="value",
        )
        .with_columns(source=pl.lit("gemini"), annotator=pl.col("model"))
        .select(
            "image_id",
            "entity_id",
            "criterion",
            "value",
            "source",
            "annotator",
            "confidence",
        )
        .sort("image_id", "criterion")
    )


def audit_sample(labels: pl.DataFrame, *, n: int, seed: int, per_positive: int) -> list[int]:
    """`n` labeled image_ids for the human audit: up to `per_positive` Gemini positives per
    rare criterion first (kappa is undefined without positives), the rest at random."""
    ok = labels.filter(pl.col("error").is_null()).sort("image_id")
    rare = [*BOOLEAN_CRITERIA, "relevance"]
    picked: list[int] = []
    for criterion in rare:
        positive = pl.col(criterion) if criterion != "relevance" else pl.col(criterion) != "yes"
        ids = ok.filter(positive & ~pl.col("image_id").is_in(picked))["image_id"]
        picked += ids.sample(n=min(per_positive, ids.len()), seed=seed).to_list()
    rest = ok.filter(~pl.col("image_id").is_in(picked))["image_id"]
    picked = picked[:n]
    picked += rest.sample(n=min(n - len(picked), rest.len()), seed=seed).to_list()
    return sorted(picked)


def write_audit_queue(
    data_dir: Path, review_dir: Path, *, image_ids: list[int], name: str = AUDIT_QUEUE
) -> Path:
    """Blind /review queue over every criterion for `image_ids`, judged against the same entity
    as the Gemini prompt (`builders.one_entity_per_image`)."""
    items = builders.moderation_queue(data_dir, n=len(image_ids), seed=0, image_ids=image_ids)
    return write_queue(review_dir, name, items)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa of two raters; None when chance agreement is 1 (a single class)."""
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def kappa_table(data_dir: Path, review_dir: Path, queue: str = AUDIT_QUEUE) -> pl.DataFrame:
    """Per criterion: n compared (both sure), agreement, kappa and positives per rater."""
    human = export_queue(review_dir, queue).select("image_id", "criterion", "value")
    gemini = to_labels_shape(pl.read_parquet(labels_path(data_dir))).select(
        "image_id", "criterion", pl.col("value").alias("gemini")
    )
    joined = human.join(gemini, on=["image_id", "criterion"]).filter(
        (pl.col("value") != "unsure") & (pl.col("gemini") != "unsure")
    )
    rows = []
    for criterion in CRITERIA:
        pairs = joined.filter(pl.col("criterion") == criterion)
        g, h = pairs["gemini"].to_list(), pairs["value"].to_list()
        positive = "PHONG_CANH" if criterion == "shot_type" else "yes"
        if criterion == "relevance":
            positive = "no"  # the rare, rejecting class
        rows.append(
            {
                "criterion": criterion,
                "n": len(g),
                "agreement": sum(x == y for x, y in zip(g, h, strict=True)) / len(g) if g else None,
                "kappa": cohen_kappa(g, h),
                "gemini_positive": sum(x == positive for x in g),
                "human_positive": sum(x == positive for x in h),
            }
        )
    schema = {"agreement": pl.Float64, "kappa": pl.Float64}
    return pl.DataFrame(rows, schema_overrides=schema)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.gemini_label")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="label the sample with Gemini (paid, Batch API)")
    run.add_argument("--n", type=int, default=5000)
    run.add_argument("--limit", type=int, help="label only the first N of the sample (pilot)")
    run.add_argument("--seed", type=int, default=0)
    audit = sub.add_parser("audit-queue", help="blind /review queue for the human audit")
    audit.add_argument("--n", type=int, default=500)
    audit.add_argument("--per-positive", type=int, default=40)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--force", action="store_true", help="replace an existing audit queue")
    sub.add_parser("kappa", help="Cohen's kappa per criterion, Gemini vs. the audit")
    args = parser.parse_args()

    data_dir = config.data_dir()
    review_dir = data_dir / "review"
    if args.command == "run":
        from pipeline.gemini.sdk import client_from_env

        summary = run_labeling(
            data_dir, client_from_env(), n=args.n, seed=args.seed, limit=args.limit
        )
        log.info(
            "%d requested, %d labeled, %d failed | %.0f in + %.0f out tokens/image | "
            "$%.4f paid now | $%.6f/image → $%.2f for %d images",
            summary.requested,
            summary.labeled,
            summary.failed,
            summary.input_tokens_per_image,
            summary.output_tokens_per_image,
            summary.usd,
            summary.usd_per_image,
            summary.usd_per_image * args.n,
            args.n,
        )
        atomic_write_text(out_dir(data_dir) / "run.json", json.dumps(asdict(summary), indent=2))
    elif args.command == "audit-queue":
        if (review_dir / "queues" / f"{AUDIT_QUEUE}.jsonl").exists() and not args.force:
            parser.error(f"queue {AUDIT_QUEUE} exists (its decisions are kept); pass --force")
        labels = pl.read_parquet(labels_path(data_dir))
        ids = audit_sample(labels, n=args.n, seed=args.seed, per_positive=args.per_positive)
        log.info(
            "Wrote %d items to %s", len(ids), write_audit_queue(data_dir, review_dir, image_ids=ids)
        )
    else:
        table = kappa_table(data_dir, review_dir)
        atomic_write(out_dir(data_dir) / "kappa.parquet", table.write_parquet)
        atomic_write_text(out_dir(data_dir) / "kappa.json", json.dumps(table.to_dicts(), indent=2))
        with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, float_precision=3):
            log.info("Gemini vs. human audit:\n%s", table)


if __name__ == "__main__":
    main()
