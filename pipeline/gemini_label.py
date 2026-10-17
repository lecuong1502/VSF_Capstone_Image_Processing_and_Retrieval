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
from pipeline.review.items import queue_path, review_dir, write_queue
from pipeline.review.store import export_queue

log = logging.getLogger(__name__)

STEP = "gemini_label"
AUDIT_QUEUE = "gemini-audit"
BOOLEAN_CRITERIA = ("watermark", "text_overlay", "nsfw", "political", "violence")
CRITERIA = (*BOOLEAN_CRITERIA, "relevance", "shot_type")
# The rare class per criterion (in the `labels` shape): audits are enriched with Gemini's
# positives, since kappa is undefined without them. shot_type is multi-class.
POSITIVE = {**dict.fromkeys(BOOLEAN_CRITERIA, "yes"), "relevance": "no"}
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
    judged = builders.one_entity_per_image(builders.labeled_pairs(data_dir))
    entities = pl.read_parquet(data_dir / "ingest" / "entities.parquet").select(
        "entity_id", "entity_name", "leaf_label", pl.col("branch").fill_null("?")
    )
    pool = (
        judged.join(entities, on="entity_id")
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
    skipped_failed: int  # failed in an earlier run and not retried (no --retry-failed)
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
    data_dir: Path,
    client: GeminiClient,
    *,
    n: int,
    seed: int,
    limit: int | None = None,
    retry_failed: bool = False,
) -> Summary:
    """Label the first `limit` (pilot) or all `n` sampled images and merge them into
    `labels.parquet`. Successes are served from the client's cache on later runs, so a pilot
    followed by the full run pays for each image once; failures are not cached, so they are
    skipped unless `retry_failed` (a truncated answer would likely fail – and be paid – again)."""
    sample = sample_images(data_dir, n=n, seed=seed)
    out = out_dir(data_dir)
    atomic_write(out / "sample.parquet", sample.write_parquet)
    previous = (
        pl.read_parquet(labels_path(data_dir))
        if labels_path(data_dir).exists()
        else pl.DataFrame(schema=LABELS_SCHEMA)
    )
    todo = sample.head(limit) if limit is not None else sample
    skipped = 0
    if not retry_failed:
        failed_before = previous.filter(pl.col("error").is_not_null()).select(
            "image_id", "entity_id"
        )
        kept = todo.join(failed_before, on=["image_id", "entity_id"], how="anti")
        skipped = todo.height - kept.height
        todo = kept
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
    merged = pl.concat(
        [previous.join(labels.select("image_id"), on="image_id", how="anti"), labels]
    ).sort("image_id")
    atomic_write(labels_path(data_dir), merged.write_parquet)
    atomic_write(out / "labels_gemini.parquet", to_labels_shape(merged).write_parquet)

    input_tokens = int(labels["input_tokens"].sum())
    output_tokens = int(labels["output_tokens"].sum())
    projected = client.pricing.cost(input_tokens, output_tokens, batch=True)
    return Summary(
        requested=len(rows),
        labeled=int(labels["error"].is_null().sum()),
        failed=int(labels["error"].is_not_null().sum()),
        skipped_failed=skipped,
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


def audit_sample(labels: pl.DataFrame, *, n: int, seed: int, per_positive: int) -> pl.DataFrame:
    """`n` labeled images for the human audit (`image_id`, `stratum`): up to `per_positive`
    Gemini positives per rare criterion (at most n / #criteria each, so every criterion gets
    some), then a uniform random rest. Kappa on the `random` stratum estimates the population;
    on all rows it is enriched toward positives."""
    long = to_labels_shape(labels)
    per = min(per_positive, n // len(POSITIVE))
    picked: dict[int, str] = {}
    for criterion, positive in POSITIVE.items():
        ids = long.filter(
            (pl.col("criterion") == criterion)
            & (pl.col("value") == positive)
            & ~pl.col("image_id").is_in(list(picked))
        )["image_id"].sort()
        for image_id in ids.sample(n=min(per, ids.len()), seed=seed).to_list():
            picked[image_id] = f"positive:{criterion}"
    rest = long.filter(~pl.col("image_id").is_in(list(picked)))["image_id"].unique().sort()
    for image_id in rest.sample(n=min(n - len(picked), rest.len()), seed=seed).to_list():
        picked[image_id] = "random"
    return pl.DataFrame(
        {"image_id": list(picked), "stratum": list(picked.values())},
        schema={"image_id": pl.Int64, "stratum": pl.String},
    ).sort("image_id")


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
    observed = _agreement(a, b) or 0.0
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def audit_ids_path(data_dir: Path) -> Path:
    return out_dir(data_dir) / "audit_ids.parquet"


def kappa_table(data_dir: Path, review_dir: Path, queue: str = AUDIT_QUEUE) -> pl.DataFrame:
    """Per criterion: pairs compared (both raters sure, same image *and* entity), agreement and
    kappa – on all audited rows and on the uniform `random` stratum – plus positives per rater."""
    key = ["image_id", "entity_id", "criterion"]
    human = export_queue(review_dir, queue).select(*key, "value")
    gemini = to_labels_shape(pl.read_parquet(labels_path(data_dir))).select(
        *key, pl.col("value").alias("gemini")
    )
    strata = audit_ids_path(data_dir)
    stratum = (
        pl.read_parquet(strata)
        if strata.exists()
        else pl.DataFrame(schema={"image_id": pl.Int64, "stratum": pl.String})
    )
    joined = (
        human.join(gemini, on=key)
        .join(stratum, on="image_id", how="left")
        .filter((pl.col("value") != "unsure") & (pl.col("gemini") != "unsure"))
    )
    rows = []
    for criterion in CRITERIA:
        pairs = joined.filter(pl.col("criterion") == criterion)
        uniform = pairs.filter(pl.col("stratum") == "random")
        g, h = pairs["gemini"].to_list(), pairs["value"].to_list()
        positive = POSITIVE.get(criterion)
        rows.append(
            {
                "criterion": criterion,
                "n": len(g),
                "agreement": _agreement(g, h),
                "kappa": cohen_kappa(g, h),
                "n_random": uniform.height,
                "kappa_random": cohen_kappa(
                    uniform["gemini"].to_list(), uniform["value"].to_list()
                ),
                "gemini_positive": sum(x == positive for x in g) if positive else None,
                "human_positive": sum(x == positive for x in h) if positive else None,
            }
        )
    schema = {
        "agreement": pl.Float64,
        "kappa": pl.Float64,
        "kappa_random": pl.Float64,
        "gemini_positive": pl.Int64,
        "human_positive": pl.Int64,
    }
    return pl.DataFrame(rows, schema_overrides=schema)


def _agreement(a: Sequence[str], b: Sequence[str]) -> float | None:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.gemini_label")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="label the sample with Gemini (paid, Batch API)")
    run.add_argument("--n", type=int, default=5000)
    run.add_argument("--limit", type=int, help="label only the first N of the sample (pilot)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--retry-failed", action="store_true", help="re-request earlier failures")
    run.add_argument(
        "--skip-pilot", action="store_true", help="allow a full run without a --limit pilot first"
    )
    audit_p = sub.add_parser("audit-queue", help="blind /review queue for the human audit")
    audit_p.add_argument("--n", type=int, default=500)
    audit_p.add_argument("--per-positive", type=int, default=40)
    audit_p.add_argument("--seed", type=int, default=0)
    audit_p.add_argument("--force", action="store_true", help="replace an existing audit queue")
    sub.add_parser("kappa", help="Cohen's kappa per criterion, Gemini vs. the audit")
    args = parser.parse_args()

    data_dir = config.data_dir()
    reviews = review_dir(data_dir)
    pilot = out_dir(data_dir) / "pilot.json"
    if args.command == "run":
        if args.limit is None and not pilot.exists() and not args.skip_pilot:
            parser.error(
                "measure real tokens first (spec §4.1 T4): run with --limit 100 and check the "
                "projected cost, or pass --skip-pilot"
            )
        from pipeline.gemini.sdk import client_from_env

        summary = run_labeling(
            data_dir,
            client_from_env(),
            n=args.n,
            seed=args.seed,
            limit=args.limit,
            retry_failed=args.retry_failed,
        )
        log.info(
            "%d requested, %d labeled, %d failed, %d earlier failures skipped | "
            "%.0f in + %.0f out tokens/image | $%.4f paid now | $%.6f/image → $%.2f for %d images",
            summary.requested,
            summary.labeled,
            summary.failed,
            summary.skipped_failed,
            summary.input_tokens_per_image,
            summary.output_tokens_per_image,
            summary.usd,
            summary.usd_per_image,
            summary.usd_per_image * args.n,
            args.n,
        )
        report = pilot if args.limit is not None else out_dir(data_dir) / "run.json"
        atomic_write_text(report, json.dumps(asdict(summary), indent=2))
    elif args.command == "audit-queue":
        if queue_path(reviews, AUDIT_QUEUE).exists() and not args.force:
            parser.error(f"queue {AUDIT_QUEUE} exists (its decisions are kept); pass --force")
        labels = pl.read_parquet(labels_path(data_dir))
        audit = audit_sample(labels, n=args.n, seed=args.seed, per_positive=args.per_positive)
        atomic_write(audit_ids_path(data_dir), audit.write_parquet)
        path = write_audit_queue(data_dir, reviews, image_ids=audit["image_id"].to_list())
        log.info("Wrote %d items to %s\n%s", audit.height, path, audit["stratum"].value_counts())
    else:
        table = kappa_table(data_dir, reviews)
        atomic_write(out_dir(data_dir) / "kappa.parquet", table.write_parquet)
        atomic_write_text(out_dir(data_dir) / "kappa.json", json.dumps(table.to_dicts(), indent=2))
        with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, float_precision=3):
            log.info("Gemini vs. human audit:\n%s", table)


if __name__ == "__main__":
    main()
