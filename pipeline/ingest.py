"""Merge the human-verified CSV exports into the `entities`, `images` and `labels` tables.

Merge rules (CLAUDE.md "Decisions"):
- labels are keyed by (image_url, entity_id); `images` is deduplicated by URL;
- when exports disagree the newer export family wins, and the pair is flagged `label_conflict`
  and listed in conflicts.csv (excluded from training/eval downstream);
- an empty `verify` is `unlabeled` unless another export labels the pair.

Deviation from the spec's §7 DBML: `images` holds one row per URL with no `entity_id`, because
~7k URLs belong to several entities. Per-(image, entity) fields (relevance label, query_type,
note, annotator) live in `labels`; join on `image_id` to get an image's entities. URLs are only
whitespace-stripped (e.g. http:// and https:// variants stay separate rows); exact duplicates
are removed later by hashing (#10).

Usage: python -m pipeline.ingest [--raw-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from pipeline import config

log = logging.getLogger(__name__)

# Export families, newest (highest priority) first.
EXPORT_TIERS = {"Cuong_entity_new": 3, "bo_sung": 2, "Cuong_phan_cong": 1}
REQUIRED_COLUMNS = ("entity_id", "image_url", "verify")
OPTIONAL_COLUMNS = (
    "entity_name",
    "entity_slug",
    "branch",
    "leaf_label",
    "description",
    "priority",
    "score",
    "page_url",
    "page_title",
    "image_source",
    "image_width",
    "image_height",
    "query_type",
    "assigned_member",
    "note",
    "ghi_chu",  # entity-level QA note on the description (constant per entity)
    "nguon",  # crawl batch name (bo_sung only)
    "crawled_at",  # bo_sung only
)
VERIFY_VALUES = {"yes", "no"}
UNLABELED = "unlabeled"
TABLES = ("entities", "images", "labels")


class IngestError(ValueError):
    pass


@dataclass(frozen=True)
class IngestResult:
    entities: pl.DataFrame
    images: pl.DataFrame
    labels: pl.DataFrame
    conflicts: pl.DataFrame
    source_files: list[str]


def image_id_for(url: str) -> int:
    """Stable positive int64 id derived from the URL (same id on every run and machine)."""
    return int.from_bytes(hashlib.sha256(url.encode()).digest()[:8], "big") & (2**63 - 1)


def ingest(raw_dir: Path) -> IngestResult:
    files = sorted(raw_dir.glob("*_verified.csv"))
    if not files:
        raise IngestError(f"no *_verified.csv files in {raw_dir}")
    rows = pl.concat([_read_export(f) for f in files]).with_row_index("row_order")
    rows = rows.with_columns(
        pl.col("image_url").map_elements(image_id_for, return_dtype=pl.Int64).alias("image_id")
    )
    if rows.select("image_id").n_unique() != rows.select("image_url").n_unique():
        raise IngestError("image_id hash collision between different URLs")

    # Preferred row first: newest export family, then file name, then row order in the file.
    ranked = rows.sort(["tier", "source", "row_order"], descending=[True, False, False])
    labels, conflicts = _labels(ranked)
    return IngestResult(
        entities=_entities(ranked, labels),
        images=_images(ranked),
        labels=labels,
        conflicts=conflicts,
        source_files=[f.name for f in files],
    )


def _tier(path: Path) -> int:
    for prefix, tier in EXPORT_TIERS.items():
        if path.name.startswith(prefix):
            return tier
    raise IngestError(
        f"{path.name}: unknown export family; expected a name starting with one of "
        f"{sorted(EXPORT_TIERS)}"
    )


def _read_export(path: Path) -> pl.DataFrame:
    tier = _tier(path)
    frame = pl.read_csv(path, infer_schema=False, encoding="utf8")
    frame = frame.rename({c: c.lstrip("﻿").strip() for c in frame.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise IngestError(f"{path.name}: missing required column(s) {missing}")

    frame = frame.select(
        *(pl.col(c).str.strip_chars() for c in REQUIRED_COLUMNS),
        *(
            pl.col(c).str.strip_chars() if c in frame.columns else pl.lit(None, pl.String).alias(c)
            for c in OPTIONAL_COLUMNS
        ),
    ).with_columns(
        pl.col("verify").str.to_lowercase().replace("", None),
        pl.lit(path.name).alias("source"),
        pl.lit(tier).alias("tier"),
    )
    bad = frame.filter(pl.col("verify").is_not_null() & ~pl.col("verify").is_in(VERIFY_VALUES))
    if bad.height:
        raise IngestError(
            f"{path.name}: unexpected verify values {sorted(set(bad['verify']))}; "
            f"expected {sorted(VERIFY_VALUES)} or empty"
        )
    empty_keys = frame.filter(
        pl.col("image_url").fill_null("").eq("") | pl.col("entity_id").fill_null("").eq("")
    )
    if empty_keys.height:
        raise IngestError(f"{path.name}: {empty_keys.height} row(s) without image_url/entity_id")
    log.info("Read %s: %d rows (tier %d)", path.name, frame.height, tier)
    return frame


def _labels(ranked: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per (image_url, entity_id). `ranked` is sorted best-row-first, and group_by with
    maintain_order keeps that order inside each group, so `.first()` is the winning row."""
    pair = ["image_url", "entity_id"]
    in_top_tier = pl.col("tier") == pl.col("tier").first()
    top_values = pl.col("verify").filter(in_top_tier)
    votes = (
        ranked.filter(pl.col("verify").is_not_null())
        .group_by(pair, maintain_order=True)
        .agg(
            (pl.col("verify").n_unique() > 1).alias("label_conflict"),
            # Disagreement inside the winning family can't be resolved by recency: be conservative.
            pl.when(top_values.n_unique() > 1)
            .then(pl.lit("no"))
            .otherwise(top_values.first())
            .alias("value"),
            pl.concat_str(pl.col("source"), pl.lit("="), pl.col("verify"))
            .unique()
            .sort()
            .str.join(";")
            .alias("votes"),
            pl.col("source").first().alias("label_source"),
            pl.col("assigned_member").first().alias("label_annotator"),
            pl.col("note").first().alias("label_note"),
            pl.col("query_type").first().alias("label_query_type"),
        )
    )
    best = ranked.group_by(pair, maintain_order=True).first()
    labels = (
        best.join(votes, on=pair, how="left", maintain_order="left")
        .select(
            "image_id",
            "image_url",
            "entity_id",
            pl.lit("relevance").alias("criterion"),
            pl.col("value").fill_null(UNLABELED),
            pl.col("label_conflict").fill_null(False),
            pl.coalesce("label_query_type", "query_type").alias("query_type"),
            pl.coalesce("label_source", "source").alias("source"),
            pl.coalesce("label_annotator", "assigned_member").alias("annotator"),
            pl.coalesce("label_note", "note").alias("note"),
            "votes",
        )
        .sort(pair)
    )
    conflicts = labels.filter("label_conflict").select(
        "image_id", "image_url", "entity_id", "value", pl.col("votes").alias("labels")
    )
    return labels.drop("votes"), conflicts


def _first_per(ranked: pl.DataFrame, key: str) -> pl.DataFrame:
    """The best-ranked row for each value of `key`, sorted by `key`."""
    return ranked.group_by(key, maintain_order=True).first().sort(key)


def _images(ranked: pl.DataFrame) -> pl.DataFrame:
    return _first_per(ranked, "image_url").select(
        "image_id",
        "image_url",
        "page_url",
        "page_title",
        "image_source",
        pl.col("image_width").cast(pl.Int32, strict=False).alias("declared_w"),
        pl.col("image_height").cast(pl.Int32, strict=False).alias("declared_h"),
        pl.col("nguon").alias("crawl_batch"),
        "crawled_at",
    )


def _entities(ranked: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    # Spec §2.1: entities with < 5 relevant images fall back to text-only matching.
    n_yes = (
        labels.filter((pl.col("value") == "yes") & ~pl.col("label_conflict"))
        .group_by("entity_id")
        .agg(pl.len().cast(pl.Int32).alias("n_yes_images"))
    )
    return (
        _first_per(ranked, "entity_id")
        .join(n_yes, on="entity_id", how="left", maintain_order="left")
        .select(
            "entity_id",
            "entity_name",
            "entity_slug",
            "branch",
            "leaf_label",
            "description",
            pl.col("ghi_chu").alias("description_note"),
            "priority",
            pl.col("score").cast(pl.Float64, strict=False),
            pl.col("n_yes_images").fill_null(0),
            # Filled later by the KG step (#18) and Gemini visual attributes (spec §7).
            pl.lit(None, pl.String).alias("province"),
            pl.lit(None, pl.List(pl.String)).alias("aliases"),
            pl.lit(None, pl.String).alias("visual_attributes"),
        )
    )


def write_outputs(result: IngestResult, out_dir: Path) -> None:
    """Write Parquet tables + conflicts.csv, and (re)build the DuckDB database from them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(result, name).write_parquet(out_dir / f"{name}.parquet")
    result.conflicts.write_csv(out_dir / "conflicts.csv")

    db_path = out_dir / "vsf.duckdb"
    db_path.unlink(missing_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for name in TABLES:
            parquet = str(out_dir / f"{name}.parquet")
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet(?)", [parquet])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.ingest")
    parser.add_argument("--raw-dir", type=Path, default=config.raw_data_dir())
    parser.add_argument("--out-dir", type=Path, default=config.data_dir() / "ingest")
    args = parser.parse_args()

    result = ingest(args.raw_dir.expanduser())
    write_outputs(result, args.out_dir)
    values = dict(result.labels.group_by("value").len().rows())
    log.info(
        "Ingested %d files → %d images, %d entities, %d (image, entity) pairs "
        "(%d labeled: yes=%d no=%d; %d unlabeled), %d conflicting pairs → %s",
        len(result.source_files),
        result.images.height,
        result.entities.height,
        result.labels.height,
        result.labels.height - values.get(UNLABELED, 0),
        values.get("yes", 0),
        values.get("no", 0),
        values.get(UNLABELED, 0),
        result.conflicts.height,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
