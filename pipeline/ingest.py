"""Merge the human-verified CSV exports into the `entities`, `images` and `labels` tables.

Merge rules (CLAUDE.md "Decisions"):
- labels are keyed by (image_url, entity_id); `images` is deduplicated by URL;
- when exports disagree the newer export family wins, and the pair is flagged `label_conflict`
  and listed in conflicts.csv (excluded from training/eval downstream);
- an empty `verify` is `unlabeled` unless another export labels the pair.

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
        entities=_entities(ranked),
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
        pl.col("image_url").is_null() | (pl.col("image_url") == "") | pl.col("entity_id").is_null()
    )
    if empty_keys.height:
        raise IngestError(f"{path.name}: {empty_keys.height} row(s) without image_url/entity_id")
    log.info("Read %s: %d rows (tier %d)", path.name, frame.height, tier)
    return frame


def _labels(ranked: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pair = ["image_url", "entity_id"]
    labeled = ranked.filter(pl.col("verify").is_not_null())
    top_tier = labeled.group_by(pair).agg(pl.col("tier").max().alias("top_tier"))
    votes = (
        labeled.join(top_tier, on=pair)
        .group_by(pair)
        .agg(
            pl.col("verify").n_unique().alias("n_values"),
            # Disagreement inside the winning family can't be resolved by recency: be conservative.
            pl.when((pl.col("tier") == pl.col("top_tier")).sum() > 0)
            .then(
                pl.when(
                    pl.col("verify").filter(pl.col("tier") == pl.col("top_tier")).n_unique() > 1
                )
                .then(pl.lit("no"))
                .otherwise(pl.col("verify").filter(pl.col("tier") == pl.col("top_tier")).first())
            )
            .alias("value"),
            pl.concat_str(pl.col("source"), pl.lit("="), pl.col("verify"))
            .unique()
            .sort()
            .str.join(";")
            .alias("labels"),
            pl.col("source").first().alias("label_source"),
            pl.col("assigned_member").first().alias("label_annotator"),
            pl.col("note").first().alias("label_note"),
            pl.col("query_type").first().alias("label_query_type"),
        )
    )
    best = ranked.group_by(pair, maintain_order=True).first()
    labels = (
        best.join(votes, on=pair, how="left")
        .select(
            "image_id",
            "image_url",
            "entity_id",
            pl.lit("relevance").alias("criterion"),
            pl.col("value").fill_null(UNLABELED),
            (pl.col("n_values").fill_null(0) > 1).alias("label_conflict"),
            pl.coalesce("label_query_type", "query_type").alias("query_type"),
            pl.coalesce("label_source", "source").alias("source"),
            pl.coalesce("label_annotator", "assigned_member").alias("annotator"),
            pl.coalesce("label_note", "note").alias("note"),
            "labels",
        )
        .sort(pair)
    )
    conflicts = labels.filter("label_conflict").select(
        "image_id", "image_url", "entity_id", "value", "labels"
    )
    return labels.drop("labels"), conflicts


def _images(ranked: pl.DataFrame) -> pl.DataFrame:
    return (
        ranked.group_by("image_url", maintain_order=True)
        .first()
        .select(
            "image_id",
            "image_url",
            "page_url",
            "page_title",
            "image_source",
            pl.col("image_width").cast(pl.Int32, strict=False).alias("declared_w"),
            pl.col("image_height").cast(pl.Int32, strict=False).alias("declared_h"),
        )
        .sort("image_url")
    )


def _entities(ranked: pl.DataFrame) -> pl.DataFrame:
    return (
        ranked.group_by("entity_id", maintain_order=True)
        .first()
        .select(
            "entity_id",
            "entity_name",
            "entity_slug",
            "branch",
            "leaf_label",
            "description",
            "priority",
            pl.col("score").cast(pl.Float64, strict=False),
            # Filled later by the KG step (#18) and Gemini visual attributes.
            pl.lit(None, pl.String).alias("province"),
            pl.lit(None, pl.List(pl.String)).alias("aliases"),
            pl.lit(None, pl.String).alias("visual_attributes"),
        )
        .sort("entity_id")
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
