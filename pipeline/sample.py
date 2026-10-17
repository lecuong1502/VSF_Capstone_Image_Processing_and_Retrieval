"""Build the committed `data/sample/` fixture from the raw exports.

The sample keeps the raw CSV layouts (so `pipeline.ingest` runs on it end-to-end, like CI does)
but is safe for the public repo: landmark/scenery entities only (nothing about people) and no
annotator columns. Selection is deterministic, so re-running produces identical files.

Usage: python -m pipeline.sample [--raw-dir DIR] [--out-dir data/sample]
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pipeline import config
from pipeline.ingest import PAIR_KEY, ingest, public_file_name, read_raw_csv

log = logging.getLogger(__name__)

# Who labeled what, and their free-text comments, never leave the private dataset.
ANNOTATOR_COLUMNS = ("assigned_member", "verified_at", "verification", "ghi_chu", "note")

# Matched against the start of the leaf category (before any "(…)" place list or "— region").
_LANDMARK_CATEGORY = re.compile(
    r"^(hang|động(?! vật)|vịnh|bãi biển|bãi|núi|đèo|đồi|thác|hồ|đầm|đảo|bán đảo|thung lũng|"
    r"cao nguyên|ruộng|rừng|vườn quốc gia|sông|suối|chùa|đình|phố cổ|cầu)\b",
    re.IGNORECASE,
)
_PEOPLE = re.compile(
    r"\b(lăng|mộ|tưởng niệm|chủ tịch|tổng bí thư|bác hồ|anh hùng|liệt sĩ|nhân vật|chân dung|"
    r"người|tượng|ông|bà|vua|danh nhân)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quotas:
    """Number of (image, entity) pairs to take per label kind."""

    yes: int = 16
    no: int = 16
    unlabeled: int = 6
    conflicts: int = 3


def is_landmark(entity_name: str, leaf_label: str) -> bool:
    """A scenery/landmark category whose name and category don't refer to people."""
    entity_name = unicodedata.normalize("NFC", entity_name)
    leaf_label = unicodedata.normalize("NFC", leaf_label)
    category = re.split(r"[(—–]", leaf_label, maxsplit=1)[0].strip()
    return bool(_LANDMARK_CATEGORY.match(category)) and not _PEOPLE.search(
        f"{entity_name} {category}"
    )


DEFAULT_QUOTAS = Quotas()


def build_sample(raw_dir: Path, out_dir: Path, quotas: Quotas = DEFAULT_QUOTAS) -> None:
    """Replace the CSVs in `out_dir` with the rows of every picked (image_url, entity_id) pair,
    copied from each raw export that contains it (so cross-export conflicts are reproduced)."""
    result = ingest(raw_dir)
    landmark_ids = [
        e["entity_id"]
        for e in result.entities.iter_rows(named=True)
        if is_landmark(e["entity_name"] or "", e["leaf_label"] or "")
    ]
    candidates = result.labels.filter(pl.col("entity_id").is_in(landmark_ids)).sort(
        "image_id", "entity_id"
    )
    picks = pl.concat(
        [
            _round_robin(candidates.filter("label_conflict"), quotas.conflicts),
            *(
                _round_robin(
                    candidates.filter(~pl.col("label_conflict") & (pl.col("value") == value)), n
                )
                for value, n in (
                    ("yes", quotas.yes),
                    ("no", quotas.no),
                    ("unlabeled", quotas.unlabeled),
                )
            ),
        ]
    ).select(PAIR_KEY)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    for name in result.source_files:
        raw = read_raw_csv(raw_dir / name)
        rows = raw.filter(_in_pairs(raw.select(PAIR_KEY), picks))
        rows = rows.drop([c for c in ANNOTATOR_COLUMNS if c in rows.columns])
        if rows.height:
            # Raw file names contain the annotator's name; the public sample uses neutral ones.
            rows.write_csv(out_dir / public_file_name(name))
            log.info("%s -> %s: %d rows", name, public_file_name(name), rows.height)


def _in_pairs(keys: pl.DataFrame, picks: pl.DataFrame) -> pl.Series:
    """Boolean mask: which raw rows belong to a picked (image_url, entity_id) pair."""
    marked = keys.with_row_index("_row").join(
        picks.with_columns(pl.lit(True).alias("_picked")),
        on=list(PAIR_KEY),
        how="left",
    )
    return marked.sort("_row")["_picked"].fill_null(False)


def _round_robin(labels: pl.DataFrame, n: int) -> pl.DataFrame:
    """Take `n` rows, alternating between source files so every export family is represented."""
    # `labels` is fully sorted by (image_id, entity_id), so ranks within a source are stable.
    ranked = labels.with_columns(pl.int_range(pl.len()).over("source").alias("_rank"))
    return ranked.sort("_rank", "source", "image_id", "entity_id").head(n).drop("_rank")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.sample")
    parser.add_argument("--raw-dir", type=Path, default=config.raw_data_dir())
    parser.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "data" / "sample")
    args = parser.parse_args()
    build_sample(args.raw_dir.expanduser(), args.out_dir)


if __name__ == "__main__":
    main()
