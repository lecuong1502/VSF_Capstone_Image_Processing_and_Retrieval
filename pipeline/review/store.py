"""Append-only decisions per queue (the latest per item wins), export and throughput stats."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ValidationError

from pipeline.files import atomic_write, atomic_write_text
from pipeline.review.items import Item, list_queues, queue_path, read_queue

# Time on one item counts at most this much toward throughput (the tab may sit idle).
MAX_ITEM_MS = 120_000

# `labels` (spec §7) plus provenance; every queue's export starts with these columns.
LABELS_SCHEMA = pl.Schema(
    {
        "image_id": pl.Int64(),
        "entity_id": pl.String(),
        "criterion": pl.String(),
        "value": pl.String(),
        "source": pl.String(),
        "annotator": pl.String(),
        "item_id": pl.String(),
        "queue": pl.String(),
        "mode": pl.String(),
        "model_version": pl.String(),
        "decided_at": pl.Datetime("us", "UTC"),
    }
)
ALL_LABELS = "labels_human"  # export/labels_human.parquet: every queue, labels columns only


class Decision(BaseModel):
    item_id: str
    values: dict[str, str]
    annotator: str
    decided_at: datetime
    elapsed_ms: int  # time the item was on screen before this decision


class DecisionStore:
    """`decisions/<queue>.jsonl`: one JSON line per decision, flushed and fsynced on write."""

    def __init__(self, review_dir: Path, queue: str) -> None:
        self.path = queue_path(review_dir, queue, kind="decisions")
        self._lock = threading.Lock()

    def append(self, decision: Decision) -> None:
        line = (decision.model_dump_json() + "\n").encode()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab+") as handle:
                end = handle.seek(0, os.SEEK_END)
                if end:  # start on a fresh line if a crash left a torn one
                    handle.seek(end - 1)
                    if handle.read(1) != b"\n":
                        line = b"\n" + line
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def record(self, items: dict[str, Item], decision: Decision) -> None:
        """Validate against the item's fields, then append. KeyError: unknown item."""
        item = items[decision.item_id]
        fields = {f.name: {o.value for o in f.options} for f in item.fields}
        if set(decision.values) != set(fields):
            raise ValueError(f"expected values for {sorted(fields)}")
        for name, value in decision.values.items():
            if value not in fields[name]:
                raise ValueError(f"{name}: {value!r} is not an option")
        self.append(decision)

    def all(self, items: dict[str, Item] | None = None) -> list[Decision]:
        """Decisions in order, optionally only for `items` (a rebuilt queue may drop some)."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_bytes().splitlines():
            try:
                decision = Decision.model_validate_json(line)
            except ValidationError:
                continue  # torn line from a crash
            if items is None or decision.item_id in items:
                out.append(decision)
        return out

    def latest(self, items: dict[str, Item] | None = None) -> dict[str, Decision]:
        return {d.item_id: d for d in self.all(items)}


def items_by_id(review_dir: Path, queue: str) -> dict[str, Item]:
    return {i.item_id: i for i in read_queue(review_dir, queue)}


def export_queue(review_dir: Path, queue: str) -> pl.DataFrame:
    """Latest decisions as labels rows (one per field, source='human') plus the items'
    `export_keys`; writes `export/<queue>.{parquet,csv}` (+ TREC `<queue>.qrels.tsv` for qrels)."""
    items = items_by_id(review_dir, queue)
    rows = []
    for decision in DecisionStore(review_dir, queue).latest(items).values():
        item = items[decision.item_id]
        asked = {f.name for f in item.fields}
        # Only what the queue asks now: rebuilding a queue without a field retracts its answers.
        for criterion, value in decision.values.items():
            if criterion not in asked:
                continue
            rows.append(
                {
                    "image_id": item.image_id,
                    "entity_id": item.entity_id,
                    "criterion": criterion,
                    "value": value,
                    "source": "human",
                    "annotator": decision.annotator,
                    "item_id": item.item_id,
                    "queue": queue,
                    "mode": item.mode,
                    "model_version": item.model_version,
                    "decided_at": decision.decided_at,
                    **item.export_keys,
                }
            )
    frame = (
        pl.DataFrame(rows, schema_overrides=LABELS_SCHEMA, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=LABELS_SCHEMA)
    ).sort("item_id", "criterion")
    out = review_dir / "export"
    atomic_write(out / f"{queue}.parquet", frame.write_parquet)
    atomic_write(out / f"{queue}.csv", frame.write_csv)
    if "qid" in frame.columns:
        atomic_write_text(out / f"{queue}.qrels.tsv", trec_qrels(frame))
    return frame


def trec_qrels(frame: pl.DataFrame) -> str:
    """TREC qrels lines `qid 0 image_id rel` (spec §6) from a qrels-mode export."""
    rels = frame.filter(pl.col("criterion") == "rel").sort("qid", "image_id")
    rows = rels.select("qid", "image_id", "value").rows()
    return "".join(f"{qid}\t0\t{image_id}\t{rel}\n" for qid, image_id, rel in rows)


def export_all(review_dir: Path) -> dict[str, pl.DataFrame]:
    """Export every queue, plus all their labels columns in `export/labels_human.parquet` – the
    human `labels` rows that downstream steps union with `$DATA_DIR/ingest/labels.parquet`."""
    frames = {q: export_queue(review_dir, q) for q in list_queues(review_dir)}
    parts = [f.select(LABELS_SCHEMA.names()) for f in frames.values()]
    labels = pl.concat(parts) if parts else pl.DataFrame(schema=LABELS_SCHEMA)
    atomic_write(review_dir / "export" / f"{ALL_LABELS}.parquet", labels.write_parquet)
    return frames


@dataclass(frozen=True)
class QueueStats:
    name: str
    total: int
    decided: int
    items_per_hour: float


def queue_stats(review_dir: Path, queue: str) -> QueueStats:
    """Throughput from the time each item was on screen (capped), so breaks don't count;
    revisions count as work done."""
    items = items_by_id(review_dir, queue)
    decisions = DecisionStore(review_dir, queue).all(items)
    seconds = sum(min(d.elapsed_ms, MAX_ITEM_MS) for d in decisions) / 1000
    return QueueStats(
        name=queue,
        total=len(items),
        decided=len({d.item_id for d in decisions}),
        items_per_hour=len(decisions) * 3600 / seconds if seconds else 0.0,
    )
