"""Append-only decisions per queue (the latest per item wins), export and throughput stats."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ValidationError

from pipeline.files import atomic_write
from pipeline.review.items import Item, queue_path, read_queue

# Time on one item counts at most this much toward throughput (the tab may sit idle).
MAX_ITEM_MS = 120_000

LABEL_COLUMNS = ["item_id", "image_id", "entity_id", "criterion", "value", "source", "annotator"]


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
        line = decision.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell():  # start on a fresh line if a crash left a torn one
                    handle.seek(handle.tell() - 1)
                    if handle.read(1) != "\n":
                        line = "\n" + line
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

    def all(self) -> list[Decision]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(Decision.model_validate_json(line))
            except ValidationError:
                continue  # torn line from a crash
        return out

    def latest(self) -> dict[str, Decision]:
        return {d.item_id: d for d in self.all()}


def export_queue(review_dir: Path, queue: str) -> pl.DataFrame:
    """Latest decisions as labels rows (one per field, source='human') plus the items' `ref` keys;
    writes `export/<queue>.{parquet,csv}`."""
    items = {i.item_id: i for i in read_queue(review_dir, queue)}
    rows = []
    for decision in DecisionStore(review_dir, queue).latest().values():
        item = items.get(decision.item_id)
        if item is None:  # the queue was rebuilt without this item
            continue
        for criterion, value in decision.values.items():
            rows.append(
                {
                    "item_id": item.item_id,
                    "image_id": item.image_id,
                    "entity_id": item.entity_id,
                    "criterion": criterion,
                    "value": value,
                    "source": "human",
                    "annotator": decision.annotator,
                    "mode": item.mode,
                    "decided_at": decision.decided_at,
                    **item.ref,
                }
            )
    schema = pl.Schema(
        {
            "item_id": pl.String(),
            "image_id": pl.Int64(),
            "entity_id": pl.String(),
            "criterion": pl.String(),
            "value": pl.String(),
            "source": pl.String(),
            "annotator": pl.String(),
            "mode": pl.String(),
            "decided_at": pl.Datetime("us", "UTC"),
        }
    )
    frame = (
        pl.DataFrame(rows, schema_overrides=schema, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=schema)
    )
    frame = frame.sort("item_id", "criterion")
    out = review_dir / "export"
    atomic_write(out / f"{queue}.parquet", frame.write_parquet)
    atomic_write(out / f"{queue}.csv", frame.write_csv)
    return frame


@dataclass(frozen=True)
class QueueStats:
    name: str
    total: int
    decided: int
    items_per_hour: float


def queue_stats(review_dir: Path, queue: str) -> QueueStats:
    """Throughput from the time each item was on screen (capped), so breaks don't count."""
    total = len(read_queue(review_dir, queue))
    decisions = DecisionStore(review_dir, queue).all()
    decided = len({d.item_id for d in decisions})
    seconds = sum(min(d.elapsed_ms, MAX_ITEM_MS) for d in decisions) / 1000
    per_hour = len(decisions) * 3600 / seconds if seconds else 0.0
    return QueueStats(name=queue, total=total, decided=decided, items_per_hour=per_hour)
