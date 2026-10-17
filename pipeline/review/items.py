"""Review items: what the human sees (image, title, context) and the fields they answer by key."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from pipeline.files import atomic_write_text

Mode = Literal["relevance", "moderation", "kg", "qrels"]

# Keys the UI uses for navigation – an option can't take them.
RESERVED_KEYS = frozenset({"Enter", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"})
_QUEUE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Option(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    key: str
    label: str = ""


class Field(BaseModel):
    """One question about the item; `name` becomes the `criterion` of the exported label."""

    model_config = ConfigDict(frozen=True)

    name: str
    options: list[Option]
    label: str = ""
    # Pre-filled answer the human confirms with Enter. For kappa audits it must not come from the
    # rater being audited (a fixed prior like "no" is fine, Gemini's answer is not).
    default: str | None = None
    # A key that flips the field between its first two options whichever field is active, so a
    # mostly-"no" flag costs no keystroke and a "yes" costs one.
    toggle_key: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        keys = [o.key for o in self.options]
        if not self.options or len(set(keys)) != len(keys):
            raise ValueError(f"field {self.name}: options need distinct keys")
        if RESERVED_KEYS & set(keys):
            raise ValueError(f"field {self.name}: keys {sorted(RESERVED_KEYS)} are reserved")
        if self.default is not None and self.default not in {o.value for o in self.options}:
            raise ValueError(f"field {self.name}: default {self.default!r} is not an option")
        if self.toggle_key is not None and (
            len(self.options) < 2 or self.toggle_key in RESERVED_KEYS | set(keys)
        ):
            raise ValueError(f"field {self.name}: toggle key {self.toggle_key!r} is not usable")
        return self


class Item(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    mode: Mode
    title: str
    fields: list[Field]
    image_id: int | None = None
    entity_id: str | None = None
    context: dict[str, str] = {}
    references: list[int] = []  # image_ids shown small next to the image, e.g. accepted examples
    scores: dict[str, float] = {}  # model outputs shown to the reviewer (e.g. p_watermark)
    model_version: str | None = None  # of the model that put the item in the queue (spec §4.1 T5)
    export_keys: dict[str, str] = {}  # columns copied into the export (e.g. src/rel/dst, qid)

    @model_validator(mode="after")
    def _check_toggles(self) -> Self:
        toggles = [f.toggle_key for f in self.fields if f.toggle_key is not None]
        option_keys = {o.key for f in self.fields for o in f.options}
        if len(set(toggles)) != len(toggles) or option_keys & set(toggles):
            raise ValueError(f"item {self.item_id}: toggle keys must be unique and unused")
        return self


def yes_no(
    name: str, label: str = "", default: str | None = None, toggle_key: str | None = None
) -> Field:
    return Field(
        name=name,
        label=label or name,
        default=default,
        toggle_key=toggle_key,
        options=[
            Option(value="yes", key="y", label="có"),
            Option(value="no", key="n", label="không"),
            Option(value="unsure", key="u", label="không chắc"),
        ],
    )


def _flag(name: str, label: str, toggle_key: str) -> Field:
    """A rare-positive criterion: pre-filled "no" (a fixed prior, not any rater's answer)."""
    return yes_no(name, label, default="no", toggle_key=toggle_key)


# Spec §4.1 Gemini schema minus shot_type (its head learns from the `query_type` column, spec
# §4.1 table). One key per typical image: flags default to "no" and flip with their toggle key;
# relevance has no default and comes last, so answering it submits the item.
MODERATION_FIELDS = [
    _flag("watermark", "Watermark / logo đè ảnh?", "w"),
    _flag("text_overlay", "Chữ quảng cáo / khung?", "t"),
    _flag("nsfw", "NSFW?", "x"),
    _flag("political", "Nhạy cảm chính trị?", "p"),
    _flag("violence", "Bạo lực?", "v"),
    yes_no("relevance", "Đúng thực thể?"),
]


def review_dir(data_dir: Path) -> Path:
    return data_dir / "review"


def queue_path(review_dir: Path, name: str, kind: str = "queues") -> Path:
    if not _QUEUE_NAME.match(name):
        raise ValueError(f"invalid queue name {name!r}")
    return review_dir / kind / f"{name}.jsonl"


def write_queue(review_dir: Path, name: str, items: list[Item]) -> Path:
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"queue {name}: duplicate item_id")
    path = queue_path(review_dir, name)
    atomic_write_text(path, "".join(i.model_dump_json() + "\n" for i in items))
    return path


def read_queue(review_dir: Path, name: str) -> list[Item]:
    lines = queue_path(review_dir, name).read_text().splitlines()
    return [Item.model_validate_json(line) for line in lines if line.strip()]


def list_queues(review_dir: Path) -> list[str]:
    return sorted(p.stem for p in (review_dir / "queues").glob("*.jsonl"))
