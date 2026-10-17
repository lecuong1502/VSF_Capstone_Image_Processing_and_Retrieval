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
    # Pre-filled answer the human confirms with Enter. Leave it None for blind audits (kappa).
    default: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        keys = [o.key for o in self.options]
        if not self.options or len(set(keys)) != len(keys):
            raise ValueError(f"field {self.name}: options need distinct keys")
        if RESERVED_KEYS & set(keys):
            raise ValueError(f"field {self.name}: keys {sorted(RESERVED_KEYS)} are reserved")
        if self.default is not None and self.default not in {o.value for o in self.options}:
            raise ValueError(f"field {self.name}: default {self.default!r} is not an option")
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


def yes_no(name: str, label: str = "", default: str | None = None) -> Field:
    return Field(
        name=name,
        label=label or name,
        default=default,
        options=[
            Option(value="yes", key="y", label="có"),
            Option(value="no", key="n", label="không"),
            Option(value="unsure", key="u", label="không chắc"),
        ],
    )


# Spec §4.1 Gemini schema: one blind field per criterion.
MODERATION_FIELDS = [
    yes_no("relevance", "Đúng thực thể?"),
    yes_no("watermark", "Watermark / logo đè ảnh?"),
    yes_no("text_overlay", "Chữ quảng cáo / khung?"),
    yes_no("nsfw", "NSFW?"),
    yes_no("political", "Nhạy cảm chính trị?"),
    yes_no("violence", "Bạo lực?"),
    Field(
        name="shot_type",
        label="Loại ảnh",
        options=[
            Option(value="PHONG_CANH", key="1", label="phong cảnh"),
            Option(value="CHECKIN", key="2", label="check-in"),
            Option(value="FLYCAM", key="3", label="flycam"),
            Option(value="KHAC", key="4", label="khác"),
        ],
    ),
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
