"""Reading images back out of the WebDataset shards written by `pipeline.download`."""

from __future__ import annotations

import tarfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import polars as pl

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})


def split_shard_path(shard_path: str) -> tuple[Path, str]:
    """`download/runs/<run>/00000.tar#000000007` → (tar path relative to DATA_DIR, sample key)."""
    tar, _, key = shard_path.partition("#")
    return Path(tar), key


def iter_shard_images(tar_path: Path, wanted: dict[str, int]) -> Iterator[tuple[int, bytes]]:
    """(image_id, encoded image bytes) for every sample key in `wanted` found in the shard."""
    with tarfile.open(tar_path) as tar:
        for member in tar:
            key, _, ext = member.name.partition(".")
            if key not in wanted or ext not in IMAGE_EXTENSIONS:
                continue
            data = tar.extractfile(member)
            if data is not None:
                yield wanted[key], data.read()


def group_by_shard(frame: pl.DataFrame, data_dir: Path) -> dict[Path, dict[str, int]]:
    """{tar path: {sample key: image_id}} for a frame with image_id and shard_path columns."""
    groups: dict[Path, dict[str, int]] = defaultdict(dict)
    for image_id, shard in frame.select("image_id", "shard_path").sort("shard_path").iter_rows():
        tar, key = split_shard_path(shard)
        groups[data_dir / tar][key] = image_id
    return dict(groups)
