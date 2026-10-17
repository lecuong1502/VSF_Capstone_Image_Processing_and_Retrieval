"""Embed every downloaded image with EMBED_MODEL (spec §4.1, §8) – one vector per image, reused by
the moderation heads and retrieval.

Output: `$DATA_DIR/embeddings/<org>--<model>/part-*.parquet` with image_id, sha256, embedding
(L2-normalized float32 array) and embedded_at. One part is written per shard, so an interrupted
run resumes where it stopped; images whose sha256 changed are embedded again and the newest
row wins in `load_embeddings`.

Usage: python -m pipeline.embed run [--batch-size 8] [--workers N]
       python -m pipeline.embed bench [--images 64] [--batch-sizes 1 4 8 16]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from pipeline import config
from pipeline.download import IMAGE_ERRORS, assets_path
from pipeline.embedder import (
    DEFAULT_BATCH_SIZE,
    Embedder,
    Preprocessor,
    model_dir_name,
    resolve_model,
)
from pipeline.files import atomic_write, atomic_write_text
from pipeline.shards import iter_shard_images, split_shard_path

log = logging.getLogger(__name__)

CHUNK = 64  # images preprocessed per worker task


def embedding_path(data_dir: Path, model: str) -> Path:
    return data_dir / "embeddings" / model_dir_name(model)


def load_embeddings(path: Path) -> pl.DataFrame:
    """All parts, newest row per image_id."""
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        return pl.DataFrame(
            schema={
                "image_id": pl.Int64,
                "sha256": pl.String,
                "embedded_at": pl.Datetime("us", "UTC"),
            }
        )
    return (
        pl.concat([pl.read_parquet(p) for p in parts])
        .sort("embedded_at")
        .unique("image_id", keep="last", maintain_order=True)
        .sort("image_id")
    )


@dataclass
class Summary:
    embedded: int = 0
    failed: int = 0
    seconds: float = 0.0

    @property
    def images_per_s(self) -> float:
        return self.embedded / self.seconds if self.seconds else 0.0


def _preprocess_chunk(
    job: tuple[Path, list[tuple[str, int]], Preprocessor],
) -> list[tuple[int, np.ndarray | None]]:
    tar_path, items, preprocessor = job
    wanted = dict(items)
    out: list[tuple[int, np.ndarray | None]] = []
    seen = set()
    for image_id, image_bytes in iter_shard_images(tar_path, wanted):
        seen.add(image_id)
        try:
            out.append((image_id, preprocessor(image_bytes)))
        except IMAGE_ERRORS as exc:
            log.warning("Cannot preprocess image %d: %s", image_id, exc)
            out.append((image_id, None))
    out.extend((i, None) for i in wanted.values() if i not in seen)
    return out


def run_embedding(
    data_dir: Path,
    embedder: Embedder,
    model: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int | None = None,
) -> Summary:
    """Embed every `ok` image not yet embedded with its current sha256."""
    out_dir = embedding_path(data_dir, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = pl.read_parquet(assets_path(data_dir)).filter(pl.col("download_status") == "ok")
    done = load_embeddings(out_dir).select("image_id", "sha256")
    todo = ok.join(done, on=["image_id", "sha256"], how="anti")
    summary = Summary()
    if todo.is_empty():
        return summary

    by_tar: dict[Path, list[tuple[str, int]]] = defaultdict(list)
    sha_of = dict(todo.select("image_id", "sha256").rows())
    for image_id, shard in todo.select("image_id", "shard_path").sort("shard_path").iter_rows():
        tar, key = split_shard_path(shard)
        by_tar[data_dir / tar].append((key, image_id))
    log.info("Embedding %d images from %d shards with %s", todo.height, len(by_tar), model)

    context = multiprocessing.get_context("spawn")
    with context.Pool(workers or max(1, (os.cpu_count() or 2) // 2)) as pool:
        for tar, items in by_tar.items():
            jobs = [
                (tar, items[i : i + CHUNK], embedder.preprocessor)
                for i in range(0, len(items), CHUNK)
            ]
            ids: list[int] = []
            vectors: list[np.ndarray] = []
            started = time.monotonic()
            for batch_ids, batch_vectors, failed in _batches(
                pool.imap(_preprocess_chunk, jobs), batch_size, embedder
            ):
                ids.extend(batch_ids)
                vectors.extend(batch_vectors)
                summary.failed += failed
            summary.seconds += time.monotonic() - started
            if ids:
                _write_part(out_dir, ids, [sha_of[i] for i in ids], np.stack(vectors))
                summary.embedded += len(ids)
            log.info(
                "%s: %d/%d images embedded (%.1f img/s)",
                tar.name,
                summary.embedded,
                todo.height,
                summary.images_per_s,
            )
    meta = {"model": model, "dim": embedder.dim, "updated_at": datetime.now(UTC).isoformat()}
    atomic_write_text(out_dir / "meta.json", json.dumps(meta, indent=2))
    return summary


def _batches(
    chunks: Iterator[list[tuple[int, np.ndarray | None]]], batch_size: int, embedder: Embedder
) -> Iterator[tuple[list[int], list[np.ndarray], int]]:
    """Group preprocessed images into GPU batches; yields (ids, vectors, failed_count)."""
    pending_ids: list[int] = []
    pending: list[np.ndarray] = []
    failed = 0
    for chunk in chunks:
        for image_id, pixels in chunk:
            if pixels is None:
                failed += 1
                continue
            pending_ids.append(image_id)
            pending.append(pixels)
            if len(pending) == batch_size:
                yield pending_ids, list(embedder.run_pixels(np.stack(pending))), failed
                pending_ids, pending, failed = [], [], 0
    if pending or failed:
        vectors = list(embedder.run_pixels(np.stack(pending))) if pending else []
        yield pending_ids, vectors, failed


def _write_part(out_dir: Path, ids: list[int], shas: list[str], vectors: np.ndarray) -> None:
    now = datetime.now(UTC)
    frame = pl.DataFrame(
        {
            "image_id": ids,
            "sha256": shas,
            "embedding": vectors.astype(np.float32),
            "embedded_at": [now] * len(ids),
        },
        schema={
            "image_id": pl.Int64,
            "sha256": pl.String,
            "embedding": pl.Array(pl.Float32, vectors.shape[1]),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    name = f"part-{now.strftime('%Y%m%dT%H%M%S%f')}.parquet"
    atomic_write(out_dir / name, frame.write_parquet)


def bench(
    embedder: Embedder,
    images: Sequence[bytes],
    batch_sizes: Sequence[int],
    repeats: int = 3,
) -> list[dict[str, float | int | None]]:
    """GPU images/s per batch size on preprocessed inputs, plus CPU preprocessing images/s."""
    started = time.monotonic()
    pixels = np.stack([embedder.preprocessor(b) for b in images])
    preprocess_per_s = len(images) / (time.monotonic() - started)
    rows: list[dict[str, float | int | None]] = []
    for size in batch_sizes:
        batch = np.resize(pixels, (size, *pixels.shape[1:]))
        try:
            embedder.forward_images(batch)  # warm-up
            started = time.monotonic()
            for _ in range(repeats):
                embedder.forward_images(batch)
            per_s: float | None = size * repeats / (time.monotonic() - started)
        except Exception as exc:
            if "Failed to allocate" not in str(exc):
                raise
            per_s = None  # does not fit in GPU memory
        rows.append(
            {
                "batch_size": size,
                "images_per_s": per_s,
                "preprocess_images_per_s_one_core": preprocess_per_s,
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.embed")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_p.add_argument("--workers", type=int)
    bench_p = sub.add_parser("bench")
    bench_p.add_argument("--images", type=int, default=64)
    bench_p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    model = config.embed_model()
    data_dir = config.data_dir()
    batch_size = getattr(args, "batch_size", DEFAULT_BATCH_SIZE)
    embedder = Embedder.load(resolve_model(model, config.models_dir()), batch_size=batch_size)
    if args.command == "run":
        summary = run_embedding(
            data_dir, embedder, model, batch_size=args.batch_size, workers=args.workers
        )
        log.info(
            "Embedded %d images (%d failed) in %.0fs, %.1f img/s",
            summary.embedded,
            summary.failed,
            summary.seconds,
            summary.images_per_s,
        )
    else:
        ok = pl.read_parquet(assets_path(data_dir)).filter(pl.col("download_status") == "ok")
        sample = ok.sort("image_id").head(args.images)
        images: list[bytes] = []
        by_tar: dict[Path, dict[str, int]] = defaultdict(dict)
        for image_id, shard in sample.select("image_id", "shard_path").iter_rows():
            tar, key = split_shard_path(shard)
            by_tar[data_dir / tar][key] = image_id
        for tar, wanted in by_tar.items():
            images.extend(b for _, b in iter_shard_images(tar, wanted))
        rows = bench(embedder, images, args.batch_sizes)
        out = embedding_path(data_dir, model) / "bench.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            out, json.dumps({"model": model, "dim": embedder.dim, "rows": rows}, indent=2)
        )
        with pl.Config(tbl_rows=-1, float_precision=1):
            log.info("%s (dim %d)\n%s", model, embedder.dim, pl.DataFrame(rows))


if __name__ == "__main__":
    main()
