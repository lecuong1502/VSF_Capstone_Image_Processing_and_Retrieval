"""Embed every downloaded image with EMBED_MODEL (spec §4.1, §8) – one vector per image, reused by
the moderation heads and retrieval.

Output in `$DATA_DIR/embeddings/<org>--<model>/`:
- `part-*.parquet`: image_id, sha256, embedding (L2-normalized float32 array), embedded_at. One
  part per shard, so an interrupted run resumes where it stopped; images whose sha256 changed are
  embedded again and the newest row wins in `load_embeddings`.
- `failed.parquet`: images that can't be decoded (not retried until their sha256 changes).
Vectors are stored at the model's full dimension; consumers that want a Matryoshka prefix (spec
§5.1: jina-clip-v2 at 512-d) truncate and re-normalize. Holds the data lock like download/dedup.

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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from pipeline import config
from pipeline.download import IMAGE_ERRORS, assets_path, lock_path
from pipeline.embedder import (
    DEFAULT_BATCH_SIZE,
    Embedder,
    Preprocessor,
    is_out_of_memory,
    model_dir_name,
    resolve_model,
)
from pipeline.files import atomic_write, atomic_write_text, exclusive_lock
from pipeline.shards import group_by_shard, iter_shard_images

log = logging.getLogger(__name__)

CHUNK = 64  # images prepared per worker task; at most one shard is in flight at a time
FAILED_SCHEMA = {"image_id": pl.Int64, "sha256": pl.String, "error": pl.String}

# (image_id, uint8 HWC image or None, error message or None)
Prepared = tuple[int, np.ndarray | None, str | None]


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
                "embedding": pl.List(pl.Float32),
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


def _prepare_chunk(job: tuple[Path, list[tuple[str, int]], Preprocessor]) -> list[Prepared]:
    tar_path, items, preprocessor = job
    wanted = dict(items)
    out: list[Prepared] = []
    seen = set()
    for image_id, image_bytes in iter_shard_images(tar_path, wanted):
        seen.add(image_id)
        try:
            out.append((image_id, preprocessor.prepare(image_bytes), None))
        except IMAGE_ERRORS as exc:
            out.append((image_id, None, f"{type(exc).__name__}: {exc}"))
    out.extend((i, None, "missing from shard") for i in wanted.values() if i not in seen)
    return out


def run_embedding(
    data_dir: Path, embedder: Embedder, model: str, *, workers: int | None = None
) -> Summary:
    """Embed every `ok` image not yet embedded (or failed) with its current sha256."""
    with exclusive_lock(lock_path(data_dir)):
        return _run_embedding(data_dir, embedder, model, workers)


def _run_embedding(data_dir: Path, embedder: Embedder, model: str, workers: int | None) -> Summary:
    out_dir = embedding_path(data_dir, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "failed.parquet"
    failed = (
        pl.read_parquet(failed_path) if failed_path.exists() else pl.DataFrame(schema=FAILED_SCHEMA)
    )
    ok = pl.read_parquet(assets_path(data_dir)).filter(pl.col("download_status") == "ok")
    done = pl.concat(
        [load_embeddings(out_dir).select("image_id", "sha256"), failed.select("image_id", "sha256")]
    )
    todo = ok.join(done, on=["image_id", "sha256"], how="anti")
    summary = Summary()
    if todo.is_empty():
        return summary

    sha_of = dict(todo.select("image_id", "sha256").rows())
    shards = group_by_shard(todo, data_dir)
    log.info("Embedding %d images from %d shards with %s", todo.height, len(shards), model)
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers or max(1, (os.cpu_count() or 2) // 2)) as pool:
        for tar, wanted in shards.items():
            items = sorted(wanted.items())
            jobs = [
                (tar, items[i : i + CHUNK], embedder.preprocessor)
                for i in range(0, len(items), CHUNK)
            ]
            started = time.monotonic()
            ids, vectors, errors = _embed_shard(pool.imap(_prepare_chunk, jobs), embedder)
            summary.seconds += time.monotonic() - started
            if ids:
                _write_part(out_dir, ids, [sha_of[i] for i in ids], vectors)
                summary.embedded += len(ids)
            if errors:
                rows = pl.DataFrame(
                    [(i, sha_of[i], e) for i, e in errors], schema=FAILED_SCHEMA, orient="row"
                )
                failed = pl.concat(
                    [failed.filter(~pl.col("image_id").is_in(rows["image_id"].implode())), rows]
                )
                atomic_write(failed_path, failed.write_parquet)
                summary.failed += len(errors)
            log.info(
                "%s: %d/%d embedded, %d failed (%.1f img/s)",
                tar.name,
                summary.embedded,
                todo.height,
                summary.failed,
                summary.images_per_s,
            )
    meta = {"model": model, "dim": embedder.dim, "updated_at": datetime.now(UTC).isoformat()}
    atomic_write_text(out_dir / "meta.json", json.dumps(meta, indent=2))
    return summary


def _embed_shard(
    chunks: Iterator[list[Prepared]], embedder: Embedder
) -> tuple[list[int], np.ndarray, list[tuple[int, str]]]:
    """Normalize and embed one shard's prepared images in batches of embedder.batch_size."""
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    errors: list[tuple[int, str]] = []
    pending_ids: list[int] = []
    pending: list[np.ndarray] = []

    def flush() -> None:
        if pending:
            batch = embedder.preprocessor.normalize(np.stack(pending))
            vectors.append(embedder.run_pixels(batch))
            ids.extend(pending_ids)
            pending_ids.clear()
            pending.clear()

    for chunk in chunks:
        for image_id, image, error in chunk:
            if image is None:
                errors.append((image_id, error or "unknown error"))
                continue
            pending_ids.append(image_id)
            pending.append(image)
            if len(pending) == embedder.batch_size:
                flush()
    flush()
    stacked = np.concatenate(vectors) if vectors else np.zeros((0, embedder.dim), np.float32)
    return ids, stacked, errors


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
            if not is_out_of_memory(exc):
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
        summary = run_embedding(data_dir, embedder, model, workers=args.workers)
        log.info(
            "Embedded %d images (%d failed) in %.0fs, %.1f img/s",
            summary.embedded,
            summary.failed,
            summary.seconds,
            summary.images_per_s,
        )
        return
    ok = pl.read_parquet(assets_path(data_dir)).filter(pl.col("download_status") == "ok")
    images = [
        image
        for tar, wanted in group_by_shard(ok.sort("image_id").head(args.images), data_dir).items()
        for _, image in iter_shard_images(tar, wanted)
    ]
    rows = bench(embedder, images, args.batch_sizes)
    out = embedding_path(data_dir, model) / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "dim": embedder.dim, "rows": rows}
    atomic_write_text(out, json.dumps(payload, indent=2))
    with pl.Config(tbl_rows=-1, float_precision=1):
        log.info("%s (dim %d)\n%s", model, embedder.dim, pl.DataFrame(rows))


if __name__ == "__main__":
    main()
