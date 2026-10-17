"""Download every image with img2dataset into WebDataset shards, make WebP thumbnails, and keep
`image_assets` (one row per image) up to date. Re-running only retries transient failures.

Statuses:
- ok           downloaded; stored resized (short side ≤ 1024 px) with sha256 and real size
- failed       transient error (timeout, 5xx, connection reset) – retried with backoff, at most
               `max_attempts` times across runs
- dead         404/410/403/401/400 or unresolvable host – not retried
- ssl_error    the server's TLS certificate fails verification – not retried
- blocked      Facebook lookaside URLs answer with an HTML page instead of the image – not retried
- not_image    any other URL that returns something that isn't an image – not retried
- opted_out    the site forbids AI use via `X-Robots-Tag` (noai/noimageai, spec §4.1) – not retried
- invalid_url  not http(s) (e.g. `x-raw-image:///…` left by Google) – never fetched

Work is checkpointed per chunk of `chunk_size` images (one img2dataset call each): a crash loses
at most the chunk in flight, and run directories no longer referenced by `image_assets` (left by
a crash) are deleted at the start of the next run.

Spec §4.1 T1 deviations: statuses are grouped by what to do next (retry or not) rather than
4xx/5xx/timeout; the raw error message is kept in `error`.

Usage: python -m pipeline.download [--rounds 3] [--backoff 60] [--max-attempts 3]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from PIL import Image

from pipeline import config
from pipeline.files import atomic_write, exclusive_lock
from pipeline.shards import iter_shard_images

log = logging.getLogger(__name__)

RETRYABLE = frozenset({"failed"})
# Errors Pillow raises for undecodable or oversized images.
IMAGE_ERRORS = (OSError, ValueError, Image.DecompressionBombError)
# Permanent failures: the image will never be available (the dead-link rate counts these).
UNREACHABLE = frozenset({"dead", "ssl_error", "blocked", "not_image", "opted_out", "invalid_url"})
STORED_SIZE = 1024  # short side of the stored image used for embeddings (spec §4.1)
THUMB_SIZE = 384  # long side of the UI thumbnail (spec §6)
CHUNK_SIZE = 5000  # images per img2dataset call = checkpoint granularity
_OPTED_OUT = "X-Robots-Tag"
_DEAD = re.compile(
    r"HTTP Error (400|401|403|404|410)|Name or service not known|No address associated|"
    r"nodename nor servname"
)
_BLOCKED_HOSTS = re.compile(r"^https?://([^/]*\.)?(fbsbx\.com|fbcdn\.net|facebook\.com)/")
_VALID_URL = r"^https?://"

ASSET_SCHEMA = pl.Schema(
    {
        "image_id": pl.Int64(),
        "image_url": pl.String(),
        "download_status": pl.String(),
        "error": pl.String(),
        "attempts": pl.Int32(),
        "real_w": pl.Int32(),
        "real_h": pl.Int32(),
        "stored_w": pl.Int32(),
        "stored_h": pl.Int32(),
        "sha256": pl.String(),
        "pdq_hash": pl.String(),  # filled by hashing (#10)
        "shard_path": pl.String(),
        "thumb_path": pl.String(),
        "last_attempt_at": pl.Datetime("us", "UTC"),
    }
)

# (batch with image_id + url, output dir) -> one row per input with img2dataset's result columns
# plus `shard` (the .tar stem the image was written to).
Fetcher = Callable[[pl.DataFrame, Path], pl.DataFrame]


def classify(status: str, error: str | None, url: str) -> str:
    if status == "success":
        return "ok"
    message = error or ""
    if status == "failed_to_resize" or "decoding error" in message.lower():
        return "blocked" if _BLOCKED_HOSTS.match(url) else "not_image"
    if _OPTED_OUT in message:
        return "opted_out"
    if "CERTIFICATE_VERIFY_FAILED" in message:
        return "ssl_error"
    if _DEAD.search(message):
        return "dead"
    return "failed"


def assets_path(data_dir: Path) -> Path:
    return data_dir / "download" / "image_assets.parquet"


def load_assets(data_dir: Path) -> pl.DataFrame:
    path = assets_path(data_dir)
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=ASSET_SCHEMA)


def plan(images: pl.DataFrame, assets: pl.DataFrame, max_attempts: int) -> pl.DataFrame:
    """Valid-URL images never attempted, or retryable and under `max_attempts`."""
    known = assets.select("image_id", "download_status", "attempts")
    return (
        images.filter(pl.col("image_url").str.contains(_VALID_URL))
        .join(known, on="image_id", how="left")
        .filter(
            pl.col("download_status").is_null()
            | (
                pl.col("download_status").is_in(list(RETRYABLE))
                & (pl.col("attempts") < max_attempts)
            )
        )
        .select("image_id", "image_url")
        .sort("image_id")
    )


@dataclass
class Summary:
    attempted: int = 0
    ok: int = 0
    seconds: float = 0.0
    statuses: dict[str, int] = field(default_factory=dict)

    @property
    def images_per_s(self) -> float:
        return self.attempted / self.seconds if self.seconds else 0.0


def run_download(
    images: pl.DataFrame,
    data_dir: Path,
    fetch: Fetcher,
    *,
    rounds: int = 3,
    backoff_s: float = 60,
    max_attempts: int = 3,
    chunk_size: int = CHUNK_SIZE,
    sleep: Callable[[float], None] = time.sleep,
) -> Summary:
    """Download every planned image; holds the data lock so `make dedup` can't run meanwhile."""
    with exclusive_lock(lock_path(data_dir)):
        return _run_download(
            images, data_dir, fetch, rounds, backoff_s, max_attempts, chunk_size, sleep
        )


def lock_path(data_dir: Path) -> Path:
    """Held by steps that rewrite image_assets.parquet (download, dedup)."""
    return data_dir / "download" / ".lock"


def _run_download(
    images: pl.DataFrame,
    data_dir: Path,
    fetch: Fetcher,
    rounds: int,
    backoff_s: float,
    max_attempts: int,
    chunk_size: int,
    sleep: Callable[[float], None],
) -> Summary:
    assets = _mark_invalid(images, load_assets(data_dir))
    _save(assets, data_dir)
    _remove_orphaned_runs(data_dir, assets)
    summary = Summary()
    attempted: set[int] = set()
    for round_no in range(rounds):
        todo = plan(images, assets, max_attempts)
        if todo.is_empty():
            break
        if round_no:
            wait = backoff_s * 2 ** (round_no - 1)
            log.info(
                "Round %d: %d retryable images, waiting %.0fs", round_no + 1, todo.height, wait
            )
            sleep(wait)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"-r{round_no + 1}"
        for chunk_no, chunk in enumerate(todo.iter_slices(chunk_size)):
            run_dir = data_dir / "download" / "runs" / f"{run_id}-c{chunk_no:03d}"
            assets, results, seconds = _fetch_chunk(chunk, run_dir, data_dir, assets, fetch)
            attempted.update(results["image_id"].to_list())
            summary.seconds += seconds

    final = assets.filter(pl.col("image_id").is_in(images["image_id"].implode()))
    summary.attempted = len(attempted)
    summary.statuses = dict(final.group_by("download_status").len().rows())
    summary.ok = summary.statuses.get("ok", 0)
    return summary


def _fetch_chunk(
    chunk: pl.DataFrame, run_dir: Path, data_dir: Path, assets: pl.DataFrame, fetch: Fetcher
) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """Fetch one chunk, then immediately persist thumbnails, assets and the run log."""
    started = time.monotonic()
    results = fetch(chunk.rename({"image_url": "url"}), run_dir)
    seconds = time.monotonic() - started
    results = results.with_columns(
        pl.struct("status", "error_message", "url")
        .map_elements(
            lambda r: classify(r["status"], r["error_message"], r["url"]), return_dtype=pl.String
        )
        .alias("download_status")
    )
    thumbs = _make_thumbnails(results, run_dir, data_dir / "thumbs")
    assets = _update(assets, results, thumbs, run_dir.relative_to(data_dir))
    _save(assets, data_dir)
    counts = dict(results.group_by("download_status").len().rows())
    _log_run(data_dir, run_dir.name, results.height, counts, seconds)
    log.info(
        "%s: %d images in %.0fs (%.1f img/s) %s",
        run_dir.name,
        results.height,
        seconds,
        results.height / seconds if seconds else 0,
        counts,
    )
    return assets, results, seconds


def _remove_orphaned_runs(data_dir: Path, assets: pl.DataFrame) -> None:
    """Delete run dirs (and their input lists) that no `image_assets` row points to."""
    runs = data_dir / "download" / "runs"
    if not runs.exists():
        return
    referenced = {
        p.split("/")[2] for p in assets["shard_path"].drop_nulls().to_list() if p.count("/") >= 3
    }
    for path in runs.iterdir():
        name = path.name.removesuffix(".input.parquet")
        if name in referenced:
            continue
        log.info("Removing orphaned run output %s", path.name)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _mark_invalid(images: pl.DataFrame, assets: pl.DataFrame) -> pl.DataFrame:
    invalid = images.filter(
        ~pl.col("image_url").str.contains(_VALID_URL)
        & ~pl.col("image_id").is_in(assets["image_id"].implode())
    )
    if invalid.is_empty():
        return assets
    rows = invalid.select(
        "image_id",
        "image_url",
        pl.lit("invalid_url").alias("download_status"),
        pl.lit("not an http(s) URL").alias("error"),
        pl.lit(0, pl.Int32).alias("attempts"),
    )
    return pl.concat([assets, rows], how="diagonal_relaxed").select(ASSET_SCHEMA.names())


def _update(
    assets: pl.DataFrame, results: pl.DataFrame, thumbs: dict[int, str], run_rel: Path
) -> pl.DataFrame:
    now = datetime.now(UTC)
    previous = assets.select("image_id", pl.col("attempts").alias("previous_attempts"))
    ok = pl.col("download_status") == "ok"
    fresh = (
        results.join(previous, on="image_id", how="left")
        .with_columns(
            (pl.col("previous_attempts").fill_null(0) + 1).cast(pl.Int32).alias("attempts"),
            pl.col("image_id")
            .replace_strict(thumbs, default=None, return_dtype=pl.String)
            .alias("thumb_path"),
        )
        .select(
            "image_id",
            pl.col("url").alias("image_url"),
            "download_status",
            pl.when(ok).then(None).otherwise(pl.col("error_message")).alias("error"),
            "attempts",
            pl.col("original_width").cast(pl.Int32).alias("real_w"),
            pl.col("original_height").cast(pl.Int32).alias("real_h"),
            pl.col("width").cast(pl.Int32).alias("stored_w"),
            pl.col("height").cast(pl.Int32).alias("stored_h"),
            "sha256",
            pl.lit(None, pl.String).alias("pdq_hash"),
            pl.when(ok)
            .then(pl.format(f"{run_rel.as_posix()}/{{}}.tar#{{}}", "shard", "key"))
            .alias("shard_path"),
            "thumb_path",
            pl.lit(now, pl.Datetime("us", "UTC")).alias("last_attempt_at"),
        )
        .cast(ASSET_SCHEMA)
    )
    kept = assets.filter(~pl.col("image_id").is_in(fresh["image_id"].implode()))
    return pl.concat([kept, fresh]).sort("image_id")


def _make_thumbnails(results: pl.DataFrame, run_dir: Path, thumbs_dir: Path) -> dict[int, str]:
    """WebP thumbnails (long side THUMB_SIZE) for the successful rows; returns id -> rel path."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    made: dict[int, str] = {}
    ok = results.filter(pl.col("download_status") == "ok")
    for shard, rows in ok.group_by("shard"):
        wanted = dict(zip(rows["key"], rows["image_id"], strict=True))
        for image_id, image_bytes in iter_shard_images(run_dir / f"{shard[0]}.tar", wanted):
            try:
                _write_thumbnail(image_bytes, thumbs_dir / f"{image_id}.webp")
            except IMAGE_ERRORS as exc:
                log.warning("No thumbnail for image %d: %s", image_id, exc)
                made.pop(image_id, None)
                continue
            made[image_id] = f"{thumbs_dir.name}/{image_id}.webp"
    return made


def _write_thumbnail(image_bytes: bytes, out: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.draft("RGB", (THUMB_SIZE, THUMB_SIZE))  # cheap JPEG downscale while decoding
        rgba = image.convert("RGBA")
    thumb = Image.new("RGB", rgba.size, (255, 255, 255))
    thumb.paste(rgba, mask=rgba.getchannel("A"))  # transparent areas become white, not black
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))

    def save(tmp: Path) -> None:
        thumb.save(tmp, "WEBP", quality=80)

    atomic_write(out, save)


def _save(assets: pl.DataFrame, data_dir: Path) -> None:
    atomic_write(assets_path(data_dir), assets.write_parquet)


def _log_run(
    data_dir: Path, run_id: str, attempted: int, counts: dict[str, int], seconds: float
) -> None:
    path = data_dir / "download" / "runs.parquet"
    row = pl.DataFrame(
        {
            "run_id": [run_id],
            "attempted": [attempted],
            "ok": [counts.get("ok", 0)],
            "seconds": [seconds],
            "images_per_s": [attempted / seconds if seconds else 0.0],
            "statuses": [json.dumps(counts, sort_keys=True)],
        }
    )
    runs = pl.concat([pl.read_parquet(path), row]) if path.exists() else row
    atomic_write(path, runs.write_parquet)


def img2dataset_fetch(batch: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
    """Real fetcher: img2dataset → WebDataset shards in `out_dir`, results with a `shard` column."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from img2dataset import download

    out_dir.mkdir(parents=True, exist_ok=True)
    processes = max(1, (os.cpu_count() or 2) // 2)
    # img2dataset parallelizes across shards: give every process at least one shard.
    per_shard = max(50, -(-batch.height // processes))
    url_list = out_dir.with_suffix(".input.parquet")
    batch.select("url", "image_id").write_parquet(url_list)
    download(
        url_list=str(url_list),
        input_format="parquet",
        url_col="url",
        save_additional_columns=["image_id"],
        output_folder=str(out_dir),
        output_format="webdataset",
        image_size=STORED_SIZE,
        resize_mode="keep_ratio",
        resize_only_if_bigger=True,
        compute_hash="sha256",
        retries=2,
        timeout=10,
        processes_count=processes,
        thread_count=32,
        number_sample_per_shard=per_shard,
        distributor="multiprocessing",
        enable_wandb=False,
        user_agent_token="vsf-capstone-research",
        # img2dataset's default: skip images whose X-Robots-Tag says noai/noimageai/noindex.
    )
    results = [
        pl.read_parquet(p).with_columns(pl.lit(p.stem).alias("shard"))
        for p in sorted(out_dir.glob("*.parquet"))
    ]
    return pl.concat(results, how="diagonal_relaxed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.download")
    parser.add_argument("--data-dir", type=Path, default=config.data_dir())
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=60, help="seconds before round 2")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="only the first N images (smoke runs)")
    args = parser.parse_args()

    images = pl.read_parquet(args.data_dir / "ingest" / "images.parquet").sort("image_id")
    if args.limit:
        images = images.head(args.limit)
    summary = run_download(
        images,
        args.data_dir,
        img2dataset_fetch,
        rounds=args.rounds,
        backoff_s=args.backoff,
        max_attempts=args.max_attempts,
    )
    total = sum(summary.statuses.values())
    unreachable = sum(summary.statuses.get(s, 0) for s in UNREACHABLE)
    log.info(
        "Done: %d attempted this run in %.0fs (%.1f img/s). Totals %s; ok %.1f%%, "
        "dead-link (%s) %.1f%%",
        summary.attempted,
        summary.seconds,
        summary.images_per_s,
        summary.statuses,
        100 * summary.ok / total if total else 0,
        "+".join(sorted(UNREACHABLE)),
        100 * unreachable / total if total else 0,
    )


if __name__ == "__main__":
    main()
