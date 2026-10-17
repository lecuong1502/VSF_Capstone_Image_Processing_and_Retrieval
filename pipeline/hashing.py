"""Exact (SHA-256) and near-duplicate (PDQ) detection over the downloaded images (spec §4.1 T1).

- `run`: PDQ hash (256 bit + quality) for every `ok` image, stored in `image_assets.pdq_hash`
  and `$DATA_DIR/dedup/pdq.parquet`; all pairs within `--sheet-distance` saved to
  `pdq_pairs.parquet`; union-find clusters of SHA-equal or PDQ-near images (Hamming ≤ τ, both
  hashes of quality ≥ `--min-quality`) written to `dedup_members.parquet`. The representative of a
  cluster is its largest original image. Re-running only hashes images without a PDQ hash.
- `label-sheet`: 300 candidate pairs stratified by distance → `reports/dedup_pairs.html`, a
  keyboard labeling page that exports a CSV.
- `evaluate --labels CSV`: precision/recall per threshold, to choose τ (target precision ≥ 0.95).

Matching is exact brute force over packed 64-bit words (fine for ~60k images); at 10–100M images
the spec's FAISS IndexBinaryHNSW replaces `near_duplicate_pairs`. The embedding-cosine pass
(SemDeDup within an entity) needs embeddings and comes after #11.

Usage: python -m pipeline.hashing run|label-sheet|evaluate [...]
"""

from __future__ import annotations

import argparse
import html
import io
import json
import logging
import multiprocessing
import os
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pdqhash
import polars as pl
from PIL import Image

from pipeline import config
from pipeline.download import assets_path
from pipeline.files import atomic_write, atomic_write_text

log = logging.getLogger(__name__)

DEFAULT_MAX_DISTANCE = 31  # spec §4.1 starting threshold, tuned on hand-labeled pairs
DEFAULT_MIN_QUALITY = 50  # PDQ's own guidance: hashes of flat images below this are unreliable
SHEET_DISTANCE = 48  # keep candidate pairs a bit beyond τ so the labels can move τ up
SHEET_BINS = (0, 8, 16, 24, 32, 40, 48)
PDQ_SIDE = 512  # PDQ downsamples internally; decoding at ~512 px is enough and much faster


# --- hashing -------------------------------------------------------------------------------------


def pdq_of(image_bytes: bytes) -> tuple[str, int]:
    """PDQ hash as 64 hex chars, plus PDQ's quality score (0–100)."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.draft("RGB", (PDQ_SIDE, PDQ_SIDE))
        pixels = np.asarray(image.convert("RGB"))
    bits, quality = pdqhash.compute(pixels)
    return np.packbits(bits.astype(np.uint8)).tobytes().hex(), int(quality)


def hamming(a: str, b: str) -> int:
    return int.bit_count(int(a, 16) ^ int(b, 16))


def to_bits(hashes: list[str]) -> np.ndarray:
    """(n, 32) uint8 matrix from 64-hex-char PDQ hashes."""
    return np.frombuffer(bytes.fromhex("".join(hashes)), dtype=np.uint8).reshape(-1, 32)


def near_duplicate_pairs(
    bits: np.ndarray, max_distance: int, block: int = 2048
) -> list[tuple[int, int, int]]:
    """All index pairs (i < j) with Hamming distance ≤ max_distance, by blocked brute force."""
    words = np.ascontiguousarray(bits).view(np.uint64)  # (n, 4)
    n = words.shape[0]
    pairs: list[tuple[int, int, int]] = []
    for i0 in range(0, n, block):
        a = words[i0 : i0 + block]
        for j0 in range(i0, n, block):
            b = words[j0 : j0 + block]
            dist = np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(axis=2, dtype=np.int32)
            ii, jj = np.nonzero(dist <= max_distance)
            for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
                if i0 + i < j0 + j:
                    pairs.append((i0 + i, j0 + j, int(dist[i, j])))
    return pairs


def _hash_shard(job: tuple[Path, dict[str, int]]) -> list[tuple[int, str, int]]:
    tar_path, wanted = job
    out: list[tuple[int, str, int]] = []
    with tarfile.open(tar_path) as tar:
        for member in tar:
            key, _, ext = member.name.partition(".")
            if key not in wanted or ext not in {"jpg", "jpeg", "png", "webp"}:
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            try:
                pdq, quality = pdq_of(data.read())
            except (OSError, ValueError) as exc:
                log.warning("PDQ failed for image %d: %s", wanted[key], exc)
                continue
            out.append((wanted[key], pdq, quality))
    return out


# --- clustering ----------------------------------------------------------------------------------


def cluster(hashes: pl.DataFrame, max_distance: int, min_quality: int) -> pl.DataFrame:
    """Union-find clusters of SHA-equal or PDQ-near images. `hashes` has image_id, sha256,
    pdq_hash, pdq_quality, area. Returns only members of clusters with ≥ 2 images."""
    ids = hashes["image_id"].to_list()
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, sha in enumerate(hashes["sha256"].to_list()):
        if sha:
            by_sha[sha].append(i)
    sha_dup = set()
    for group in by_sha.values():
        if len(group) > 1:
            sha_dup.update(group)
            for i in group[1:]:
                union(group[0], i)

    trusted = [
        i
        for i, (pdq, q) in enumerate(
            zip(hashes["pdq_hash"].to_list(), hashes["pdq_quality"].to_list(), strict=True)
        )
        if pdq and q is not None and q >= min_quality
    ]
    if trusted:
        bits = to_bits([hashes["pdq_hash"][i] for i in trusted])
        for a, b, _ in near_duplicate_pairs(bits, max_distance):
            union(trusted[a], trusted[b])

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        groups[find(i)].append(i)
    areas = hashes["area"].fill_null(0).to_list()
    pdqs = hashes["pdq_hash"].to_list()
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda i: (areas[i], -ids[i]))
        for i in members:
            distance = hamming(pdqs[i], pdqs[rep]) if pdqs[i] and pdqs[rep] else None
            rows.append(
                {
                    "cluster_id": ids[rep],
                    "image_id": ids[i],
                    "method": "sha256" if i in sha_dup else "pdq",
                    "distance": 0 if i == rep else distance,
                    "is_representative": i == rep,
                }
            )
    schema = {
        "cluster_id": pl.Int64,
        "image_id": pl.Int64,
        "method": pl.String,
        "distance": pl.Int32,
        "is_representative": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema).sort("cluster_id", "image_id")


# --- run -----------------------------------------------------------------------------------------


def run_hashing(
    data_dir: Path,
    *,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    min_quality: int = DEFAULT_MIN_QUALITY,
    processes: int | None = None,
) -> dict[str, float]:
    out_dir = data_dir / "dedup"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdq_path = out_dir / "pdq.parquet"
    assets = pl.read_parquet(assets_path(data_dir))
    ok = assets.filter(pl.col("download_status") == "ok")
    known = pl.read_parquet(pdq_path) if pdq_path.exists() else None

    todo = ok if known is None else ok.join(known.select("image_id"), on="image_id", how="anti")
    if todo.height:
        jobs: dict[Path, dict[str, int]] = defaultdict(dict)
        for image_id, shard in todo.select("image_id", "shard_path").iter_rows():
            tar, _, key = shard.partition("#")
            jobs[data_dir / tar][key] = image_id
        log.info("PDQ-hashing %d images from %d shards", todo.height, len(jobs))
        # spawn, not fork: polars is multi-threaded and fork() can deadlock the children.
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes or max(1, (os.cpu_count() or 2) - 2)) as pool:
            results = [r for chunk in pool.imap_unordered(_hash_shard, jobs.items()) for r in chunk]
        fresh = pl.DataFrame(
            results,
            schema={"image_id": pl.Int64, "pdq_hash": pl.String, "pdq_quality": pl.Int32},
            orient="row",
        )
        known = fresh if known is None else pl.concat([known, fresh])
        atomic_write(pdq_path, known.sort("image_id").write_parquet)
    assert known is not None

    updated = (
        assets.drop("pdq_hash")
        .join(known.select("image_id", "pdq_hash"), on="image_id", how="left")
        .select(assets.columns)
    )
    atomic_write(assets_path(data_dir), updated.write_parquet)

    hashes = ok.select(
        "image_id",
        "sha256",
        (pl.col("real_w").cast(pl.Int64) * pl.col("real_h").cast(pl.Int64)).alias("area"),
    ).join(known, on="image_id", how="left")

    trusted = hashes.filter(pl.col("pdq_quality") >= min_quality).sort("image_id")
    pair_index = near_duplicate_pairs(to_bits(trusted["pdq_hash"].to_list()), SHEET_DISTANCE)
    trusted_ids = trusted["image_id"].to_list()
    pairs = pl.DataFrame(
        [(trusted_ids[a], trusted_ids[b], d) for a, b, d in pair_index],
        schema={"image_a": pl.Int64, "image_b": pl.Int64, "distance": pl.Int32},
        orient="row",
    )
    atomic_write(out_dir / "pdq_pairs.parquet", pairs.write_parquet)

    members = cluster(hashes, max_distance, min_quality)
    atomic_write(out_dir / "dedup_members.parquet", members.write_parquet)

    duplicates = members.filter(~pl.col("is_representative"))
    sizes = members.group_by("cluster_id").len()
    summary = {
        "images": hashes.height,
        "low_quality_pdq": int((hashes["pdq_quality"] < min_quality).sum()),
        "clusters": sizes.height,
        "largest_cluster": int(sizes.select(pl.col("len").max()).item() or 0),
        "duplicate_images": duplicates.height,
        "exact_duplicate_images": int((duplicates["method"] == "sha256").sum()),
        "near_duplicate_images": int((duplicates["method"] == "pdq").sum()),
        "duplicate_rate": duplicates.height / hashes.height if hashes.height else 0.0,
        "max_distance": max_distance,
        "min_quality": min_quality,
    }
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2))
    return summary


# --- labeling and evaluation ---------------------------------------------------------------------


def sample_pairs(
    pairs: pl.DataFrame, n: int, bins: tuple[int, ...] = SHEET_BINS, seed: int = 0
) -> pl.DataFrame:
    """`n` pairs spread evenly over distance bins (lo, hi]; first bin includes 0."""
    labels = [
        pl.when(pl.col("distance") <= hi).then(pl.lit(f"({lo}, {hi}]"))
        for lo, hi in zip(bins[:-1], bins[1:], strict=True)
    ]
    labeled = pairs.filter(pl.col("distance") <= bins[-1]).with_columns(
        pl.coalesce(labels).alias("bin")
    )
    groups = sorted(
        labeled.partition_by("bin", as_dict=False),
        key=lambda g: int(g.select(pl.col("distance").min()).item()),
    )
    quota = [0] * len(groups)
    remaining = n
    open_groups = list(range(len(groups)))
    while remaining and open_groups:
        share = max(1, remaining // len(open_groups))
        for g in list(open_groups):
            take = min(share, groups[g].height - quota[g], remaining)
            quota[g] += take
            remaining -= take
            if quota[g] >= groups[g].height:
                open_groups.remove(g)
            if not remaining:
                break
    picked = [g.sample(q, seed=seed, shuffle=True) for g, q in zip(groups, quota, strict=True) if q]
    if not picked:
        return labeled.head(0)
    return pl.concat(picked).sample(fraction=1.0, seed=seed, shuffle=True).with_row_index("pair_id")


def precision_by_threshold(labels: pl.DataFrame, thresholds: list[int]) -> pl.DataFrame:
    """Precision (duplicates among pairs ≤ t) and recall (of all labeled duplicates) per t."""
    total_dups = int(labels["duplicate"].sum())
    rows = []
    for t in thresholds:
        within = labels.filter(pl.col("distance") <= t)
        dups = int(within["duplicate"].sum())
        rows.append(
            {
                "threshold": t,
                "pairs": within.height,
                "duplicates": dups,
                "precision": dups / within.height if within.height else 0.0,
                "recall": dups / total_dups if total_dups else 0.0,
            }
        )
    return pl.DataFrame(rows)


ASSETS = Path(__file__).parent / "assets"


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def render_label_sheet(sample: pl.DataFrame, thumb_for: dict[int, str]) -> str:
    cards = "".join(
        f'<div class="pair" data-id="{r["pair_id"]}" data-a="{r["image_a"]}" '
        f'data-b="{r["image_b"]}" data-d="{r["distance"]}"><div class="imgs">'
        f'<img loading="lazy" src="{html.escape(thumb_for.get(r["image_a"], ""))}" alt="A">'
        f'<img loading="lazy" src="{html.escape(thumb_for.get(r["image_b"], ""))}" alt="B">'
        f'</div><div class="meta"><span>#{r["pair_id"]} · PDQ distance {r["distance"]}</span>'
        '<span class="tag"></span></div></div>'
        for r in sample.iter_rows(named=True)
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Dedup Labels</title><style>{_asset('dedup_sheet.css')}</style></head>"
        "<body><header>"
        f"<strong>Near-duplicate labeling</strong> · {sample.height} pairs · labeled "
        '<span id="done">0</span> · <kbd>y</kbd> same image (crop/resize/re-encode/watermark) · '
        "<kbd>n</kbd> different photo · <kbd>s</kbd> skip · <kbd>↑</kbd>/<kbd>↓</kbd> move · "
        '<button id="export">Export CSV</button></header>'
        f"<main>{cards}</main><script>{_asset('dedup_sheet.js')}</script></body></html>"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.hashing")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE)
    run_p.add_argument("--min-quality", type=int, default=DEFAULT_MIN_QUALITY)
    sheet_p = sub.add_parser("label-sheet")
    sheet_p.add_argument("--pairs", type=int, default=300)
    sheet_p.add_argument(
        "--out", type=Path, default=config.REPO_ROOT / "reports" / "dedup_pairs.html"
    )
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    data_dir = config.data_dir()

    if args.command == "run":
        summary = run_hashing(
            data_dir, max_distance=args.max_distance, min_quality=args.min_quality
        )
        log.info("Dedup summary: %s", json.dumps(summary))
    elif args.command == "label-sheet":
        pairs = pl.read_parquet(data_dir / "dedup" / "pdq_pairs.parquet")
        sample = sample_pairs(pairs, args.pairs)
        assets = pl.read_parquet(assets_path(data_dir)).filter(pl.col("thumb_path").is_not_null())
        out_dir = args.out.parent.resolve()
        thumb_for = {
            image_id: os.path.relpath(data_dir / thumb, out_dir)
            for image_id, thumb in assets.select("image_id", "thumb_path").iter_rows()
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_label_sheet(sample, thumb_for), encoding="utf-8")
        log.info(
            "Wrote %s (%d pairs by distance bin: %s)",
            args.out,
            sample.height,
            dict(sample.group_by("bin").len().sort("bin").rows()),
        )
    else:
        labels = (
            pl.read_csv(args.labels)
            .filter(pl.col("label").is_in(["yes", "no"]))
            .with_columns((pl.col("label") == "yes").alias("duplicate"))
        )
        table = precision_by_threshold(labels, list(range(4, SHEET_DISTANCE + 1, 4)))
        with pl.Config(tbl_rows=-1, float_precision=3):
            log.info("%d labeled pairs\n%s", labels.height, table)
        good = table.filter(pl.col("precision") >= 0.95)
        if good.height:
            log.info("Largest threshold with precision ≥ 0.95: %d", good["threshold"].max())
        else:
            log.info("No threshold reaches precision 0.95")


if __name__ == "__main__":
    main()
