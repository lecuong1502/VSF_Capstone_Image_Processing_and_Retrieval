"""Exact (SHA-256) and near-duplicate (PDQ) detection over the downloaded images (spec §4.1 T1).

- `run`: PDQ hash (256 bit + quality) for every `ok` image → `image_assets.pdq_hash` and
  `$DATA_DIR/dedup/pdq.parquet` (re-runs hash only new images or images whose SHA changed);
  candidate pairs up to distance 48 → `pdq_pairs.parquet`; union-find clusters of SHA-equal or
  PDQ-near (Hamming ≤ τ, both hashes of quality ≥ 50) images → `dedup_members.parquet`.
  The representative is an image without a brand/logo or phone number if the cluster has one
  (flags from `dedup/representative_flags.parquet`, written by the watermark head #14 and OCR
  #16; images without a flag count as clean), then the largest original image. Holds the data
  lock, so it never runs alongside `make download`.
- `label-sheet`: 300 non-SHA candidate pairs stratified by distance (bins edged at τ=31) →
  `reports/dedup_pairs.html` (keyboard labeling, CSV export) + `dedup/label_sheet.parquet` with
  each bin's population so precision can be weighted back to the real pair distribution.
- `evaluate --labels CSV`: population-weighted precision per threshold, and the largest τ with
  precision ≥ 0.95.

Matching is exact brute force over packed 64-bit words (fine for ~60k images); at 10–100M images
the spec's FAISS IndexBinaryHNSW replaces `near_duplicate_pairs`. The embedding-cosine pass
(SemDeDup within an entity) needs embeddings and comes after #11.

Usage: python -m pipeline.hashing run|label-sheet|evaluate [...]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pdqhash
import polars as pl
from PIL import Image

from pipeline import config
from pipeline.download import IMAGE_ERRORS, assets_path, lock_path
from pipeline.files import atomic_write, atomic_write_text, exclusive_lock
from pipeline.shards import group_by_shard, iter_shard_images

log = logging.getLogger(__name__)

# Tuned on 300 hand-labeled pairs (spec §4.1 started at 31): population-weighted precision is
# 0.980 at 31 and 0.978 at 48, the widest labeled distance.
DEFAULT_MAX_DISTANCE = 48
DEFAULT_MIN_QUALITY = 50  # PDQ's own guidance: hashes of flat images below this are unreliable
CANDIDATE_DISTANCE = 48  # candidate pairs kept beyond τ so labels can move τ up
# Bin edges (lo, hi]; -1 so distance 0 falls in the first bin; 31 = default τ is an edge.
SHEET_BINS = (-1, 7, 15, 23, 31, 39, 48)
SHEET_SIZE = 300
PDQ_SIDE = 512  # PDQ downsamples internally; decoding at ~512 px is enough and much faster
ASSETS = Path(__file__).parent / "assets"

PAIR_SCHEMA = {
    "image_a": pl.Int64,
    "image_b": pl.Int64,
    "distance": pl.Int32,
    "same_sha": pl.Boolean,
}
PDQ_SCHEMA = {
    "image_id": pl.Int64,
    "sha256": pl.String,
    "pdq_hash": pl.String,
    "pdq_quality": pl.Int32,
}


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


def _hash_shard(job: tuple[Path, dict[str, int]]) -> list[tuple[int, str | None, int | None]]:
    """(image_id, pdq, quality) per wanted image; pdq/quality None if the image can't be read."""
    tar_path, wanted = job
    out: list[tuple[int, str | None, int | None]] = []
    seen: set[int] = set()
    for image_id, image_bytes in iter_shard_images(tar_path, wanted):
        seen.add(image_id)
        try:
            pdq, quality = pdq_of(image_bytes)
        except IMAGE_ERRORS as exc:
            log.warning("PDQ failed for image %d: %s", image_id, exc)
            out.append((image_id, None, None))
            continue
        out.append((image_id, pdq, quality))
    out.extend((i, None, None) for i in wanted.values() if i not in seen)
    return out


# --- pairs and clustering ------------------------------------------------------------------------


def candidate_pairs(hashes: pl.DataFrame, max_distance: int, min_quality: int) -> pl.DataFrame:
    """Pairs of trusted-PDQ images within `max_distance`, flagged when their SHA-256 is equal."""
    trusted = hashes.filter(pl.col("pdq_quality") >= min_quality).sort("image_id")
    ids = trusted["image_id"].to_list()
    shas = trusted["sha256"].to_list()
    bits = to_bits(trusted["pdq_hash"].to_list())
    rows = [
        (ids[a], ids[b], d, shas[a] is not None and shas[a] == shas[b])
        for a, b, d in near_duplicate_pairs(bits, max_distance)
    ]
    return pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row")


def cluster(hashes: pl.DataFrame, pairs: pl.DataFrame, max_distance: int) -> pl.DataFrame:
    """Union-find clusters of SHA-equal images and `pairs` within `max_distance`.

    `hashes` has image_id, sha256, pdq_hash, area and optionally has_brand_or_phone (the
    representative avoids flagged images, then is the largest). Returns members of clusters
    with ≥ 2 images:
    cluster_id (= representative's image_id), method ("representative"; "sha256" if identical to
    the representative; otherwise "pdq"), distance to the representative, is_representative.
    """
    ids = hashes["image_id"].to_list()
    index = {image_id: i for i, image_id in enumerate(ids)}
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

    shas = hashes["sha256"].to_list()
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, sha in enumerate(shas):
        if sha:
            by_sha[sha].append(i)
    for group in by_sha.values():
        for i in group[1:]:
            union(group[0], i)
    near = pairs.filter(pl.col("distance") <= max_distance).select("image_a", "image_b")
    for a, b in near.rows():
        if a in index and b in index:
            union(index[a], index[b])

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ids)):
        groups[find(i)].append(i)
    areas = hashes["area"].fill_null(0).to_list()
    flagged = (
        hashes["has_brand_or_phone"].fill_null(False).to_list()
        if "has_brand_or_phone" in hashes.columns
        else [False] * len(ids)
    )
    pdqs = hashes["pdq_hash"].to_list()
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda i: (not flagged[i], areas[i], -ids[i]))
        for i in members:
            distance: int | None
            if i == rep:
                method, distance = "representative", 0
            else:
                method = "sha256" if shas[i] and shas[i] == shas[rep] else "pdq"
                distance = hamming(pdqs[i], pdqs[rep]) if pdqs[i] and pdqs[rep] else None
            rows.append((ids[rep], ids[i], method, distance, i == rep))
    schema = {
        "cluster_id": pl.Int64,
        "image_id": pl.Int64,
        "method": pl.String,
        "distance": pl.Int32,
        "is_representative": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema, orient="row").sort("cluster_id", "image_id")


# --- run -----------------------------------------------------------------------------------------


def run_hashing(
    data_dir: Path,
    *,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    min_quality: int = DEFAULT_MIN_QUALITY,
    processes: int | None = None,
) -> dict[str, int | float]:
    """Hash new images, rewrite image_assets.pdq_hash, candidate pairs, clusters and summary.json.

    Returns the summary: images considered, unhashable/low-quality counts, cluster stats and the
    duplicate rate (non-representative cluster members / images). Raises LockedError while
    `make download` is running.
    """
    with exclusive_lock(lock_path(data_dir)):
        return _run_hashing(data_dir, max_distance, min_quality, processes)


def _run_hashing(
    data_dir: Path, max_distance: int, min_quality: int, processes: int | None
) -> dict[str, int | float]:
    out_dir = data_dir / "dedup"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = pl.read_parquet(assets_path(data_dir))
    ok = assets.filter(pl.col("download_status") == "ok")
    known = _update_pdq(ok, data_dir, out_dir / "pdq.parquet", processes)

    current = ok.select("image_id", "sha256").join(known.drop("sha256"), on="image_id", how="left")
    updated = (
        assets.drop("pdq_hash")
        .join(current.select("image_id", "pdq_hash"), on="image_id", how="left")
        .select(assets.columns)
    )
    atomic_write(assets_path(data_dir), updated.write_parquet)

    hashes = current.join(
        ok.select(
            "image_id",
            (pl.col("real_w").cast(pl.Int64) * pl.col("real_h").cast(pl.Int64)).alias("area"),
        ),
        on="image_id",
    )
    flags_path = out_dir / "representative_flags.parquet"
    if flags_path.exists():
        flags = pl.read_parquet(flags_path).select("image_id", "has_brand_or_phone")
        hashes = hashes.join(flags, on="image_id", how="left")
    pairs = candidate_pairs(hashes, CANDIDATE_DISTANCE, min_quality)
    atomic_write(out_dir / "pdq_pairs.parquet", pairs.write_parquet)
    members = cluster(hashes, pairs, max_distance)
    atomic_write(out_dir / "dedup_members.parquet", members.write_parquet)

    summary = _summary(hashes, members, max_distance, min_quality)
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2))
    return summary


def _update_pdq(
    ok: pl.DataFrame, data_dir: Path, pdq_path: Path, processes: int | None
) -> pl.DataFrame:
    """PDQ rows for every ok image, hashing only images that are new or whose SHA changed."""
    empty = pl.DataFrame(schema=PDQ_SCHEMA)
    known = pl.read_parquet(pdq_path) if pdq_path.exists() else empty
    known = known.join(ok.select("image_id", "sha256"), on=["image_id", "sha256"], how="semi")
    todo = ok.join(known.select("image_id"), on="image_id", how="anti")
    if todo.height:
        jobs = group_by_shard(todo, data_dir)
        log.info("PDQ-hashing %d images from %d shards", todo.height, len(jobs))
        # spawn, not fork: polars is multi-threaded and fork() can deadlock the children.
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes or max(1, (os.cpu_count() or 2) - 2)) as pool:
            results = [r for chunk in pool.imap_unordered(_hash_shard, jobs.items()) for r in chunk]
        hashed_schema = {k: v for k, v in PDQ_SCHEMA.items() if k != "sha256"}
        fresh = pl.DataFrame(results, schema=hashed_schema, orient="row").join(
            todo.select("image_id", "sha256"), on="image_id"
        )
        known = pl.concat([known, fresh.select(list(PDQ_SCHEMA))])
        atomic_write(pdq_path, known.sort("image_id").write_parquet)
    return known


def _summary(
    hashes: pl.DataFrame, members: pl.DataFrame, max_distance: int, min_quality: int
) -> dict[str, int | float]:
    duplicates = members.filter(~pl.col("is_representative"))
    sizes = members.group_by("cluster_id").agg(pl.len().alias("size"))
    return {
        "images": hashes.height,
        "unhashable": int(hashes["pdq_hash"].null_count()),
        "low_quality_pdq": int((hashes["pdq_quality"] < min_quality).sum()),
        "clusters": sizes.height,
        "largest_cluster": int(sizes.select(pl.col("size").max()).item() or 0),
        "duplicate_images": duplicates.height,
        "exact_duplicate_images": int((duplicates["method"] == "sha256").sum()),
        "near_duplicate_images": int((duplicates["method"] == "pdq").sum()),
        # Union-find chains: members linked to the representative only through other members.
        "members_beyond_tau": int((duplicates["distance"] > max_distance).sum()),
        "duplicate_rate": duplicates.height / hashes.height if hashes.height else 0.0,
        "max_distance": max_distance,
        "min_quality": min_quality,
    }


# --- labeling and evaluation ---------------------------------------------------------------------


def _bin_expr() -> pl.Expr:
    edges = [
        pl.when(pl.col("distance") <= hi).then(pl.lit(f"({lo}, {hi}]"))
        for lo, hi in zip(SHEET_BINS[:-1], SHEET_BINS[1:], strict=True)
    ]
    return pl.coalesce(edges).alias("bin")


def sample_pairs(pairs: pl.DataFrame, n: int = SHEET_SIZE, seed: int = 0) -> pl.DataFrame:
    """`n` non-SHA pairs spread evenly over SHEET_BINS, with each bin's population."""
    population = pairs.filter(~pl.col("same_sha") & (pl.col("distance") <= SHEET_BINS[-1]))
    population = population.with_columns(_bin_expr())
    groups = sorted(
        population.partition_by("bin", as_dict=False),
        key=lambda g: int(g.select(pl.col("distance").min()).item()),
    )
    quota = [0] * len(groups)
    remaining, open_groups = n, list(range(len(groups)))
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
    picked = [
        g.sample(q, seed=seed).with_columns(pl.lit(g.height).alias("bin_population"))
        for g, q in zip(groups, quota, strict=True)
        if q
    ]
    if not picked:
        return population.head(0).with_columns(pl.lit(0).alias("bin_population"))
    return pl.concat(picked).sample(fraction=1.0, seed=seed, shuffle=True)


def evaluate_labels(
    sheet: pl.DataFrame, labels: pl.DataFrame, thresholds: list[int]
) -> pl.DataFrame:
    """Population-weighted precision per threshold (thresholds should be bin upper edges).

    precision(t) = Σ_bins≤t population_b · dup_rate_b / Σ_bins≤t population_b, where dup_rate_b
    is the share of "yes" among the labeled pairs of bin b ("skip" labels are ignored).
    """
    labeled = sheet.join(
        labels.filter(pl.col("label").is_in(["yes", "no"])), on=["image_a", "image_b"]
    )
    per_bin = labeled.group_by("bin").agg(
        pl.col("distance").max().alias("hi"),
        pl.col("bin_population").first(),
        pl.len().alias("labeled"),
        (pl.col("label") == "yes").mean().alias("dup_rate"),
    )
    rows = []
    for t in thresholds:
        within = per_bin.filter(pl.col("hi") <= t)
        population = int(within["bin_population"].sum())
        expected_dups = float((within["bin_population"] * within["dup_rate"]).sum())
        rows.append(
            {
                "threshold": t,
                "labeled": int(within["labeled"].sum()),
                "pairs": population,
                "precision": expected_dups / population if population else 0.0,
            }
        )
    return pl.DataFrame(rows)


def best_threshold(table: pl.DataFrame, target: float = 0.95) -> int | None:
    """Largest labeled threshold whose cumulative precision (all pairs ≤ τ) is ≥ target."""
    ok = table.filter((pl.col("labeled") > 0) & (pl.col("precision") >= target))
    return int(ok.select(pl.col("threshold").max()).item()) if ok.height else None


def thumb_links(assets: pl.DataFrame, data_dir: Path, out_dir: Path) -> dict[int, str]:
    """image_id → thumbnail path relative to the directory the sheet is written to."""
    rows = assets.select("image_id", "thumb_path").drop_nulls().iter_rows()
    return {image_id: os.path.relpath(data_dir / thumb, out_dir) for image_id, thumb in rows}


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def render_label_sheet(sample: pl.DataFrame, thumb_for: dict[int, str]) -> str:
    pair_ids = ",".join(f"{a}-{b}" for a, b in sample.select("image_a", "image_b").rows())
    sheet_id = hashlib.sha256(pair_ids.encode()).hexdigest()[:12]
    cards = "".join(
        f'<div class="pair" data-a="{r["image_a"]}" data-b="{r["image_b"]}" '
        f'data-d="{r["distance"]}"><div class="imgs">'
        f'<img loading="lazy" src="{html.escape(thumb_for.get(r["image_a"], ""))}" alt="A">'
        f'<img loading="lazy" src="{html.escape(thumb_for.get(r["image_b"], ""))}" alt="B">'
        f'</div><div class="meta"><span>#{i + 1} · PDQ distance {r["distance"]}</span>'
        '<span class="tag"></span></div></div>'
        for i, r in enumerate(sample.iter_rows(named=True))
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Dedup Labels</title><style>{_asset('dedup_sheet.css')}</style></head>"
        f'<body data-sheet="{sheet_id}"><header>'
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
    sheet_p.add_argument("--pairs", type=int, default=SHEET_SIZE)
    default_sheet = config.REPO_ROOT / "reports" / "dedup_pairs.html"
    sheet_p.add_argument("--out", type=Path, default=default_sheet)
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    data_dir = config.data_dir()
    dedup_dir = data_dir / "dedup"

    if args.command == "run":
        summary = run_hashing(
            data_dir, max_distance=args.max_distance, min_quality=args.min_quality
        )
        log.info("Dedup summary: %s", json.dumps(summary))
    elif args.command == "label-sheet":
        sample = sample_pairs(pl.read_parquet(dedup_dir / "pdq_pairs.parquet"), args.pairs)
        atomic_write(dedup_dir / "label_sheet.parquet", sample.write_parquet)
        links = thumb_links(
            pl.read_parquet(assets_path(data_dir)), data_dir, args.out.parent.resolve()
        )
        atomic_write_text(args.out, render_label_sheet(sample, links))
        counts = sample.group_by("bin").agg(pl.len(), pl.col("distance").min()).sort("distance")
        per_bin = dict(counts.select("bin", "len").rows())
        log.info("Wrote %s (%d pairs per bin: %s)", args.out, sample.height, per_bin)
    else:
        sheet = pl.read_parquet(dedup_dir / "label_sheet.parquet")
        table = evaluate_labels(sheet, pl.read_csv(args.labels), list(SHEET_BINS[1:]))
        with pl.Config(tbl_rows=-1, float_precision=3):
            log.info("Population-weighted precision by PDQ threshold\n%s", table)
        log.info("Largest threshold with precision ≥ 0.95: %s", best_threshold(table))


if __name__ == "__main__":
    main()
