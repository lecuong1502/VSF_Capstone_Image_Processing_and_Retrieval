"""Exact (SHA-256) and near-duplicate (PDQ) detection, clustering and precision evaluation."""

import io
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image, ImageDraw

from pipeline.files import LockedError, exclusive_lock
from pipeline.hashing import (
    SHEET_BINS,
    best_threshold,
    candidate_pairs,
    cluster,
    evaluate_labels,
    hamming,
    near_duplicate_pairs,
    pdq_of,
    run_hashing,
    sample_pairs,
    thumb_links,
    to_bits,
)


def scene(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, tuple(int(c) for c in rng.integers(0, 255, 3)))
    draw = ImageDraw.Draw(image)
    for _ in range(25):
        x, y = (int(v) for v in rng.integers(0, size[0], 2))
        draw.ellipse(
            (x, y % size[1], x + 120, y % size[1] + 90),
            fill=tuple(int(c) for c in rng.integers(0, 255, 3)),
        )
    return image


def jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# --- hashing primitives --------------------------------------------------------------------------


def test_pdq_is_robust_to_resize_and_recompression() -> None:
    original = scene(1)

    a, quality = pdq_of(jpeg(original))
    b, _ = pdq_of(jpeg(original.resize((320, 240)), quality=60))
    other, _ = pdq_of(jpeg(scene(2)))

    assert len(a) == 64 and quality > 50
    assert hamming(a, b) <= 16
    assert hamming(a, other) > 64


def test_hamming_and_bits() -> None:
    assert hamming("00" * 32, "00" * 32) == 0
    assert hamming("ff" + "00" * 31, "00" * 32) == 8
    bits = to_bits(["ff" + "00" * 31, "00" * 32])
    assert bits.shape == (2, 32) and bits.dtype == np.uint8 and bits[0, 0] == 255


def test_near_duplicate_pairs_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 256, (60, 32), dtype=np.uint8)
    bits[10] = bits[3]
    bits[11] = bits[3] ^ np.array([1] + [0] * 31, dtype=np.uint8)

    pairs = near_duplicate_pairs(bits, max_distance=100, block=7)

    expected = {
        (i, j, int(np.unpackbits(bits[i] ^ bits[j]).sum()))
        for i in range(60)
        for j in range(i + 1, 60)
        if np.unpackbits(bits[i] ^ bits[j]).sum() <= 100
    }
    assert set(pairs) == expected
    assert (3, 10, 0) in pairs and (3, 11, 1) in pairs


# --- clustering ----------------------------------------------------------------------------------

NEAR = "f0" + "00" * 31  # 4 bits from zeros
HASHES = pl.DataFrame(
    {
        "image_id": [1, 2, 3, 4, 5, 6, 7],
        "sha256": ["s1", "s1", "s3", "s4", "s5", "s6", "s1"],
        "pdq_hash": ["11" * 32, "11" * 32, "00" * 32, NEAR, "ff" * 32, "00" * 32, NEAR],
        "pdq_quality": [100, 100, 100, 100, 100, 10, 100],  # 6: too flat to trust PDQ
        "area": [100, 200, 50, 400, 100, 100, 10],
    }
)


def test_candidate_pairs_flag_same_sha_and_skip_low_quality() -> None:
    pairs = candidate_pairs(HASHES, max_distance=48, min_quality=50)
    by_pair = {(r["image_a"], r["image_b"]): r for r in pairs.iter_rows(named=True)}

    assert by_pair[(1, 2)]["same_sha"] is True and by_pair[(1, 2)]["distance"] == 0
    assert by_pair[(3, 4)]["same_sha"] is False and by_pair[(3, 4)]["distance"] == 4
    assert not [p for p in by_pair if 6 in p]


def test_cluster_methods_and_representative() -> None:
    pairs = candidate_pairs(HASHES, max_distance=48, min_quality=50)

    members = cluster(HASHES, pairs, max_distance=31)
    by_id = {r["image_id"]: r for r in members.iter_rows(named=True)}

    # 1, 2, 7 share a SHA; 7's PDQ also links them with 3-4 into one cluster.
    assert set(by_id) == {1, 2, 3, 4, 7}
    assert len({r["cluster_id"] for r in by_id.values()}) == 1
    rep = [i for i, r in by_id.items() if r["is_representative"]]
    assert rep == [4]  # largest original image
    assert by_id[4]["method"] == "representative" and by_id[4]["distance"] == 0
    assert by_id[3]["method"] == "pdq" and by_id[3]["distance"] == 4
    # 1 and 2 share a SHA with each other, not with the representative.
    assert by_id[1]["method"] == "pdq"


def test_cluster_exact_duplicates_of_the_representative() -> None:
    hashes = HASHES.filter(pl.col("image_id").is_in([1, 2]))
    pairs = candidate_pairs(hashes, max_distance=48, min_quality=50)

    members = cluster(hashes, pairs, max_distance=31)

    assert dict(members.select("image_id", "method").rows()) == {
        1: "sha256",
        2: "representative",
    }


# --- labeling and evaluation ---------------------------------------------------------------------


def test_sample_pairs_skips_sha_duplicates_and_records_bin_population() -> None:
    pairs = pl.DataFrame(
        {
            "image_a": list(range(200)),
            "image_b": list(range(200, 400)),
            "distance": [d % 49 for d in range(200)],
            "same_sha": [d < 20 for d in range(200)],
        }
    )

    first = sample_pairs(pairs, n=60, seed=1)
    second = sample_pairs(pairs, n=60, seed=1)

    assert first.equals(second)
    assert first.height == 60
    assert not set(first["image_a"]) & set(range(20))
    assert first["bin"].n_unique() == len(SHEET_BINS) - 1
    # Population = all non-SHA candidate pairs in the bin, used to weight precision.
    population = first.group_by("bin").agg(pl.col("bin_population").first())
    assert population["bin_population"].sum() == 180


def test_bins_put_the_default_threshold_on_an_edge() -> None:
    assert 31 in SHEET_BINS


def test_weighted_precision_by_threshold() -> None:
    sheet = pl.DataFrame(
        {
            "image_a": [1, 2, 3, 4],
            "image_b": [11, 12, 13, 14],
            "distance": [5, 6, 20, 30],
            "bin": ["(-1, 7]", "(-1, 7]", "(15, 23]", "(23, 31]"],
            "bin_population": [900, 900, 50, 50],
        }
    )
    labels = pl.DataFrame(
        {
            "image_a": [1, 2, 3, 4],
            "image_b": [11, 12, 13, 14],
            "label": ["yes", "yes", "no", "yes"],
        }
    )

    table = evaluate_labels(sheet, labels, [7, 23, 31])
    by_t = {r["threshold"]: r for r in table.iter_rows(named=True)}

    assert by_t[7]["precision"] == 1.0
    # bins ≤ 23: (900 pairs × 100% + 50 pairs × 0%) / 950
    assert by_t[23]["precision"] == pytest.approx(900 / 950)
    assert by_t[31]["precision"] == pytest.approx(950 / 1000)
    assert by_t[31]["labeled"] == 4
    assert best_threshold(table, target=0.95) == 31


def test_thumb_links_are_relative_to_the_sheet(tmp_path: Path) -> None:
    assets = pl.DataFrame({"image_id": [1, 2], "thumb_path": ["thumbs/1.webp", None]})

    links = thumb_links(assets, tmp_path / "data", tmp_path / "reports")

    assert links == {1: "../data/thumbs/1.webp"}


# --- end to end ----------------------------------------------------------------------------------


def _write_shard(path: Path, images: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in images.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    run = data / "download" / "runs" / "r1"
    run.mkdir(parents=True)
    base = scene(7)
    _write_shard(
        run / "00000.tar",
        {
            "000000000.jpg": jpeg(base),
            "000000001.jpg": jpeg(base.resize((400, 300)), quality=70),
            "000000002.jpg": jpeg(scene(8)),
            "000000003.jpg": b"not an image",
        },
    )
    pl.DataFrame(
        {
            "image_id": [1, 2, 3, 4, 5],
            "download_status": ["ok", "ok", "ok", "ok", "dead"],
            "sha256": ["a", "b", "c", "d", None],
            "real_w": [1280, 400, 640, 10, None],
            "real_h": [960, 300, 480, 10, None],
            "pdq_hash": [None] * 5,
            "shard_path": [
                "download/runs/r1/00000.tar#000000000",
                "download/runs/r1/00000.tar#000000001",
                "download/runs/r1/00000.tar#000000002",
                "download/runs/r1/00000.tar#000000003",
                None,
            ],
        },
        schema_overrides={"pdq_hash": pl.String, "real_w": pl.Int32, "real_h": pl.Int32},
    ).write_parquet(data / "download" / "image_assets.parquet")
    return data


def test_run_hashing_end_to_end(data_dir: Path) -> None:
    summary = run_hashing(data_dir, max_distance=31, min_quality=50, processes=1)

    assets = pl.read_parquet(data_dir / "download" / "image_assets.parquet")
    hashed = dict(assets.select("image_id", "pdq_hash").rows())
    assert hashed[1] and hashed[2] and hashed[3]
    assert hashed[4] is None and hashed[5] is None  # unreadable image, not downloaded
    members = pl.read_parquet(data_dir / "dedup" / "dedup_members.parquet")
    assert set(members["image_id"]) == {1, 2}
    assert summary["images"] == 4
    assert summary["unhashable"] == 1
    assert summary["duplicate_images"] == 1


def test_run_hashing_rehashes_only_new_or_changed_images(data_dir: Path) -> None:
    run_hashing(data_dir, processes=1)
    first = pl.read_parquet(data_dir / "dedup" / "pdq.parquet")
    assets_file = data_dir / "download" / "image_assets.parquet"
    pl.read_parquet(assets_file).with_columns(
        pl.when(pl.col("image_id") == 3)
        .then(pl.lit("changed"))
        .otherwise(pl.col("sha256"))
        .alias("sha256")
    ).write_parquet(assets_file)

    run_hashing(data_dir, processes=1)

    second = pl.read_parquet(data_dir / "dedup" / "pdq.parquet")
    assert second.height == first.height
    assert dict(second.select("image_id", "sha256").rows())[3] == "changed"


def test_run_hashing_refuses_to_run_during_a_download(data_dir: Path) -> None:
    with exclusive_lock(data_dir / "download" / ".lock"), pytest.raises(LockedError):
        run_hashing(data_dir, processes=1)
