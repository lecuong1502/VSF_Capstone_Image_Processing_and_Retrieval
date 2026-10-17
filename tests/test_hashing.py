"""Exact (SHA-256) and near-duplicate (PDQ) detection, clustering and precision evaluation."""

import io
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image, ImageDraw

from pipeline.hashing import (
    cluster,
    hamming,
    near_duplicate_pairs,
    pdq_of,
    precision_by_threshold,
    run_hashing,
    sample_pairs,
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


def test_pdq_is_robust_to_resize_and_recompression() -> None:
    original = scene(1)
    copy = original.resize((320, 240))

    a, quality = pdq_of(jpeg(original))
    b, _ = pdq_of(jpeg(copy, quality=60))
    other, _ = pdq_of(jpeg(scene(2)))

    assert len(a) == 64 and quality > 50
    assert hamming(a, b) <= 16
    assert hamming(a, other) > 64


def test_hamming() -> None:
    assert hamming("00" * 32, "00" * 32) == 0
    assert hamming("ff" + "00" * 31, "00" * 32) == 8


def test_near_duplicate_pairs_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 256, (60, 32), dtype=np.uint8)
    bits[10] = bits[3]
    bits[11] = bits[3] ^ np.array([1] + [0] * 31, dtype=np.uint8)  # distance 1

    pairs = near_duplicate_pairs(bits, max_distance=100, block=7)

    expected = {
        (i, j, int(np.unpackbits(bits[i] ^ bits[j]).sum()))
        for i in range(60)
        for j in range(i + 1, 60)
        if np.unpackbits(bits[i] ^ bits[j]).sum() <= 100
    }
    assert set(pairs) == expected
    assert (3, 10, 0) in pairs and (3, 11, 1) in pairs


def test_cluster_exact_and_near_duplicates() -> None:
    near = "f0" + "00" * 31  # 4 bits from zeros
    hashes = pl.DataFrame(
        {
            "image_id": [1, 2, 3, 4, 5, 6],
            "sha256": ["s1", "s1", "s3", "s4", "s5", "s6"],
            "pdq_hash": ["11" * 32, "11" * 32, "00" * 32, near, "ff" * 32, "00" * 32],
            "pdq_quality": [100, 100, 100, 100, 100, 10],  # 6: too flat to trust PDQ
            "area": [100, 200, 50, 400, 100, 100],
        }
    )

    members = cluster(hashes, max_distance=31, min_quality=50)
    by_id = {r["image_id"]: r for r in members.iter_rows(named=True)}

    assert set(by_id) == {1, 2, 3, 4}  # 5 and 6 are singletons
    assert by_id[1]["cluster_id"] == by_id[2]["cluster_id"]
    assert by_id[1]["method"] == "sha256"
    assert by_id[3]["cluster_id"] == by_id[4]["cluster_id"] != by_id[1]["cluster_id"]
    assert by_id[3]["method"] == "pdq" and by_id[3]["distance"] == 4
    # Representative = largest original image in the cluster.
    assert [i for i, r in by_id.items() if r["is_representative"]] == [2, 4]


def test_sample_pairs_is_stratified_and_deterministic() -> None:
    pairs = pl.DataFrame(
        {
            "image_a": list(range(100)),
            "image_b": list(range(100, 200)),
            "distance": list(range(100)),
        }
    )

    first = sample_pairs(pairs, n=20, bins=(0, 10, 20, 40), seed=1)
    second = sample_pairs(pairs, n=20, bins=(0, 10, 20, 40), seed=1)

    assert first.equals(second)
    assert first.height == 20
    assert first.select(pl.col("distance").max()).item() <= 40
    assert first["bin"].n_unique() == 3


def test_precision_by_threshold() -> None:
    labels = pl.DataFrame(
        {"distance": [0, 5, 10, 20, 30, 40], "duplicate": [True, True, True, True, False, False]}
    )

    table = {
        r["threshold"]: r
        for r in precision_by_threshold(labels, [10, 30, 40]).iter_rows(named=True)
    }

    assert table[10]["precision"] == 1.0 and table[10]["pairs"] == 3
    assert table[30]["precision"] == pytest.approx(4 / 5)
    assert table[40]["recall"] == 1.0


def _write_shard(path: Path, images: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in images.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_run_hashing_updates_assets_and_writes_clusters(tmp_path: Path) -> None:
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
        },
    )
    assets = pl.DataFrame(
        {
            "image_id": [1, 2, 3, 4],
            "download_status": ["ok", "ok", "ok", "dead"],
            "sha256": ["a", "b", "c", None],
            "real_w": [1280, 400, 640, None],
            "real_h": [960, 300, 480, None],
            "pdq_hash": [None, None, None, None],
            "shard_path": [
                "download/runs/r1/00000.tar#000000000",
                "download/runs/r1/00000.tar#000000001",
                "download/runs/r1/00000.tar#000000002",
                None,
            ],
        },
        schema_overrides={"pdq_hash": pl.String, "real_w": pl.Int32, "real_h": pl.Int32},
    )
    (data / "download").mkdir(exist_ok=True)
    assets.write_parquet(data / "download" / "image_assets.parquet")

    summary = run_hashing(data, max_distance=31, min_quality=50, processes=1)

    updated = pl.read_parquet(data / "download" / "image_assets.parquet")
    assert updated.filter(pl.col("download_status") == "ok")["pdq_hash"].null_count() == 0
    members = pl.read_parquet(data / "dedup" / "dedup_members.parquet")
    assert set(members["image_id"]) == {1, 2}
    assert summary["images"] == 3 and summary["duplicate_images"] == 1


def test_to_bits_roundtrip() -> None:
    bits = to_bits(["ff" + "00" * 31, "00" * 32])
    assert bits.shape == (2, 32) and bits.dtype == np.uint8 and bits[0, 0] == 255
