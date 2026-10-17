"""Download orchestration with a fake fetcher: status mapping, retries, idempotence, thumbnails."""

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from pipeline.download import (
    UNREACHABLE,
    Summary,
    classify,
    load_assets,
    plan,
    run_download,
)

IMAGES = pl.DataFrame(
    {
        "image_id": [1, 2, 3, 4, 5, 6],
        "image_url": [
            "https://ok.example/a.jpg",
            "https://flaky.example/b.jpg",  # times out once, then works
            "https://gone.example/c.jpg",  # 404
            "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1",
            "https://html.example/d.jpg",  # returns an HTML page
            "x-raw-image:///abc",
        ],
    }
)


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


class FakeFetcher:
    """Mimics img2dataset: writes one shard (.tar + results) per call."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def __call__(self, batch: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
        ids = batch["image_id"].to_list()
        self.calls.append(ids)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with tarfile.open(out_dir / "00000.tar", "w") as tar:
            for key, (image_id, url) in enumerate(batch.select("image_id", "url").rows()):
                status, error = "success", None
                if "flaky" in url and len(self.calls) == 1:
                    status, error = "failed_to_download", "<urlopen error timed out>"
                elif "gone" in url:
                    status, error = "failed_to_download", "HTTP Error 404: Not Found"
                elif "fbsbx" in url or "html" in url:
                    status, error = "failed_to_resize", "Image decoding error"
                k = f"{key:09d}"
                if status == "success":
                    data = _jpeg(640, 480)
                    info = tarfile.TarInfo(f"{k}.jpg")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                ok = status == "success"
                rows.append(
                    {
                        "image_id": image_id,
                        "url": url,
                        "key": k,
                        "status": status,
                        "error_message": error,
                        "width": 640 if ok else None,
                        "height": 480 if ok else None,
                        "original_width": 1280 if ok else None,
                        "original_height": 960 if ok else None,
                        "sha256": f"sha-{image_id}" if ok else None,
                        "shard": "00000",
                    }
                )
        return pl.DataFrame(rows)


def run(
    tmp_path: Path,
    fetch: Callable[[pl.DataFrame, Path], pl.DataFrame],
    rounds: int = 3,
    chunk_size: int = 100,
) -> Summary:
    return run_download(
        IMAGES,
        tmp_path / "data",
        fetch,
        rounds=rounds,
        backoff_s=0,
        max_attempts=3,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize(
    ("status", "error", "url", "expected"),
    [
        ("success", None, "https://a/x.jpg", "ok"),
        ("failed_to_download", "HTTP Error 404: Not Found", "https://a/x.jpg", "dead"),
        ("failed_to_download", "HTTP Error 403: Forbidden", "https://a/x.jpg", "dead"),
        (
            "failed_to_download",
            "<urlopen error [Errno -2] Name or service not known>",
            "https://a/x.jpg",
            "dead",
        ),
        ("failed_to_download", "<urlopen error timed out>", "https://a/x.jpg", "failed"),
        ("failed_to_download", "HTTP Error 503: Service Unavailable", "https://a/x.jpg", "failed"),
        (
            "failed_to_download",
            "<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>",
            "https://a/x.jpg",
            "ssl_error",
        ),
        (
            "failed_to_download",
            "<urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred>",
            "https://a/x.jpg",
            "failed",
        ),
        (
            "failed_to_download",
            "Use of image disallowed by X-Robots-Tag directives",
            "https://a/x.jpg",
            "opted_out",
        ),
        ("failed_to_resize", "Image decoding error", "https://lookaside.fbsbx.com/x", "blocked"),
        ("failed_to_resize", "Image decoding error", "https://a/x.jpg", "not_image"),
    ],
)
def test_classify(status: str, error: str | None, url: str, expected: str) -> None:
    assert classify(status, error, url) == expected


def test_full_run_statuses_and_retry(tmp_path: Path) -> None:
    fetch = FakeFetcher()

    summary = run(tmp_path, fetch)
    assets = {r["image_id"]: r for r in load_assets(tmp_path / "data").iter_rows(named=True)}

    assert {i: a["download_status"] for i, a in assets.items()} == {
        1: "ok",
        2: "ok",  # retried in round 2
        3: "dead",
        4: "blocked",
        5: "not_image",
        6: "invalid_url",
    }
    assert fetch.calls == [[1, 2, 3, 4, 5], [2]]  # invalid URL never fetched
    assert assets[2]["attempts"] == 2
    assert assets[1]["real_w"] == 1280 and assets[1]["stored_w"] == 640
    assert summary.attempted == 5
    assert summary.ok == 2


def test_thumbnails_are_small_webp(tmp_path: Path) -> None:
    run(tmp_path, FakeFetcher())
    assets = load_assets(tmp_path / "data").filter(pl.col("download_status") == "ok")

    for thumb in assets["thumb_path"]:
        path = tmp_path / "data" / thumb
        with Image.open(path) as image:
            assert image.format == "WEBP"
            assert max(image.size) == 384
    for shard in assets["shard_path"]:
        tar_path, _, key = shard.partition("#")
        assert (tmp_path / "data" / tar_path).exists() and len(key) == 9


def test_rerun_only_retries_retryable_failures(tmp_path: Path) -> None:
    run(tmp_path, FakeFetcher())

    second = FakeFetcher()
    run(tmp_path, second)

    assert second.calls == []  # nothing left: all ok or permanently failed


def test_failures_stop_after_max_attempts(tmp_path: Path) -> None:
    class AlwaysTimeout(FakeFetcher):
        def __call__(self, batch: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
            result = super().__call__(batch, out_dir)
            return result.with_columns(
                pl.lit("failed_to_download").alias("status"),
                pl.lit("<urlopen error timed out>").alias("error_message"),
            )

    fetch = AlwaysTimeout()
    run(tmp_path, fetch, rounds=5)

    assets = load_assets(tmp_path / "data")
    assert len(fetch.calls) == 3
    assert set(assets.filter(pl.col("download_status") == "failed")["attempts"]) == {3}


def test_plan_skips_done_and_permanent(tmp_path: Path) -> None:
    run(tmp_path, FakeFetcher())
    more = pl.concat([IMAGES, pl.DataFrame({"image_id": [7], "image_url": ["https://new/e.jpg"]})])

    todo = plan(more, load_assets(tmp_path / "data"), max_attempts=3)

    assert todo["image_id"].to_list() == [7]
    assert {"dead", "blocked", "invalid_url", "opted_out"} <= UNREACHABLE


def test_each_chunk_is_saved_so_a_crash_keeps_earlier_chunks(tmp_path: Path) -> None:
    class CrashOnSecondChunk(FakeFetcher):
        def __call__(self, batch: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
            if self.calls:
                raise KeyboardInterrupt
            return super().__call__(batch, out_dir)

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, CrashOnSecondChunk(), chunk_size=2)
    assert set(load_assets(tmp_path / "data")["image_id"]) == {1, 2, 6}

    resumed = FakeFetcher()
    run(tmp_path, resumed, chunk_size=2)
    assert resumed.calls[0] == [2, 3]  # 2 was a transient failure, 1 is done
    assert 1 not in {i for call in resumed.calls for i in call}


def test_orphaned_run_dirs_are_removed(tmp_path: Path) -> None:
    orphan = tmp_path / "data" / "download" / "runs" / "20000101T000000-r1-c000"
    orphan.mkdir(parents=True)
    (orphan / "00000.tar").write_bytes(b"partial")

    run(tmp_path, FakeFetcher())

    assert not orphan.exists()
    referenced = load_assets(tmp_path / "data")["shard_path"].drop_nulls()
    assert all((tmp_path / "data" / p.partition("#")[0]).exists() for p in referenced)


def test_bad_image_does_not_break_bookkeeping(tmp_path: Path) -> None:
    class CorruptFirstImage(FakeFetcher):
        def __call__(self, batch: pl.DataFrame, out_dir: Path) -> pl.DataFrame:
            result = super().__call__(batch, out_dir)
            with tarfile.open(out_dir / "00000.tar", "a") as tar:
                info = tarfile.TarInfo("000000000.jpg")
                info.size = 3
                tar.addfile(info, io.BytesIO(b"bad"))  # later member wins on read
            return result

    run(tmp_path, CorruptFirstImage())

    asset = load_assets(tmp_path / "data").filter(pl.col("image_id") == 1).row(0, named=True)
    assert asset["download_status"] == "ok"
    assert asset["thumb_path"] is None


def test_transparent_images_get_a_white_thumbnail_background(tmp_path: Path) -> None:
    from pipeline.download import _write_thumbnail

    buf = io.BytesIO()
    Image.new("RGBA", (500, 500), (0, 0, 0, 0)).save(buf, "PNG")
    out = tmp_path / "t.webp"

    _write_thumbnail(buf.getvalue(), out)

    with Image.open(out) as thumb:
        assert thumb.convert("RGB").getpixel((10, 10)) == (255, 255, 255)
