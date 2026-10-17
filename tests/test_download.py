"""Download orchestration with a fake fetcher: status mapping, retries, idempotence, thumbnails."""

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from pipeline.download import (
    RETRYABLE,
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


def run(tmp_path: Path, fetch: Callable[[pl.DataFrame, Path], pl.DataFrame], rounds: int = 3):  # type: ignore[no-untyped-def]
    return run_download(
        IMAGES, tmp_path / "data", fetch, rounds=rounds, backoff_s=0, max_attempts=3
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
            assert max(image.size) == 256
    shard = assets["shard_path"][0]
    assert shard.endswith(".tar#000000000") or "#" in shard


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
    assert "failed" in RETRYABLE
