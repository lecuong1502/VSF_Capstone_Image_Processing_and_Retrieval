import io
import tarfile
from pathlib import Path

from pipeline.shards import iter_shard_images, split_shard_path


def test_split_shard_path() -> None:
    assert split_shard_path("download/runs/r1/00000.tar#000000007") == (
        Path("download/runs/r1/00000.tar"),
        "000000007",
    )


def test_iter_shard_images_yields_only_wanted_images(tmp_path: Path) -> None:
    tar_path = tmp_path / "00000.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name, data in {
            "000000000.jpg": b"a",
            "000000000.json": b"{}",
            "000000001.webp": b"b",
            "000000002.jpg": b"c",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    got = dict(iter_shard_images(tar_path, {"000000000": 10, "000000001": 11}))

    assert got == {10: b"a", 11: b"b"}
