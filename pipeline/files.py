"""Small filesystem helpers shared by pipeline steps."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write via a unique temp file in the same directory, then rename over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(text))


class LockedError(RuntimeError):
    """Another pipeline step holds the lock."""


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Non-blocking cross-process lock: raises LockedError if another process holds `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockedError(
                f"{path} is locked by another pipeline step (e.g. make download / make dedup); "
                "wait for it to finish"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
