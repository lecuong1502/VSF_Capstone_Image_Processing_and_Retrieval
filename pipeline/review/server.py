"""Local review server (stdlib, localhost only): JSON API, thumbnails and the built web app.

GET  /api/review/queues                       -> [{name, total, decided}]
GET  /api/review/queues/<name>                -> {items (+ image_url, reference_urls), decisions}
POST /api/review/queues/<name>/decisions      <- {item_id, values, elapsed_ms}
GET  /thumbs/<image_id>.webp                  -> thumbnail from image_assets.thumb_path
GET  anything else                            -> file from the web build, or its index.html
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import polars as pl
from pydantic import BaseModel, ValidationError

from pipeline.download import assets_path
from pipeline.review.items import Item, list_queues, read_queue
from pipeline.review.store import MAX_ITEM_MS, Decision, DecisionStore

log = logging.getLogger(__name__)

_QUEUE = re.compile(r"^/api/review/queues/([^/]+)$")
_DECISIONS = re.compile(r"^/api/review/queues/([^/]+)/decisions$")
_THUMB = re.compile(r"^/thumbs/(\d+)\.webp$")
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json",
}
MAX_BODY = 64 * 1024


class DecisionIn(BaseModel):
    item_id: str
    values: dict[str, str]
    elapsed_ms: int


class ReviewApp:
    """State shared by request handlers: thumbnail paths, cached queues, one store per queue."""

    def __init__(self, data_dir: Path, review_dir: Path, static_dir: Path, annotator: str) -> None:
        self.data_dir = data_dir.resolve()
        self.review_dir = review_dir
        self.static_dir = static_dir.resolve()
        self.annotator = annotator
        assets = pl.read_parquet(assets_path(data_dir)).filter(pl.col("thumb_path").is_not_null())
        self.thumbs: dict[int, str] = dict(assets.select("image_id", "thumb_path").iter_rows())
        self._queues: dict[str, tuple[float, dict[str, Item]]] = {}
        self._stores: dict[str, DecisionStore] = {}
        self._lock = threading.Lock()

    def items(self, name: str) -> dict[str, Item] | None:
        """Queue items by id, re-read when the queue file changes; None if there is no queue."""
        if name not in list_queues(self.review_dir):
            return None
        path = self.review_dir / "queues" / f"{name}.jsonl"
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._queues.get(name)
            if cached is None or cached[0] != mtime:
                cached = (mtime, {i.item_id: i for i in read_queue(self.review_dir, name)})
                self._queues[name] = cached
            return cached[1]

    def store(self, name: str) -> DecisionStore:
        with self._lock:
            return self._stores.setdefault(name, DecisionStore(self.review_dir, name))

    def thumb_url(self, image_id: int | None) -> str | None:
        return f"/thumbs/{image_id}.webp" if image_id in self.thumbs else None

    def queue_payload(self, name: str, items: dict[str, Item]) -> dict[str, Any]:
        out = []
        for item in items.values():
            row = item.model_dump()
            row["image_url"] = self.thumb_url(item.image_id)
            row["reference_urls"] = [
                url for url in map(self.thumb_url, item.references) if url is not None
            ]
            out.append(row)
        decisions = {k: d.values for k, d in self.store(name).latest().items() if k in items}
        return {"name": name, "annotator": self.annotator, "items": out, "decisions": decisions}


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: Any) -> None:
        log.debug(format, *args)

    @property
    def app(self) -> ReviewApp:
        return self.server.app

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Thumbnails and hashed build assets are immutable; API data and index.html are not.
        fresh = "json" in content_type or "html" in content_type
        self.send_header("Cache-Control", "no-cache" if fresh else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/review/queues":
            rows = []
            for name in list_queues(self.app.review_dir):
                items = self.app.items(name) or {}
                decided = sum(1 for k in self.app.store(name).latest() if k in items)
                rows.append({"name": name, "total": len(items), "decided": decided})
            return self._json(rows)
        if match := _QUEUE.match(path):
            queue = self.app.items(match[1])
            if queue is None:
                return self._error(HTTPStatus.NOT_FOUND, "no such queue")
            return self._json(self.app.queue_payload(match[1], queue))
        if path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        if path.startswith("/thumbs/"):
            return self._thumb(path)
        return self._static(path)

    def _thumb(self, path: str) -> None:
        match = _THUMB.match(path)
        relative = self.app.thumbs.get(int(match[1])) if match else None
        file = (self.app.data_dir / relative).resolve() if relative else None
        if file is None or not file.is_relative_to(self.app.data_dir) or not file.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "no thumbnail")
        self._send(HTTPStatus.OK, file.read_bytes(), "image/webp")

    def _static(self, path: str) -> None:
        root = self.app.static_dir
        file = (root / unquote(path).lstrip("/")).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            file = root / "index.html"  # client-side routes such as /review
        if not file.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "web app not built – run: make build-web")
        content_type = _TYPES.get(file.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, file.read_bytes(), content_type)

    def do_POST(self) -> None:
        match = _DECISIONS.match(urlsplit(self.path).path)
        items = self.app.items(match[1]) if match else None
        if match is None or items is None:
            return self._error(HTTPStatus.NOT_FOUND, "no such queue")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
        try:
            body = DecisionIn.model_validate_json(self.rfile.read(length))
        except ValidationError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        decision = Decision(
            item_id=body.item_id,
            values=body.values,
            annotator=self.app.annotator,
            decided_at=datetime.now(UTC),
            elapsed_ms=max(0, min(body.elapsed_ms, MAX_ITEM_MS)),
        )
        try:
            self.app.store(match[1]).record(items, decision)
        except KeyError:
            return self._error(HTTPStatus.NOT_FOUND, "no such item")
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        self._json({"ok": True})


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: ReviewApp) -> None:
        self.app = app
        super().__init__(address, Handler)


def make_server(
    data_dir: Path,
    review_dir: Path,
    static_dir: Path,
    *,
    annotator: str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ReviewServer:
    return ReviewServer((host, port), ReviewApp(data_dir, review_dir, static_dir, annotator))
