"""`python -m pipeline.review` – build queues, serve /review, export decisions, report throughput.

Usage: python -m pipeline.review queue relevance [--n 300] [--blind] [--name relevance-audit]
       python -m pipeline.review queue moderation [--n 500]
       python -m pipeline.review queue kg --edges kg.sqlite [--n 100]
       python -m pipeline.review queue qrels --qrels qrels.tsv --queries q.jsonl [--fraction 0.2]
       python -m pipeline.review serve [--port 8765]
       python -m pipeline.review export [QUEUE ...]
       python -m pipeline.review stats
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import polars as pl

from pipeline import config
from pipeline.review import builders
from pipeline.review.items import Item, list_queues, queue_path, write_queue
from pipeline.review.server import make_server
from pipeline.review.store import export_queue, queue_stats

log = logging.getLogger("pipeline.review")

WEB_DIST = config.REPO_ROOT / "apps" / "web" / "dist"


def review_dir(data_dir: Path) -> Path:
    return data_dir / "review"


def _build(args: argparse.Namespace, data_dir: Path) -> list[Item]:
    if args.mode == "relevance":
        return builders.relevance_queue(data_dir, n=args.n, seed=args.seed, blind=args.blind)
    if args.mode == "moderation":
        return builders.moderation_queue(data_dir, n=args.n, seed=args.seed)
    if args.mode == "kg":
        return builders.kg_queue(data_dir, args.edges, n=args.n, seed=args.seed)
    return builders.qrels_queue(args.qrels, args.queries, fraction=args.fraction, seed=args.seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.review")
    sub = parser.add_subparsers(dest="command", required=True)

    queue = sub.add_parser("queue", help="build a review queue")
    modes = queue.add_subparsers(dest="mode", required=True)
    for mode, n in (("relevance", 300), ("moderation", 500), ("kg", 100)):
        p = modes.add_parser(mode)
        p.add_argument("--n", type=int, default=n)
    modes.choices["relevance"].add_argument(
        "--blind", action="store_true", help="hide the existing verify label (for kappa)"
    )
    modes.choices["kg"].add_argument("--edges", type=Path, required=True)
    qrels = modes.add_parser("qrels")
    qrels.add_argument("--qrels", type=Path, required=True)
    qrels.add_argument("--queries", type=Path, required=True)
    qrels.add_argument("--fraction", type=float, default=0.2)
    for p in modes.choices.values():
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--name", help="queue name (default: the mode)")
        p.add_argument("--force", action="store_true", help="replace an existing queue")

    serve = sub.add_parser("serve", help="serve /review on localhost")
    serve.add_argument("--port", type=int, default=8765)
    export = sub.add_parser("export", help="write decisions in the labels shape")
    export.add_argument("queues", nargs="*")
    sub.add_parser("stats", help="progress and items/hour per queue")
    args = parser.parse_args()

    data_dir = config.data_dir()
    reviews = review_dir(data_dir)
    if args.command == "queue":
        name = args.name or args.mode
        if queue_path(reviews, name).exists() and not args.force:
            parser.error(f"queue {name} exists (its decisions are kept); pass --force to rebuild")
        items = _build(args, data_dir)
        log.info("Wrote %d items to %s", len(items), write_queue(reviews, name, items))
    elif args.command == "serve":
        if not (WEB_DIST / "index.html").exists():
            log.warning("No web build at %s – run: make build-web", WEB_DIST)
        annotator = os.environ.get("REVIEW_ANNOTATOR") or "human"
        server = make_server(data_dir, reviews, WEB_DIST, annotator=annotator, port=args.port)
        log.info("Review tool at http://127.0.0.1:%d/review (Ctrl+C to stop)", args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "export":
        for name in args.queues or list_queues(reviews):
            frame = export_queue(reviews, name)
            log.info("%s: %d label rows → %s", name, frame.height, reviews / "export")
    else:
        rows = [queue_stats(reviews, name) for name in list_queues(reviews)]
        with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, float_precision=0):
            log.info("%s", pl.DataFrame(rows))


if __name__ == "__main__":
    main()
