"""Keyboard-driven human review (spec §4.1 T5, §6): queues of items, append-only decisions, export.

Producers write a queue with `items.write_queue` (or a builder in `builders`); `python -m
pipeline.review serve` shows it at /review; `export` writes the decisions in the `labels` shape with
`source='human'`. Files live in `$DATA_DIR/review/`: `queues/<name>.jsonl`,
`decisions/<name>.jsonl`, `export/<name>.{parquet,csv}`.
"""
