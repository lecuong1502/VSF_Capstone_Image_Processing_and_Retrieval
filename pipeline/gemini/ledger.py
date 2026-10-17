"""Append-only Parquet log of every Gemini call (including cache hits) and its cost."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pipeline.gemini.client import GeminiResult

SCHEMA = pl.Schema(
    {
        "ts": pl.Datetime("us", "UTC"),
        "step": pl.String(),
        "model": pl.String(),
        "mode": pl.String(),
        "cached": pl.Boolean(),
        "ok": pl.Boolean(),
        "input_tokens": pl.Int64(),
        "output_tokens": pl.Int64(),
        "usd": pl.Float64(),
    }
)


class UsageLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame(schema=SCHEMA)
        return pl.read_parquet(self.path)

    def record(
        self, step: str, model: str, results: Sequence[GeminiResult], *, batch: bool
    ) -> None:
        """Append one row per result. Cache hits are logged with zero billed tokens and cost."""
        now = datetime.now(UTC)
        rows = pl.DataFrame(
            {
                "ts": [now] * len(results),
                "step": [step] * len(results),
                "model": [model] * len(results),
                "mode": ["batch" if batch else "online"] * len(results),
                "cached": [r.cached for r in results],
                "ok": [r.ok for r in results],
                "input_tokens": [0 if r.cached else r.input_tokens for r in results],
                "output_tokens": [0 if r.cached else r.output_tokens for r in results],
                "usd": [r.usd for r in results],
            },
            schema=SCHEMA,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        pl.concat([self._read(), rows]).write_parquet(tmp)
        tmp.replace(self.path)

    def spent_usd(self) -> float:
        return float(self._read()["usd"].sum())

    def summary(self) -> pl.DataFrame:
        """Per-step totals: calls, cached_calls, billed tokens and USD."""
        return (
            self._read()
            .group_by("step")
            .agg(
                pl.len().alias("calls"),
                pl.col("cached").sum().alias("cached_calls"),
                pl.col("input_tokens").sum(),
                pl.col("output_tokens").sum(),
                pl.col("usd").sum(),
            )
            .sort("step")
        )
