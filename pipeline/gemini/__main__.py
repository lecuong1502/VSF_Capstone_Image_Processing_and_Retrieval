"""`python -m pipeline.gemini [cost|smoke]` – spend report per step, or one real smoke call."""

from __future__ import annotations

import argparse
import logging

import polars as pl
from pydantic import BaseModel

from pipeline.gemini.client import GeminiRequest
from pipeline.gemini.ledger import UsageLedger
from pipeline.gemini.settings import GeminiSettings

log = logging.getLogger("pipeline.gemini")


class SmokeAnswer(BaseModel):
    city: str
    country: str


def report_cost(settings: GeminiSettings) -> None:
    ledger = UsageLedger(settings.usage_log)
    summary = ledger.summary()
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, float_precision=6):
        log.info("Gemini usage (%s):\n%s", settings.usage_log, summary)
    spent = ledger.spent_usd()
    log.info(
        "Total: $%.6f of GEMINI_BUDGET_USD=$%.2f (%.1f%%)",
        spent,
        settings.budget_usd,
        100 * spent / settings.budget_usd if settings.budget_usd else 0,
    )


def smoke(settings: GeminiSettings) -> None:
    from pipeline.gemini.sdk import client_from_env

    client = client_from_env(settings)
    request = GeminiRequest(
        prompt="Vịnh Hạ Long thuộc thành phố nào, quốc gia nào? Trả lời bằng JSON.",
        schema=SmokeAnswer,
        max_output_tokens=128,
    )
    result = client.generate("smoke", request)
    log.info(
        "Smoke call: %s | in=%d out=%d tokens | $%.6f | cached=%s",
        result.parse(SmokeAnswer) if result.ok else result.error,
        result.input_tokens,
        result.output_tokens,
        result.usd,
        result.cached,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m pipeline.gemini")
    parser.add_argument("command", nargs="?", choices=["cost", "smoke"], default="cost")
    args = parser.parse_args()
    settings = GeminiSettings.from_env()
    if args.command == "smoke":
        smoke(settings)
    report_cost(settings)


if __name__ == "__main__":
    main()
