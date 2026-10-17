"""Budget-capped, cached Gemini client used by every pipeline step that calls the LLM."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from pipeline.gemini.ledger import UsageLedger

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Conservative pre-call estimates used only for the budget check (real usage is logged after).
CHARS_PER_TOKEN_ESTIMATE = 3  # Vietnamese text tokenizes worse than English
IMAGE_TOKENS_ESTIMATE = 1120  # upper bound per image for Gemini 3.x default media resolution


@dataclass(frozen=True)
class Image:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GeminiRequest:
    prompt: str
    images: tuple[Image, ...] = ()
    schema: type[BaseModel] | None = None
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class RawResponse:
    """What a transport returns for one request. Output tokens include thinking tokens."""

    text: str
    input_tokens: int
    output_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class GeminiResult:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float
    cached: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def parse(self, schema: type[T]) -> T:
        return schema.model_validate_json(self.text)


@dataclass(frozen=True)
class Pricing:
    """USD per 1M tokens for online calls; batch calls are billed at `batch_discount` of that."""

    input_per_m: float
    output_per_m: float
    batch_discount: float = 0.5

    def cost(self, input_tokens: int, output_tokens: int, *, batch: bool) -> float:
        usd = (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1e6
        return usd * self.batch_discount if batch else usd


class Transport(Protocol):
    """The only part that talks to the Gemini API; faked in tests."""

    def generate(self, model: str, request: GeminiRequest) -> RawResponse: ...

    def run_batch(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]: ...


class BudgetExceededError(RuntimeError):
    def __init__(self, *, spent_usd: float, estimate_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.estimate_usd = estimate_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Gemini budget exceeded: spent ${spent_usd:.4f} + estimated ${estimate_usd:.4f} "
            f"for this call > GEMINI_BUDGET_USD=${budget_usd:.2f}. "
            "Raise GEMINI_BUDGET_USD in .env (after topping up) to continue."
        )


@dataclass
class GeminiClient:
    transport: Transport
    model: str
    pricing: Pricing
    budget_usd: float
    cache_dir: Path
    ledger: UsageLedger

    def generate(self, step: str, request: GeminiRequest) -> GeminiResult:
        """One online call, served from cache when the identical request was made before."""
        return self._run(step, [request], batch=False)[0]

    def batch(self, step: str, requests: Sequence[GeminiRequest]) -> list[GeminiResult]:
        """Batch API job for offline steps; only cache misses are submitted. Order is preserved."""
        return self._run(step, list(requests), batch=True)

    def _run(self, step: str, requests: list[GeminiRequest], *, batch: bool) -> list[GeminiResult]:
        keys = [self._cache_key(r) for r in requests]
        results: list[GeminiResult | None] = [self._from_cache(k) for k in keys]
        misses = [i for i, r in enumerate(results) if r is None]

        if misses:
            self._check_budget([requests[i] for i in misses], batch=batch)
            if batch:
                log.info("Submitting Gemini batch for %s: %d requests", step, len(misses))
                raws = self.transport.run_batch(self.model, [requests[i] for i in misses])
            else:
                raws = [self.transport.generate(self.model, requests[misses[0]])]
            for i, raw in zip(misses, raws, strict=True):
                usd = self.pricing.cost(raw.input_tokens, raw.output_tokens, batch=batch)
                results[i] = GeminiResult(
                    raw.text,
                    raw.input_tokens,
                    raw.output_tokens,
                    usd,
                    cached=False,
                    error=raw.error,
                )
                if raw.error is None:
                    self._write_cache(keys[i], raw)
                else:
                    log.warning("Gemini request failed in %s: %s", step, raw.error)

        final = [r for r in results if r is not None]
        self.ledger.record(step, self.model, final, batch=batch)
        return final

    def _check_budget(self, requests: Sequence[GeminiRequest], *, batch: bool) -> None:
        estimate = sum(
            self.pricing.cost(
                math.ceil(len(r.prompt) / CHARS_PER_TOKEN_ESTIMATE)
                + IMAGE_TOKENS_ESTIMATE * len(r.images),
                r.max_output_tokens,
                batch=batch,
            )
            for r in requests
        )
        spent = self.ledger.spent_usd()
        if spent + estimate > self.budget_usd:
            raise BudgetExceededError(
                spent_usd=spent, estimate_usd=estimate, budget_usd=self.budget_usd
            )

    def _cache_key(self, request: GeminiRequest) -> str:
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "images": [
                [hashlib.sha256(img.data).hexdigest(), img.mime_type] for img in request.images
            ],
            "schema": request.schema.model_json_schema() if request.schema else None,
            "max_output_tokens": request.max_output_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _from_cache(self, key: str) -> GeminiResult | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        hit = json.loads(path.read_text())
        return GeminiResult(
            hit["text"], hit["input_tokens"], hit["output_tokens"], usd=0.0, cached=True
        )

    def _write_cache(self, key: str, raw: RawResponse) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "text": raw.text,
                    "input_tokens": raw.input_tokens,
                    "output_tokens": raw.output_tokens,
                }
            )
        )
        tmp.replace(path)
