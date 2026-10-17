"""Budget-capped, cached Gemini client used by every pipeline step that calls the LLM."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline.files import atomic_write_text
from pipeline.gemini.ledger import UsageLedger

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Pre-call estimates used only for the budget check; real usage is logged after each call.
# Input is a heuristic (images are not resized here), output is bounded by max_output_tokens.
CHARS_PER_TOKEN_ESTIMATE = 3  # Vietnamese text tokenizes worse than English
IMAGE_TOKENS_ESTIMATE = 1120  # upper bound per image for Gemini 3.x default media resolution
# Inline Batch API jobs are capped at 20 MB; leave headroom for prompts and JSON overhead.
MAX_BATCH_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class Image:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GeminiRequest:
    """One prompt (+ optional images). Defaults follow spec §4.3/§5.2: temperature 0, minimal
    thinking."""

    prompt: str
    images: tuple[Image, ...] = ()
    schema: type[BaseModel] | None = None
    max_output_tokens: int = 1024
    temperature: float = 0.0
    thinking_level: str = "MINIMAL"

    @property
    def size_bytes(self) -> int:
        return len(self.prompt.encode()) + sum(len(img.data) for img in self.images)


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

    def run_batch(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]:
        """Run one Batch API job and block until it finishes; results in request order."""
        ...


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
    max_batch_bytes: int = MAX_BATCH_BYTES

    def generate(self, step: str, request: GeminiRequest) -> GeminiResult:
        """One online call, served from cache when the identical request was made before."""
        return self._run(step, [request], batch=False)[0]

    def batch(self, step: str, requests: Sequence[GeminiRequest]) -> list[GeminiResult]:
        """Batch API for offline steps. Only cache misses are submitted, split into jobs under
        the inline size limit; each job is logged and cached as soon as it finishes, so an
        interrupted run never loses paid results. Order is preserved."""
        return self._run(step, list(requests), batch=True)

    def _run(self, step: str, requests: list[GeminiRequest], *, batch: bool) -> list[GeminiResult]:
        keys = [self._cache_key(r) for r in requests]
        results: list[GeminiResult | None] = [self._from_cache(k) for k in keys]
        hits = [r for r in results if r is not None]
        if hits:
            self.ledger.record(step, self.model, hits, batch=batch)

        misses = [i for i, r in enumerate(results) if r is None]
        # The lock spans check → call → record so concurrent processes cannot both pass the
        # budget check on the same remaining amount.
        with self.ledger.lock():
            if misses:
                self._check_budget([requests[i] for i in misses], batch=batch)
            for chunk in self._chunks(misses, requests, batch=batch):
                chunk_requests = [requests[i] for i in chunk]
                if batch:
                    log.info("Gemini batch job for %s: %d requests", step, len(chunk))
                    raws = self.transport.run_batch(self.model, chunk_requests)
                else:
                    raws = [self.transport.generate(self.model, chunk_requests[0])]
                fresh = [
                    self._accept(step, keys[i], requests[i], raw, batch=batch)
                    for i, raw in zip(chunk, raws, strict=True)
                ]
                self.ledger.record(step, self.model, fresh, batch=batch)
                for i, result in zip(chunk, fresh, strict=True):
                    results[i] = result

        return [r for r in results if r is not None]

    def _chunks(
        self, misses: list[int], requests: list[GeminiRequest], *, batch: bool
    ) -> Iterator[list[int]]:
        if not batch:
            yield from ([i] for i in misses)
            return
        chunk: list[int] = []
        size = 0
        for i in misses:
            request_bytes = requests[i].size_bytes
            if chunk and size + request_bytes > self.max_batch_bytes:
                yield chunk
                chunk, size = [], 0
            chunk.append(i)
            size += request_bytes
        if chunk:
            yield chunk

    def _accept(
        self, step: str, key: str, request: GeminiRequest, raw: RawResponse, *, batch: bool
    ) -> GeminiResult:
        """Price a fresh response, validate it against the schema, cache it only if usable."""
        usd = self.pricing.cost(raw.input_tokens, raw.output_tokens, batch=batch)
        error = raw.error
        if error is None and request.schema is not None:
            try:
                request.schema.model_validate_json(raw.text)
            except ValidationError as exc:
                error = f"response does not match schema {request.schema.__name__}: {exc}"
        if error is None:
            self._write_cache(key, raw)
        else:
            log.warning("Gemini request failed in %s: %s", step, error)
        return GeminiResult(
            raw.text, raw.input_tokens, raw.output_tokens, usd, cached=False, error=error
        )

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
            "temperature": request.temperature,
            "thinking_level": request.thinking_level,
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
        payload = {
            "text": raw.text,
            "input_tokens": raw.input_tokens,
            "output_tokens": raw.output_tokens,
        }
        atomic_write_text(self._cache_path(key), json.dumps(payload))
