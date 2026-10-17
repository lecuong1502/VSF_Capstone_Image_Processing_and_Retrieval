"""`Transport` implementation on top of the google-genai SDK, and the env-configured client."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence

from google import genai
from google.genai import types

from pipeline.gemini.client import GeminiClient, GeminiRequest, RawResponse
from pipeline.gemini.ledger import UsageLedger
from pipeline.gemini.settings import GeminiSettings

log = logging.getLogger(__name__)

TERMINAL_STATES = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_EXPIRED,
}
DONE_STATES = {types.JobState.JOB_STATE_SUCCEEDED, types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED}
# Inline batch requests are capped at 20 MB per job; leave headroom for prompts and JSON overhead.
MAX_INLINE_BATCH_BYTES = 15 * 1024 * 1024


class GenAITransport:
    def __init__(
        self, api_key: str, *, thinking_level: str = "MINIMAL", poll_interval_s: float = 30
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set – add it to .env")
        self._client = genai.Client(api_key=api_key)
        self._thinking_level = types.ThinkingLevel(thinking_level)
        self._poll_interval_s = poll_interval_s

    def generate(self, model: str, request: GeminiRequest) -> RawResponse:
        response = self._client.models.generate_content(
            model=model, contents=self._contents(request), config=self._config(request)
        )
        return _to_raw(response)

    def run_batch(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]:
        results: list[RawResponse] = []
        for chunk in _chunks_by_size(requests, MAX_INLINE_BATCH_BYTES):
            results.extend(self._run_one_job(model, chunk))
        return results

    def _run_one_job(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]:
        job = self._client.batches.create(
            model=model,
            src=[
                types.InlinedRequest(contents=self._contents(r), config=self._config(r))
                for r in requests
            ],
        )
        assert job.name is not None
        log.info("Gemini batch job %s submitted (%d requests)", job.name, len(requests))
        while job.state not in TERMINAL_STATES:
            time.sleep(self._poll_interval_s)
            job = self._client.batches.get(name=job.name)
        if job.state not in DONE_STATES:
            message = f"Gemini batch job {job.name} ended in {job.state}: {job.error}"
            return [RawResponse("", 0, 0, error=message) for _ in requests]

        inlined = job.dest.inlined_responses if job.dest else None
        if not inlined or len(inlined) != len(requests):
            raise RuntimeError(f"Gemini batch job {job.name} returned an unexpected result count")
        return [
            _to_raw(item.response)
            if item.response is not None
            else RawResponse("", 0, 0, error=str(item.error))
            for item in inlined
        ]

    @staticmethod
    def _contents(request: GeminiRequest) -> list[types.Part]:
        parts = [
            types.Part.from_bytes(data=img.data, mime_type=img.mime_type) for img in request.images
        ]
        return [*parts, types.Part.from_text(text=request.prompt)]

    def _config(self, request: GeminiRequest) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            max_output_tokens=request.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self._thinking_level),
        )
        if request.schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = request.schema
        return config


def _to_raw(response: types.GenerateContentResponse) -> RawResponse:
    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (
        ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
    )
    text = response.text
    if text is None:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        return RawResponse("", input_tokens, output_tokens, error=f"empty response ({reason})")
    return RawResponse(text, input_tokens, output_tokens)


def _chunks_by_size(
    requests: Sequence[GeminiRequest], max_bytes: int
) -> Iterator[Sequence[GeminiRequest]]:
    start, size = 0, 0
    for i, request in enumerate(requests):
        request_bytes = len(request.prompt.encode()) + sum(len(img.data) for img in request.images)
        if i > start and size + request_bytes > max_bytes:
            yield requests[start:i]
            start, size = i, 0
        size += request_bytes
    if start < len(requests):
        yield requests[start:]


def client_from_env(settings: GeminiSettings | None = None) -> GeminiClient:
    """The client every pipeline step should use: real API, budget, cache and ledger from .env."""
    settings = settings or GeminiSettings.from_env()
    return GeminiClient(
        transport=GenAITransport(settings.api_key, thinking_level=settings.thinking_level),
        model=settings.model,
        pricing=settings.pricing,
        budget_usd=settings.budget_usd,
        cache_dir=settings.cache_dir,
        ledger=UsageLedger(settings.usage_log),
    )
