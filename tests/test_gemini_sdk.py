"""GenAITransport against a fake google-genai client (no network)."""

from types import SimpleNamespace
from typing import Any

from google.genai import types
from pydantic import BaseModel

from pipeline.gemini import GeminiRequest, Image
from pipeline.gemini.sdk import GenAITransport, _config


def _response(text: str, reason: types.FinishReason = types.FinishReason.STOP) -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                finish_reason=reason,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
        ),
    )


class FakeModels:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeBatches:
    def __init__(self, states: list[types.JobState], responses: list[Any]) -> None:
        self.states = states
        self.responses = responses

    def _job(self) -> Any:
        state = self.states.pop(0)
        inlined = [types.InlinedResponse(response=r) for r in self.responses]
        return types.BatchJob(
            name="batches/1",
            state=state,
            dest=types.BatchJobDestination(inlined_responses=inlined),
        )

    def create(self, **kwargs: Any) -> Any:
        return self._job()

    def get(self, name: str) -> Any:
        return self._job()


def _transport(models: Any = None, batches: Any = None) -> GenAITransport:
    client = SimpleNamespace(models=models, batches=batches)
    return GenAITransport(client=client, poll_interval_s=0)


def test_generate_counts_thinking_tokens_as_output() -> None:
    raw = _transport(models=FakeModels(_response("hi"))).generate("m", GeminiRequest(prompt="p"))

    assert (raw.text, raw.input_tokens, raw.output_tokens, raw.error) == ("hi", 100, 25, None)


def test_truncated_response_is_an_error() -> None:
    response = _response('{"a": ', types.FinishReason.MAX_TOKENS)

    raw = _transport(models=FakeModels(response)).generate("m", GeminiRequest(prompt="p"))

    assert raw.error is not None and "MAX_TOKENS" in raw.error
    assert raw.input_tokens == 100  # still billed


class Answer(BaseModel):
    city: str


def test_config_carries_schema_temperature_and_thinking() -> None:
    request = GeminiRequest(
        prompt="p", images=(Image(b"x", "image/webp"),), schema=Answer, temperature=0.2
    )

    config = _config(request)

    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer
    assert config.temperature == 0.2
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_batch_polls_until_done_and_keeps_order() -> None:
    batches = FakeBatches(
        states=[types.JobState.JOB_STATE_PENDING, types.JobState.JOB_STATE_SUCCEEDED],
        responses=[_response("a"), _response("b")],
    )

    raws = _transport(batches=batches).run_batch("m", [GeminiRequest("1"), GeminiRequest("2")])

    assert [r.text for r in raws] == ["a", "b"]


def test_failed_batch_returns_errors_for_every_request() -> None:
    batches = FakeBatches(states=[types.JobState.JOB_STATE_FAILED], responses=[])

    raws = _transport(batches=batches).run_batch("m", [GeminiRequest("1"), GeminiRequest("2")])

    assert all(r.error and "JOB_STATE_FAILED" in r.error for r in raws)
