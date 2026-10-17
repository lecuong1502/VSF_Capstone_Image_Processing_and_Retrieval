"""GeminiClient behaviour against a fake transport: caching, cost accounting, budget stop, batch."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from pipeline.gemini import (
    BudgetExceededError,
    GeminiClient,
    GeminiRequest,
    Image,
    Pricing,
    RawResponse,
    UsageLedger,
)

PRICING = Pricing(input_per_m=0.30, output_per_m=2.50, batch_discount=0.5)


class FakeTransport:
    """Records calls and answers with fixed token counts."""

    def __init__(self, text: str = '{"label": "ok"}', input_tokens: int = 1000) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.online_calls: list[GeminiRequest] = []
        self.batches: list[list[GeminiRequest]] = []

    def _answer(self, request: GeminiRequest) -> RawResponse:
        return RawResponse(text=self.text, input_tokens=self.input_tokens, output_tokens=100)

    def generate(self, model: str, request: GeminiRequest) -> RawResponse:
        self.online_calls.append(request)
        return self._answer(request)

    def run_batch(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]:
        self.batches.append(list(requests))
        return [self._answer(r) for r in requests]


def make_client(tmp_path: Path, transport: FakeTransport, budget_usd: float = 5.0) -> GeminiClient:
    return GeminiClient(
        transport=transport,
        model="gemini-test",
        pricing=PRICING,
        budget_usd=budget_usd,
        cache_dir=tmp_path / "cache",
        ledger=UsageLedger(tmp_path / "gemini_usage.parquet"),
    )


# 1000 input tokens * $0.30/1M + 100 output tokens * $2.50/1M
ONLINE_COST = 1000 * 0.30 / 1e6 + 100 * 2.50 / 1e6


def test_generate_returns_text_and_logs_cost(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = make_client(tmp_path, transport)

    result = client.generate("labels", GeminiRequest(prompt="hello"))

    assert result.text == '{"label": "ok"}'
    assert not result.cached
    assert result.usd == pytest.approx(ONLINE_COST)
    assert client.ledger.spent_usd() == pytest.approx(ONLINE_COST)
    summary = client.ledger.summary()
    assert summary["step"].to_list() == ["labels"]
    assert summary["input_tokens"].to_list() == [1000]
    assert summary["output_tokens"].to_list() == [100]


def test_identical_request_is_served_from_cache_for_free(tmp_path: Path) -> None:
    transport = FakeTransport()
    request = GeminiRequest(prompt="describe", images=(Image(b"\x89PNG", "image/png"),))

    first = make_client(tmp_path, transport).generate("captions", request)
    # A fresh client (e.g. a pipeline re-run) shares the on-disk cache.
    second = make_client(tmp_path, transport).generate("captions", request)

    assert len(transport.online_calls) == 1
    assert second.cached and second.usd == 0
    assert second.text == first.text
    assert second.input_tokens == first.input_tokens


def test_cache_key_covers_image_bytes_and_model(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = make_client(tmp_path, transport)

    client.generate("s", GeminiRequest(prompt="p", images=(Image(b"a", "image/png"),)))
    client.generate("s", GeminiRequest(prompt="p", images=(Image(b"b", "image/png"),)))
    other_model = GeminiClient(
        transport=transport,
        model="gemini-other",
        pricing=PRICING,
        budget_usd=5.0,
        cache_dir=tmp_path / "cache",
        ledger=client.ledger,
    )
    other_model.generate("s", GeminiRequest(prompt="p", images=(Image(b"a", "image/png"),)))

    assert len(transport.online_calls) == 3


def test_budget_stop_raises_before_spending(tmp_path: Path) -> None:
    transport = FakeTransport(input_tokens=1_000_000)  # $0.30 + output per call
    client = make_client(tmp_path, transport, budget_usd=0.50)
    client.generate("labels", GeminiRequest(prompt="first"))
    spent = client.ledger.spent_usd()

    with pytest.raises(BudgetExceededError, match="GEMINI_BUDGET_USD") as excinfo:
        client.generate("labels", GeminiRequest(prompt="x" * 2_000_000))

    assert len(transport.online_calls) == 1
    assert client.ledger.spent_usd() == spent
    assert excinfo.value.spent_usd == pytest.approx(spent)
    assert excinfo.value.budget_usd == 0.50


def test_spend_persists_across_clients(tmp_path: Path) -> None:
    make_client(tmp_path, FakeTransport()).generate("a", GeminiRequest(prompt="1"))
    make_client(tmp_path, FakeTransport()).generate("b", GeminiRequest(prompt="2"))

    ledger = UsageLedger(tmp_path / "gemini_usage.parquet")
    assert ledger.spent_usd() == pytest.approx(2 * ONLINE_COST)
    assert sorted(ledger.summary()["step"].to_list()) == ["a", "b"]


def test_cache_hits_are_logged_but_cost_nothing(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeTransport())
    client.generate("labels", GeminiRequest(prompt="same"))
    client.generate("labels", GeminiRequest(prompt="same"))

    row = client.ledger.summary().row(0, named=True)
    assert row["calls"] == 2
    assert row["cached_calls"] == 1
    assert row["usd"] == pytest.approx(ONLINE_COST)


class Label(BaseModel):
    label: str


def test_structured_output_parses_into_schema(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeTransport(text='{"label": "nsfw"}'))

    result = client.generate("labels", GeminiRequest(prompt="classify", schema=Label))

    assert result.parse(Label) == Label(label="nsfw")


def test_batch_submits_only_cache_misses_at_batch_price(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = make_client(tmp_path, transport)
    client.generate("captions", GeminiRequest(prompt="already cached"))

    requests = [GeminiRequest(prompt=p) for p in ("already cached", "new 1", "new 2")]
    results = client.batch("captions", requests)

    assert [len(b) for b in transport.batches] == [2]
    assert [r.cached for r in results] == [True, False, False]
    assert results[1].usd == pytest.approx(ONLINE_COST * 0.5)
    # Batch results are cached too: re-running the job submits nothing.
    client.batch("captions", requests)
    assert len(transport.batches) == 1


def test_batch_budget_is_checked_for_the_whole_job(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = make_client(tmp_path, transport, budget_usd=0.001)

    with pytest.raises(BudgetExceededError):
        client.batch("captions", [GeminiRequest(prompt=str(i)) for i in range(100)])

    assert transport.batches == []
