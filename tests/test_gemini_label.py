"""Gemini moderation labels: stratified sample, requests, results, audit queue and kappa."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pipeline.gemini import GeminiClient, GeminiRequest, Pricing, RawResponse, UsageLedger
from pipeline.gemini_label import (
    CRITERIA,
    STEP,
    ModerationLabel,
    audit_ids_path,
    audit_sample,
    build_request,
    cohen_kappa,
    kappa_table,
    labels_path,
    run_labeling,
    sample_images,
    write_audit_queue,
)
from pipeline.review.items import read_queue
from pipeline.review.store import Decision, DecisionStore

PRICING = Pricing(input_per_m=0.30, output_per_m=2.50, batch_discount=0.5)


def _answer(**overrides: object) -> str:
    label = {
        "watermark": False,
        "text_overlay": False,
        "nsfw": False,
        "political": False,
        "violence": False,
        "relevance": "yes",
        "shot_type": "PHONG_CANH",
        "confidence": 0.9,
        "reason": "ảnh chụp toàn cảnh",
    }
    return json.dumps(label | overrides)


class FakeTransport:
    """Answers per image from `answers` (keyed by the image bytes), else a default label."""

    def __init__(self, answers: dict[bytes, str] | None = None) -> None:
        self.answers = answers or {}
        self.batches: list[list[GeminiRequest]] = []

    def _raw(self, request: GeminiRequest) -> RawResponse:
        text = self.answers.get(request.images[0].data, _answer())
        if text == "TRUNCATED":
            return RawResponse('{"watermark": tr', 300, 256, error="finish_reason=MAX_TOKENS")
        return RawResponse(text, 300, 80)

    def generate(self, model: str, request: GeminiRequest) -> RawResponse:
        return self._raw(request)

    def run_batch(self, model: str, requests: Sequence[GeminiRequest]) -> list[RawResponse]:
        self.batches.append(list(requests))
        return [self._raw(r) for r in requests]


def _client(tmp_path: Path, transport: FakeTransport, budget: float = 5.0) -> GeminiClient:
    return GeminiClient(
        transport=transport,
        model="gemini-test",
        pricing=PRICING,
        budget_usd=budget,
        cache_dir=tmp_path / "cache",
        ledger=UsageLedger(tmp_path / "usage.parquet"),
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """40 images: entities e0..e3 in branches A/A/B/B; image i is `yes` unless i % 4 == 0."""
    data = tmp_path / "data"
    (data / "ingest").mkdir(parents=True)
    (data / "download").mkdir()
    (data / "thumbs").mkdir()
    ids = list(range(1, 41))
    pl.DataFrame(
        {
            "entity_id": ["e0", "e1", "e2", "e3"],
            "entity_name": ["Vịnh Hạ Long", "Chùa Một Cột", "Phở bò", "Nón lá"],
            "leaf_label": ["Vịnh", "Chùa", "Món ăn", "Thủ công"],
            "branch": ["A", "A", "B", "B"],
            "province": [None] * 4,
        }
    ).write_parquet(data / "ingest" / "entities.parquet")
    pl.DataFrame({"image_id": ids, "page_title": ["t"] * 40}).write_parquet(
        data / "ingest" / "images.parquet"
    )
    pl.DataFrame(
        {
            "image_id": ids,
            "entity_id": [f"e{i % 4}" for i in ids],
            "criterion": ["relevance"] * 40,
            "value": ["no" if i % 4 == 0 else "yes" for i in ids],
            "label_conflict": [False] * 40,
            "query_type": ["CHINH"] * 40,
        }
    ).write_parquet(data / "ingest" / "labels.parquet")
    for i in ids:
        (data / "thumbs" / f"{i}.webp").write_bytes(f"img{i}".encode())
    pl.DataFrame(
        {
            "image_id": ids,
            "download_status": ["ok"] * 40,
            "thumb_path": [f"thumbs/{i}.webp" for i in ids],
        }
    ).write_parquet(data / "download" / "image_assets.parquet")
    return data


def test_sample_is_stratified_by_branch_and_verify_and_deterministic(data_dir: Path) -> None:
    sample = sample_images(data_dir, n=20, seed=0)

    assert sample.height == 20
    assert sample["image_id"].n_unique() == 20
    assert sample.equals(sample_images(data_dir, n=20, seed=0))
    # proportional: branch A (e0, e1) and B (e2, e3) are 20 images each; e0 images are all `no`
    counts = {(b, v): c for b, v, c in sample.group_by("branch", "verify").len().iter_rows()}
    assert counts == {("A", "no"): 5, ("A", "yes"): 5, ("B", "yes"): 10}
    assert set(sample.columns) >= {"image_id", "entity_id", "entity_name", "leaf_label"}


def test_sample_takes_everything_when_n_is_large(data_dir: Path) -> None:
    assert sample_images(data_dir, n=10_000, seed=0).height == 40


def test_request_follows_the_spec_prompt_and_schema(data_dir: Path) -> None:
    row = sample_images(data_dir, n=40, seed=0).row(0, named=True)
    request = build_request(row, b"webp-bytes")

    assert request.schema is ModerationLabel
    assert request.images[0].mime_type == "image/webp"
    assert f'"{row["entity_name"]}"' in request.prompt
    assert row["leaf_label"] in request.prompt
    assert request.temperature == 0
    assert request.thinking_level == "MINIMAL"


def test_schema_rejects_out_of_enum_and_cuts_long_reasons() -> None:
    with pytest.raises(ValueError):
        ModerationLabel.model_validate_json(_answer(shot_type="SELFIE"))
    long = ModerationLabel.model_validate_json(_answer(reason="x" * 200))
    assert long.reason == "x" * 120  # cut, not rejected
    assert ModerationLabel.model_json_schema()["properties"]["reason"]["maxLength"] == 120


def test_run_labels_writes_results_and_cost(data_dir: Path, tmp_path: Path) -> None:
    transport = FakeTransport(
        {b"img3": _answer(watermark=True, relevance="no"), b"img5": "TRUNCATED"}
    )
    summary = run_labeling(data_dir, _client(tmp_path, transport), n=40, seed=0)

    assert (summary.requested, summary.labeled, summary.failed) == (40, 39, 1)
    assert summary.input_tokens_per_image == pytest.approx(300)
    # batch price: (300 * 0.30 + 80 * 2.50) / 1e6 / 2 per labeled image; failures are paid too
    assert summary.usd == pytest.approx(39 * 0.000145 + (300 * 0.30 + 256 * 2.50) / 2e6)
    assert summary.usd_per_image == pytest.approx(summary.usd / 40)
    assert len(transport.batches) == 1

    labels = pl.read_parquet(labels_path(data_dir))
    assert labels.height == 40
    three = labels.filter(pl.col("image_id") == 3).row(0, named=True)
    assert (three["watermark"], three["relevance"], three["error"]) == (True, "no", None)
    five = labels.filter(pl.col("image_id") == 5).row(0, named=True)
    assert five["error"].startswith("finish_reason") and five["watermark"] is None
    assert three["entity_id"] == "e3"


def test_pilot_then_full_run_pays_only_once(data_dir: Path, tmp_path: Path) -> None:
    transport = FakeTransport()
    client = _client(tmp_path, transport)

    pilot = run_labeling(data_dir, client, n=40, seed=0, limit=10)
    full = run_labeling(data_dir, client, n=40, seed=0)

    assert pilot.requested == 10
    assert [len(b) for b in transport.batches] == [10, 30]  # the pilot's 10 come from cache
    assert full.labeled == 40
    assert full.usd_per_image == pytest.approx(pilot.usd_per_image)  # projected from tokens
    ledger = pl.read_parquet(tmp_path / "usage.parquet")
    assert set(ledger["step"]) == {STEP}


def test_failures_are_not_repaid_unless_retried(data_dir: Path, tmp_path: Path) -> None:
    transport = FakeTransport({b"img5": "TRUNCATED"})
    client = _client(tmp_path, transport)
    run_labeling(data_dir, client, n=40, seed=0)

    again = run_labeling(data_dir, client, n=40, seed=0)
    assert again.skipped_failed == 1 and again.usd == 0  # the rest are cache hits
    assert len(transport.batches) == 1

    retried = run_labeling(data_dir, client, n=40, seed=0, retry_failed=True)
    assert [len(b) for b in transport.batches] == [40, 1]
    assert retried.failed == 1


def test_a_small_run_after_the_full_run_keeps_every_label(data_dir: Path, tmp_path: Path) -> None:
    client = _client(tmp_path, FakeTransport())
    run_labeling(data_dir, client, n=40, seed=0)
    run_labeling(data_dir, client, n=40, seed=0, limit=5)

    assert pl.read_parquet(labels_path(data_dir)).height == 40


def test_sample_keeps_entities_without_a_branch(data_dir: Path) -> None:
    entities = data_dir / "ingest" / "entities.parquet"
    pl.read_parquet(entities).with_columns(branch=pl.lit(None, pl.String)).write_parquet(entities)

    assert sample_images(data_dir, n=10, seed=0).height == 10


def test_long_format_labels_for_training(data_dir: Path, tmp_path: Path) -> None:
    run_labeling(data_dir, _client(tmp_path, FakeTransport()), n=4, seed=0)

    long = pl.read_parquet(labels_path(data_dir).with_name("labels_gemini.parquet"))
    assert long.height == 4 * len(CRITERIA)
    assert set(long["source"]) == {"gemini"}
    assert set(long["criterion"]) == set(CRITERIA)
    assert set(long.filter(pl.col("criterion") == "watermark")["value"]) == {"no"}


def test_audit_sample_enriches_rare_positives(data_dir: Path, tmp_path: Path) -> None:
    answers = {b"img7": _answer(nsfw=True), b"img8": _answer(political=True)}
    run_labeling(data_dir, _client(tmp_path, FakeTransport(answers)), n=40, seed=0)
    labels = pl.read_parquet(labels_path(data_dir))

    audit = audit_sample(labels, n=10, seed=0, per_positive=5)

    ids = audit["image_id"].to_list()
    assert len(ids) == len(set(ids)) == 10
    strata = dict(audit.iter_rows())
    assert (strata[7], strata[8]) == ("positive:nsfw", "positive:political")
    assert sum(v == "random" for v in strata.values()) == 8


def test_audit_gives_every_rare_criterion_a_share(data_dir: Path, tmp_path: Path) -> None:
    # 20 watermark positives would fill a small audit on their own
    answers = {f"img{i}".encode(): _answer(watermark=True) for i in range(1, 21)}
    answers[b"img30"] = _answer(violence=True)
    answers[b"img31"] = _answer(relevance="unsure")  # not a relevance positive
    run_labeling(data_dir, _client(tmp_path, FakeTransport(answers)), n=40, seed=0)
    labels = pl.read_parquet(labels_path(data_dir))

    strata = dict(audit_sample(labels, n=12, seed=0, per_positive=40).iter_rows())

    assert sum(v == "positive:watermark" for v in strata.values()) == 2  # 12 // 6 criteria
    assert strata[30] == "positive:violence"
    assert strata.get(31) != "positive:relevance"


def test_cohen_kappa() -> None:
    assert cohen_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == pytest.approx(1.0)
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)
    # 82% agreement, marginals 151/300 vs 191/300 → kappa ≈ 0.64 (the #12 relevance audit)
    a = ["y"] * 144 + ["y"] * 7 + ["n"] * 47 + ["n"] * 102
    b = ["y"] * 144 + ["n"] * 7 + ["y"] * 47 + ["n"] * 102
    assert cohen_kappa(a, b) == pytest.approx(0.6399, abs=1e-3)
    assert cohen_kappa(["y", "y"], ["y", "y"]) is None  # undefined: one class only


def test_kappa_table_against_the_human_audit(data_dir: Path, tmp_path: Path) -> None:
    answers = {b"img1": _answer(watermark=True), b"img2": _answer(watermark=True)}
    run_labeling(data_dir, _client(tmp_path, FakeTransport(answers)), n=40, seed=0)
    review = data_dir / "review"
    write_audit_queue(data_dir, review, image_ids=[1, 2, 3, 4], name="gemini-audit")
    items = {i.item_id: i for i in read_queue(review, "gemini-audit")}
    assert len(items) == 4
    store = DecisionStore(review, "gemini-audit")
    human = {1: "yes", 2: "no", 3: "no", 4: "unsure"}  # 4 is excluded as unsure
    for item in items.values():
        values = {f.name: f.options[1].value for f in item.fields}  # "no"
        values["watermark"] = human[item.image_id or 0]
        values["relevance"] = "yes"
        store.append(
            Decision(
                item_id=item.item_id,
                values=values,
                annotator="t",
                decided_at=datetime(2026, 10, 17, tzinfo=UTC),
                elapsed_ms=1000,
            )
        )

    pl.DataFrame(
        {"image_id": [1, 2, 3, 4], "stratum": ["positive:watermark"] * 2 + ["random"] * 2}
    ).write_parquet(audit_ids_path(data_dir))
    table = kappa_table(data_dir, review, "gemini-audit")

    row = table.filter(pl.col("criterion") == "watermark").row(0, named=True)
    assert (row["n"], row["agreement"]) == (3, pytest.approx(2 / 3))
    assert row["gemini_positive"] == 2 and row["human_positive"] == 1
    shot = table.filter(pl.col("criterion") == "shot_type").row(0, named=True)
    assert (shot["n"], shot["kappa"]) == (0, None)  # not in the audit (learned from query_type)
    assert set(table["criterion"]) == set(CRITERIA)
    assert row["n_random"] == 1  # image 3; image 4 was unsure

    # a re-run of ingest that changes an image's entity must not be compared silently
    labels = pl.read_parquet(labels_path(data_dir))
    labels.with_columns(entity_id=pl.lit("e9")).write_parquet(labels_path(data_dir))
    assert kappa_table(data_dir, review, "gemini-audit")["n"].sum() == 0
