import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from pipeline.review.builders import kg_queue, moderation_queue, qrels_queue, relevance_queue
from pipeline.review.items import (
    MODERATION_FIELDS,
    Field,
    Item,
    Option,
    read_queue,
    write_queue,
)
from pipeline.review.server import make_server
from pipeline.review.store import (
    LABELS_SCHEMA,
    Decision,
    DecisionStore,
    export_all,
    export_queue,
    queue_stats,
)

YES_NO = [Option(value="yes", key="y"), Option(value="no", key="n")]


def _item(item_id: str, image_id: int | None = 1, default: str | None = None) -> Item:
    return Item(
        item_id=item_id,
        mode="moderation",
        title="Vịnh Hạ Long",
        image_id=image_id,
        entity_id="e1",
        fields=[Field(name="relevance", options=YES_NO, default=default)],
    )


def _decision(item_id: str, value: str, at: datetime, elapsed_ms: int = 5000) -> Decision:
    return Decision(
        item_id=item_id,
        values={"relevance": value},
        annotator="tester",
        decided_at=at,
        elapsed_ms=elapsed_ms,
    )


T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


# --- items -------------------------------------------------------------------------------------


def test_queue_round_trips(tmp_path: Path) -> None:
    items = [_item("a"), _item("b", image_id=None, default="no")]
    write_queue(tmp_path, "audit", items)

    assert read_queue(tmp_path, "audit") == items


def test_queue_names_are_restricted(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_queue(tmp_path, "../escape", [_item("a")])


def test_items_need_unique_ids_and_valid_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        write_queue(tmp_path, "q", [_item("a"), _item("a")])
    with pytest.raises(ValueError):
        Field(name="x", options=YES_NO, default="maybe")  # default not among options
    with pytest.raises(ValueError):
        Field(name="x", options=[Option(value="a", key="y"), Option(value="b", key="y")])
    with pytest.raises(ValueError):
        Field(name="x", options=[Option(value="a", key="Enter")])  # reserved navigation key


def test_moderation_fields_cover_the_spec_criteria() -> None:
    names = [f.name for f in MODERATION_FIELDS]
    assert names == [
        "relevance",
        "watermark",
        "text_overlay",
        "nsfw",
        "political",
        "violence",
        "shot_type",
    ]
    assert all(f.default is None for f in MODERATION_FIELDS)  # blind by default (for kappa)


# --- store -------------------------------------------------------------------------------------


def test_latest_decision_per_item_wins(tmp_path: Path) -> None:
    store = DecisionStore(tmp_path, "q")
    store.append(_decision("a", "yes", T0))
    store.append(_decision("b", "no", T0 + timedelta(seconds=5)))
    store.append(_decision("a", "no", T0 + timedelta(seconds=9)))  # revised after going back

    latest = store.latest()
    assert {k: d.values["relevance"] for k, d in latest.items()} == {"a": "no", "b": "no"}


def test_store_survives_a_torn_last_line(tmp_path: Path) -> None:
    store = DecisionStore(tmp_path, "q")
    store.append(_decision("a", "yes", T0))
    with store.path.open("ab") as handle:
        handle.write('{"item_id": "b", "annotator": "Cường'.encode()[:-1])  # torn mid-character

    assert list(store.latest()) == ["a"]
    store.append(_decision("c", "no", T0))  # appends still parse afterwards
    assert sorted(store.latest()) == ["a", "c"]


def test_decision_must_match_the_item_fields(tmp_path: Path) -> None:
    write_queue(tmp_path, "q", [_item("a")])
    store = DecisionStore(tmp_path, "q")
    items = {i.item_id: i for i in read_queue(tmp_path, "q")}

    with pytest.raises(ValueError):
        store.record(items, _decision("a", "maybe", T0))
    with pytest.raises(KeyError):
        store.record(items, _decision("zzz", "yes", T0))
    store.record(items, _decision("a", "yes", T0))
    assert list(store.latest()) == ["a"]


def test_export_uses_the_labels_shape(tmp_path: Path) -> None:
    item = Item(
        item_id="k1",
        mode="kg",
        title="A → B",
        model_version="kg-v1",
        export_keys={"src": "e1", "rel": "located_in", "dst": "p1"},
        fields=[Field(name="edge_correct", options=YES_NO)],
    )
    write_queue(tmp_path, "kg", [item, _item("unseen")])
    store = DecisionStore(tmp_path, "kg")
    store.append(
        Decision(
            item_id="k1",
            values={"edge_correct": "yes"},
            annotator="tester",
            decided_at=T0,
            elapsed_ms=3000,
        )
    )

    frame = export_queue(tmp_path, "kg")

    assert frame.columns[: len(LABELS_SCHEMA)] == LABELS_SCHEMA.names()
    assert frame.height == 1  # undecided items are not exported
    row = frame.row(0, named=True)
    assert row["source"] == "human"
    assert (row["criterion"], row["value"]) == ("edge_correct", "yes")
    assert (row["src"], row["rel"], row["dst"]) == ("e1", "located_in", "p1")
    assert (row["mode"], row["model_version"], row["queue"]) == ("kg", "kg-v1", "kg")
    assert (tmp_path / "export" / "kg.parquet").exists()
    assert (tmp_path / "export" / "kg.csv").exists()


def test_export_all_collects_human_labels_and_trec_qrels(tmp_path: Path) -> None:
    rel = Field(name="rel", options=[Option(value=v, key=v) for v in "012"])
    qrels = Item(
        item_id="q1:7", mode="qrels", title="q", image_id=7, export_keys={"qid": "q1"}, fields=[rel]
    )
    write_queue(tmp_path, "qrels", [qrels])
    write_queue(tmp_path, "audit", [_item("a")])
    DecisionStore(tmp_path, "audit").append(_decision("a", "no", T0))
    DecisionStore(tmp_path, "qrels").append(
        Decision(item_id="q1:7", values={"rel": "2"}, annotator="t", decided_at=T0, elapsed_ms=1)
    )

    export_all(tmp_path)

    labels = pl.read_parquet(tmp_path / "export" / "labels_human.parquet")
    assert labels.columns == LABELS_SCHEMA.names()
    assert sorted(labels["queue"]) == ["audit", "qrels"]
    assert (tmp_path / "export" / "qrels.qrels.tsv").read_text() == "q1\t0\t7\t2\n"
    assert not (tmp_path / "export" / "audit.qrels.tsv").exists()


def test_stats_items_per_hour_ignores_breaks(tmp_path: Path) -> None:
    write_queue(tmp_path, "q", [_item(str(i)) for i in range(10)])
    store = DecisionStore(tmp_path, "q")
    at = T0
    for i in range(6):
        at += timedelta(seconds=10) if i != 3 else timedelta(hours=2)  # a lunch break
        store.append(_decision(str(i), "yes", at, elapsed_ms=10_000))

    store.append(_decision("gone", "yes", at, elapsed_ms=10_000))  # item dropped by a rebuild

    stats = queue_stats(tmp_path, "q")

    assert (stats.total, stats.decided) == (10, 6)
    assert stats.items_per_hour == pytest.approx(360, rel=0.01)  # 10 s per item


# --- builders ----------------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    ingest = data / "ingest"
    ingest.mkdir(parents=True)
    pl.DataFrame(
        {
            "entity_id": ["e1", "e2"],
            "entity_name": ["Vịnh Hạ Long", "Chùa Một Cột"],
            "leaf_label": ["Vịnh", "Chùa"],
            "province": ["Quảng Ninh", None],
            "branch": ["A", "B"],
        }
    ).write_parquet(ingest / "entities.parquet")
    ids = list(range(1, 11))
    pl.DataFrame(
        {
            "image_id": ids,
            "page_title": [f"page {i}" for i in ids],
        }
    ).write_parquet(ingest / "images.parquet")
    pl.DataFrame(
        {
            "image_id": ids,
            "entity_id": ["e1"] * 5 + ["e2"] * 5,
            "criterion": ["relevance"] * 10,
            "value": ["yes", "yes", "yes", "no", "unlabeled", "yes", "no", "no", "yes", "yes"],
            "label_conflict": [False] * 9 + [True],
            "query_type": ["CHINH"] * 10,
        }
    ).write_parquet(ingest / "labels.parquet")
    (data / "download").mkdir()
    pl.DataFrame(
        {
            "image_id": ids,
            "download_status": ["ok"] * 9 + ["dead"],
            "thumb_path": [f"thumbs/{i}.webp" for i in ids[:9]] + [None],
        }
    ).write_parquet(data / "download" / "image_assets.parquet")
    (data / "dedup").mkdir()
    pl.DataFrame(
        {
            "cluster_id": [1, 1],
            "image_id": [1, 2],
            "method": ["representative", "pdq"],
            "distance": [0, 3],
            "is_representative": [True, False],
        }
    ).write_parquet(data / "dedup" / "dedup_members.parquet")
    return data


def test_relevance_queue_samples_reviewable_labeled_pairs(data_dir: Path) -> None:
    items = relevance_queue(data_dir, n=100, seed=0, blind=False)

    ids = {i.image_id for i in items}
    # 5 is unlabeled, 10 conflicts (and is not downloaded), 2 is a near-duplicate
    assert ids == {1, 3, 4, 6, 7, 8, 9}
    one = next(i for i in items if i.image_id == 4)
    assert one.mode == "relevance"
    assert one.title == "Vịnh Hạ Long"
    assert one.fields[0].name == "relevance"
    assert one.fields[0].default == "no"  # the existing verify label
    assert set(one.references) <= {1, 3}  # other accepted images of the same entity
    assert 4 not in one.references
    assert one.context["province"] == "Quảng Ninh"


def test_relevance_queue_blind_hides_labels_and_is_deterministic(data_dir: Path) -> None:
    a = relevance_queue(data_dir, n=3, seed=7, blind=True)
    b = relevance_queue(data_dir, n=3, seed=7, blind=True)

    assert a == b
    assert len(a) == 3
    assert all(f.default is None for i in a for f in i.fields)


def test_moderation_queue_asks_every_criterion(data_dir: Path) -> None:
    items = moderation_queue(data_dir, n=4, seed=0)

    assert len(items) == 4
    assert all([f.name for f in i.fields] == [f.name for f in MODERATION_FIELDS] for i in items)
    exact = moderation_queue(data_dir, n=1, seed=0, image_ids=[3, 7, 9])
    assert [i.image_id for i in exact] == [3, 7, 9]
    assert exact[1].entity_id == "e2"


def test_kg_queue_from_edges(data_dir: Path, tmp_path: Path) -> None:
    edges = tmp_path / "edges.parquet"
    pl.DataFrame(
        {
            "src": ["e1", "e2"],
            "rel": ["located_in", "near"],
            "dst": ["Quảng Ninh", "e1"],
            "evidence": ["description says so", None],
            "confidence": [0.9, 0.4],
        }
    ).write_parquet(edges)

    items = kg_queue(data_dir, edges, n=10, seed=0)

    assert len(items) == 2
    near = next(i for i in items if i.export_keys["rel"] == "near")
    assert near.title == "Chùa Một Cột —near→ Vịnh Hạ Long"  # entity ids resolved to names
    assert near.mode == "kg"
    assert near.image_id is None
    assert near.fields[0].name == "edge_correct"


def test_qrels_queue_samples_a_fraction_blind(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("".join(f"q{q} 0 {d} {d % 3}\n" for q in (1, 2) for d in range(1, 11)))
    queries = tmp_path / "queries.jsonl"
    (tmp_path / "empty.jsonl").write_text("")
    queries.write_text(
        "\n".join(
            json.dumps({"qid": f"q{q}", "text": f"ảnh {q}", "qtype": "single"}) for q in (1, 2)
        )
    )

    items = qrels_queue(qrels, queries, fraction=0.2, seed=0)
    with pytest.raises(ValueError, match="not in"):
        qrels_queue(qrels, tmp_path / "empty.jsonl", fraction=0.2, seed=0)

    assert len(items) == 4  # 20% of 20 judgments
    assert all(i.fields[0].default is None for i in items)
    assert {o.value for o in items[0].fields[0].options} == {"0", "1", "2"}
    assert items[0].title.startswith("ảnh ")
    assert items[0].export_keys["qid"] in {"q1", "q2"}


# --- server ------------------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    data = tmp_path / "data"
    (data / "thumbs").mkdir(parents=True)
    (data / "thumbs" / "1.webp").write_bytes(b"RIFFfakewebp")
    (data / "secret.txt").write_text("nope")
    (data / "download").mkdir()
    pl.DataFrame({"image_id": [1], "thumb_path": ["thumbs/1.webp"]}).write_parquet(
        data / "download" / "image_assets.parquet"
    )
    review = data / "review"
    write_queue(review, "audit", [_item("a"), _item("b")])
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html>spa</html>")
    httpd: ThreadingHTTPServer = make_server(
        data, review, static, annotator="tester", host="127.0.0.1", port=0
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_server_does_not_cache_the_app_shell(server: str) -> None:
    with urllib.request.urlopen(f"{server}/review") as response:
        assert response.headers["Cache-Control"] == "no-cache"
    with urllib.request.urlopen(f"{server}/thumbs/1.webp") as response:
        assert "max-age" in response.headers["Cache-Control"]


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, body: object, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_server_lists_queues_and_records_decisions(server: str) -> None:
    status, body = _get(f"{server}/api/review/queues")
    assert status == 200
    assert json.loads(body) == [{"name": "audit", "total": 2, "decided": 0}]

    status, body = _post(
        f"{server}/api/review/queues/audit/decisions",
        {"item_id": "a", "values": {"relevance": "no"}, "elapsed_ms": 4200},
    )
    assert status == 200

    status, body = _get(f"{server}/api/review/queues/audit")
    payload = json.loads(body)
    assert [i["item_id"] for i in payload["items"]] == ["a", "b"]
    assert payload["items"][0]["image_url"] == "/thumbs/1.webp"
    assert payload["decisions"] == {"a": {"relevance": "no"}}


def test_server_rejects_bad_decisions(server: str) -> None:
    url = f"{server}/api/review/queues/audit/decisions"
    assert _post(url, {"item_id": "a", "values": {"relevance": "maybe"}, "elapsed_ms": 1})[0] == 400
    assert _post(url, {"item_id": "zz", "values": {"relevance": "no"}, "elapsed_ms": 1})[0] == 404
    assert _post(f"{server}/api/review/queues/nope/decisions", {})[0] == 404


def test_server_refuses_cross_site_requests(server: str) -> None:
    url = f"{server}/api/review/queues/audit/decisions"
    body = {"item_id": "a", "values": {"relevance": "no"}, "elapsed_ms": 1}
    # a form or text/plain POST from any page skips the CORS preflight
    assert _post(url, body, {"Content-Type": "text/plain"})[0] == 415
    assert _post(url, body, {"Origin": "https://evil.example"})[0] == 403
    assert _post(url, body, {"Origin": "http://localhost:5173"})[0] == 200  # vite dev server
    request = urllib.request.Request(
        f"{server}/api/review/queues", headers={"Host": "evil.example"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request)  # DNS rebinding: the page's own host name
    assert exc.value.code == 403


def test_server_serves_thumbs_and_the_spa_but_nothing_else(server: str) -> None:
    assert _get(f"{server}/thumbs/1.webp") == (200, b"RIFFfakewebp")
    assert _get(f"{server}/thumbs/2.webp")[0] == 404
    assert _get(f"{server}/thumbs/..%2Fsecret.txt")[0] == 404
    assert _get(f"{server}/thumbs/../secret.txt")[0] == 404
    assert _get(f"{server}/review") == (200, b"<html>spa</html>")  # SPA fallback
    assert _get(f"{server}/../secret.txt")[1] != b"nope"
