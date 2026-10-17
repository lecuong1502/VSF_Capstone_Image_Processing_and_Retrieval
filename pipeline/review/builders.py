"""Queue builders for the three review modes, from pipeline outputs in `$DATA_DIR`.

Only images that were downloaded (have a thumbnail) and are not near-duplicates of a kept image
are reviewed. Samples are seeded, so rebuilding a queue gives the same items.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl

from pipeline.download import assets_path
from pipeline.review.items import MODERATION_FIELDS, Field, Item, Option, yes_no

N_REFERENCES = 4  # accepted images of the same entity shown next to the reviewed one

REL_FIELD = Field(
    name="rel",
    label="Mức liên quan",
    options=[
        Option(value="0", key="0", label="không liên quan"),
        Option(value="1", key="1", label="liên quan một phần"),
        Option(value="2", key="2", label="rất liên quan"),
    ],
)


def _reviewable_images(data_dir: Path) -> pl.DataFrame:
    """image_id of downloaded images with a thumbnail, minus non-representative duplicates."""
    assets = pl.read_parquet(assets_path(data_dir)).filter(
        (pl.col("download_status") == "ok") & pl.col("thumb_path").is_not_null()
    )
    members = data_dir / "dedup" / "dedup_members.parquet"
    if members.exists():
        dropped = pl.read_parquet(members).filter(~pl.col("is_representative"))
        assets = assets.join(dropped.select("image_id"), on="image_id", how="anti")
    return assets.select("image_id")


def _sample(frame: pl.DataFrame, n: int, seed: int, by: list[str]) -> pl.DataFrame:
    frame = frame.sort(by)
    return frame.sample(n=min(n, frame.height), seed=seed, shuffle=True)


def labeled_pairs(data_dir: Path) -> pl.DataFrame:
    """Non-conflicting yes/no relevance labels of reviewable images."""
    labels = pl.read_parquet(data_dir / "ingest" / "labels.parquet").filter(
        (pl.col("criterion") == "relevance")
        & pl.col("value").is_in(["yes", "no"])
        & ~pl.col("label_conflict")
    )
    return labels.join(_reviewable_images(data_dir), on="image_id")


def _entities(data_dir: Path) -> dict[str, dict[str, str | None]]:
    entities = pl.read_parquet(data_dir / "ingest" / "entities.parquet")
    return {row["entity_id"]: row for row in entities.iter_rows(named=True)}


def _references(pairs: pl.DataFrame, seed: int) -> dict[str, list[int]]:
    """Up to N_REFERENCES accepted image_ids per entity (one extra so the item itself can drop)."""
    accepted = _sample(
        pairs.filter(pl.col("value") == "yes"), pairs.height, seed, ["entity_id", "image_id"]
    )
    grouped = accepted.group_by("entity_id").agg(pl.col("image_id").head(N_REFERENCES + 1))
    return dict(grouped.iter_rows())


def _entity_context(entity: dict[str, str | None]) -> dict[str, str]:
    keys = ("leaf_label", "province")
    return {k: str(entity[k]) for k in keys if entity.get(k)}


def _pick(refs: list[int], image_id: int | None) -> list[int]:
    return [r for r in refs if r != image_id][:N_REFERENCES]


def relevance_queue(data_dir: Path, *, n: int, seed: int, blind: bool) -> list[Item]:
    """Audit the human `verify` labels; `blind` hides the existing label (for kappa)."""
    pairs = labeled_pairs(data_dir)
    titles = pl.read_parquet(data_dir / "ingest" / "images.parquet").select(
        "image_id", "page_title"
    )
    entities = _entities(data_dir)
    refs = _references(pairs, seed)
    picked = _sample(pairs, n, seed, ["image_id", "entity_id"]).join(
        titles, on="image_id", how="left"
    )
    items = []
    for row in picked.iter_rows(named=True):
        entity = entities[row["entity_id"]]
        context = _entity_context(entity) | {
            k: row[k] for k in ("query_type", "page_title") if row.get(k)
        }
        items.append(
            Item(
                item_id=f"{row['image_id']}:{row['entity_id']}",
                mode="relevance",
                title=str(entity["entity_name"]),
                image_id=row["image_id"],
                entity_id=row["entity_id"],
                context=context,
                references=_pick(refs.get(row["entity_id"], []), row["image_id"]),
                fields=[yes_no("relevance", "Đúng thực thể?", None if blind else row["value"])],
            )
        )
    return items


def one_entity_per_image(pairs: pl.DataFrame) -> pl.DataFrame:
    """The entity an image is judged against: an accepted one if any, else the first by id.
    Gemini labeling (#13) uses the same rule, so its prompt and the audit show the same entity."""
    return pairs.sort("image_id", pl.col("value") != "yes", "entity_id").unique(
        "image_id", keep="first", maintain_order=True
    )


def moderation_queue(
    data_dir: Path, *, n: int, seed: int, image_ids: list[int] | None = None
) -> list[Item]:
    """Blind audit of every spec criterion for `n` images (or exactly `image_ids`), each shown
    with one of its labeled entities."""
    pairs = labeled_pairs(data_dir)
    one = one_entity_per_image(pairs)
    if image_ids is not None:
        picked = one.filter(pl.col("image_id").is_in(image_ids))
    else:
        picked = _sample(one, n, seed, ["image_id"])
    entities = _entities(data_dir)
    refs = _references(pairs, seed)
    return [
        Item(
            item_id=f"{row['image_id']}:{row['entity_id']}",
            mode="moderation",
            title=str(entities[row["entity_id"]]["entity_name"]),
            image_id=row["image_id"],
            entity_id=row["entity_id"],
            context=_entity_context(entities[row["entity_id"]]),
            references=_pick(refs.get(row["entity_id"], []), row["image_id"]),
            fields=list(MODERATION_FIELDS),
        )
        for row in picked.sort("image_id").iter_rows(named=True)
    ]


def _read_edges(path: Path) -> pl.DataFrame:
    if path.suffix in {".sqlite", ".db"}:
        with sqlite3.connect(path) as conn:
            cursor = conn.execute("SELECT * FROM kg_edges")
            columns = [c[0] for c in cursor.description]
            return pl.DataFrame(cursor.fetchall(), schema=columns, orient="row")
    return pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)


def kg_queue(data_dir: Path, edges_path: Path, *, n: int, seed: int) -> list[Item]:
    """Hand-check `n` KG edges (`src`, `rel`, `dst`, optional `evidence`, `confidence`)."""
    edges = _read_edges(edges_path)
    entities = _entities(data_dir)
    refs = _references(labeled_pairs(data_dir), seed)

    def name(node: str) -> str:
        return str(entities[node]["entity_name"]) if node in entities else node

    items = []
    for row in _sample(edges, n, seed, ["src", "rel", "dst"]).iter_rows(named=True):
        src, rel, dst = str(row["src"]), str(row["rel"]), str(row["dst"])
        context = {k: str(row[k]) for k in ("evidence", "confidence") if row.get(k) is not None}
        items.append(
            Item(
                item_id=f"{src}|{rel}|{dst}",
                mode="kg",
                title=f"{name(src)} —{rel}→ {name(dst)}",
                entity_id=src if src in entities else None,
                context=context,
                references=(refs.get(src, []) + refs.get(dst, []))[:N_REFERENCES],
                export_keys={"src": src, "rel": rel, "dst": dst},
                fields=[yes_no("edge_correct", "Cạnh đúng?")],
            )
        )
    return items


def qrels_queue(qrels_path: Path, queries_path: Path, *, fraction: float, seed: int) -> list[Item]:
    """Blind re-judgment of a `fraction` of TREC qrels (`qid 0 image_id rel`) on 0/1/2."""
    rows = [line.split() for line in qrels_path.read_text().splitlines() if line.strip()]
    qrels = pl.DataFrame(
        [(qid, int(image_id), int(rel)) for qid, _, image_id, rel in rows],
        schema={"qid": pl.String, "image_id": pl.Int64, "rel": pl.Int64},
        orient="row",
    )
    queries = {
        q["qid"]: q
        for q in (json.loads(line) for line in queries_path.read_text().splitlines() if line)
    }
    missing = set(qrels["qid"]) - set(queries)
    if missing:
        raise ValueError(f"{len(missing)} qids in {qrels_path} are not in {queries_path}")
    n = round(fraction * qrels.height)
    items = []
    for row in _sample(qrels, n, seed, ["qid", "image_id"]).iter_rows(named=True):
        query = queries[row["qid"]]  # checked above
        items.append(
            Item(
                item_id=f"{row['qid']}:{row['image_id']}",
                mode="qrels",
                title=str(query["text"]),
                image_id=row["image_id"],
                context={"qtype": str(query.get("qtype", ""))},
                export_keys={"qid": row["qid"]},
                fields=[REL_FIELD],
            )
        )
    return items
