"""ingest: merging the verified CSV exports (three schemas) into entities / images / labels."""

from pathlib import Path

import polars as pl
import pytest

from pipeline.ingest import IngestError, image_id_for, ingest, write_outputs

# Real header layouts of the three export families (see RAW_DATA_DIR).
PHAN_CONG_COLS = [
    "STT",
    "STT_goc_14k",
    "branch",
    "leaf_label",
    "leaf_id",
    "leaf_level",
    "leaf_depth",
    "leaf_total_expected",
    "entity_id",
    "entity_name",
    "description",
    "entity_slug",
    "entity_status",
    "entity_kind",
    "entity_approved_photos",
    "verification",
    "ghi_chu",
    "verified_at",
    "priority",
    "score",
    "query_type",
    "image_order",
    "image_url",
    "page_url",
    "page_title",
    "image_source",
    "image_width",
    "image_height",
    "verify",
]
ENTITY_NEW_COLS = [
    "STT",
    "STT_goc_14k",
    "entity_stt_new",
    "assigned_member",
    "branch",
    "leaf_label",
    "leaf_id",
    "leaf_level",
    "leaf_depth",
    "leaf_total_expected",
    "entity_id",
    "entity_name",
    "description",
    "entity_slug",
    "entity_status",
    "entity_kind",
    "entity_approved_photos",
    "verification",
    "ghi_chu",
    "verified_at",
    "priority",
    "score",
    "query_type",
    "image_order",
    "image_url",
    "page_url",
    "page_title",
    "image_source",
    "image_width",
    "image_height",
    "target_group",
    "source_dataset",
    "verify",
    "note",
]
BO_SUNG_COLS = [
    "STT",
    "entity_stt",
    "entity_id",
    "entity_name",
    "entity_slug",
    "branch",
    "priority",
    "score",
    "leaf_label",
    "description",
    "so_anh_pass_master",
    "entity_duoi_5_master",
    "image_order",
    "image_url",
    "page_url",
    "page_title",
    "image_source",
    "image_width",
    "image_height",
    "query_type",
    "nguon",
    "crawled_at",
    "verify",
    "note",
    "verified_at",
    "assigned_member",
]


def row(entity: str, url: str, verify: str, **extra: str) -> dict[str, str]:
    return {
        "entity_id": entity,
        "entity_name": f"Name {entity}",
        "entity_slug": f"scope-1-{entity}",
        "branch": "A",
        "leaf_label": f"Leaf {entity}",
        "description": f"About {entity}",
        "priority": "P0",
        "score": "5.0",
        "image_url": url,
        "page_url": f"https://page/{url}",
        "page_title": f"Title {url}",
        "image_source": "ivivu.com",
        "image_width": "1080",
        "image_height": "720",
        "query_type": "CHINH",
        "verify": verify,
        **extra,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    frame = pl.DataFrame([{c: r.get(c, "") for c in columns} for r in rows], schema=columns)
    # Exports start with a UTF-8 BOM, like the real files.
    path.write_text("﻿" + frame.write_csv(), encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_csv(
        raw / "Cuong_phan_cong_verified.csv",
        PHAN_CONG_COLS,
        [
            row("e1", "u1", "yes", entity_name="Name e1 (old)"),
            row("e1", "u2", "yes", entity_name="Name e1 (old)"),  # conflicts with entity_new
            row("e2", "u3", ""),  # unlabeled everywhere
            row("e2", "u4", "no"),
            row("e2", "u4", "no"),  # duplicate row in the same export
            row("e3", "u5", ""),  # labeled by bo_sung below
        ],
    )
    write_csv(
        raw / "Cuong_entity_new_1-10_verified.csv",
        ENTITY_NEW_COLS,
        [
            row("e1", "u2", "no", assigned_member="Cuong", note="watermark"),
            row("e4", "u1", "No "),  # same image, different entity; messy casing/whitespace
            row("e1", "u6", "yes"),
        ],
    )
    write_csv(
        raw / "bo_sung_anh_cho_entities_nhom_1_Cuong_verified.csv",
        BO_SUNG_COLS,
        [row("e3", "u5", "yes", nguon="google", crawled_at="2026-01-01")],
    )
    (raw / "Cuong_phan_cong.csv").write_text("not a verified export, must be ignored\n")
    return raw


def labels_by_pair(labels: pl.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    return {(r["image_url"], r["entity_id"]): r for r in labels.iter_rows(named=True)}


def test_reads_every_schema_and_counts(raw_dir: Path) -> None:
    result = ingest(raw_dir)

    assert result.images.height == 6  # u1..u6
    assert result.labels.height == 7  # unique (image_url, entity_id) pairs
    assert sorted(result.entities["entity_id"].to_list()) == ["e1", "e2", "e3", "e4"]
    assert sorted(result.source_files) == [
        "Cuong_entity_new_1-10_verified.csv",
        "Cuong_phan_cong_verified.csv",
        "bo_sung_anh_cho_entities_nhom_1_Cuong_verified.csv",
    ]


def test_labels_are_keyed_by_image_and_entity(raw_dir: Path) -> None:
    labels = labels_by_pair(ingest(raw_dir).labels)

    assert labels[("u1", "e1")]["value"] == "yes"
    assert labels[("u1", "e4")]["value"] == "no"  # normalized from "No "
    assert labels[("u1", "e1")]["image_id"] == labels[("u1", "e4")]["image_id"]


def test_newer_export_wins_and_conflict_is_flagged(raw_dir: Path) -> None:
    result = ingest(raw_dir)
    labels = labels_by_pair(result.labels)

    conflict = labels[("u2", "e1")]
    assert conflict["value"] == "no"
    assert conflict["label_conflict"] is True
    assert conflict["source"] == "Cuong_entity_new_1-10_verified.csv"
    assert conflict["note"] == "watermark"
    assert result.conflicts.select("image_url", "entity_id").rows() == [("u2", "e1")]
    assert set(result.conflicts["labels"].to_list()[0].split(";")) == {
        "Cuong_entity_new_1-10_verified.csv=no",
        "Cuong_phan_cong_verified.csv=yes",
    }
    assert sum(bool(r["label_conflict"]) for r in labels.values()) == 1


def test_empty_verify_is_unlabeled_unless_another_export_labels_it(raw_dir: Path) -> None:
    labels = labels_by_pair(ingest(raw_dir).labels)

    assert labels[("u3", "e2")]["value"] == "unlabeled"
    assert labels[("u5", "e3")]["value"] == "yes"
    assert labels[("u5", "e3")]["label_conflict"] is False


def test_entity_attributes_come_from_the_newest_export(raw_dir: Path) -> None:
    entities = {r["entity_id"]: r for r in ingest(raw_dir).entities.iter_rows(named=True)}

    assert entities["e1"]["entity_name"] == "Name e1"  # not "(old)" from Cuong_phan_cong
    assert entities["e2"]["leaf_label"] == "Leaf e2"


def test_image_ids_are_stable_hashes_of_the_url(raw_dir: Path) -> None:
    images = ingest(raw_dir).images

    assert images["image_id"].n_unique() == images.height
    assert dict(images.select("image_url", "image_id").rows())["u1"] == image_id_for("u1")
    assert image_id_for("u1") == image_id_for("u1") > 0


def test_rerun_produces_identical_output(raw_dir: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "out1", tmp_path / "out2"

    write_outputs(ingest(raw_dir), first)
    write_outputs(ingest(raw_dir), second)

    for name in ("entities.parquet", "images.parquet", "labels.parquet", "conflicts.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert (first / "vsf.duckdb").exists()


def test_unknown_verified_export_is_rejected(raw_dir: Path) -> None:
    write_csv(raw_dir / "someone_else_verified.csv", BO_SUNG_COLS, [row("e9", "u9", "yes")])

    with pytest.raises(IngestError, match="someone_else_verified.csv"):
        ingest(raw_dir)


def test_missing_required_column_is_rejected(raw_dir: Path) -> None:
    columns = [c for c in BO_SUNG_COLS if c != "verify"]
    write_csv(raw_dir / "bo_sung_extra_verified.csv", columns, [row("e9", "u9", "yes")])

    with pytest.raises(IngestError, match="verify"):
        ingest(raw_dir)


def test_unexpected_verify_value_is_rejected(raw_dir: Path) -> None:
    write_csv(raw_dir / "bo_sung_x_verified.csv", BO_SUNG_COLS, [row("e9", "u9", "maybe")])

    with pytest.raises(IngestError, match="maybe"):
        ingest(raw_dir)
