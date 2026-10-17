"""ingest: merging the verified CSV exports (three schemas) into entities / images / labels."""

from pathlib import Path

import polars as pl
import pytest

from pipeline.ingest import IngestError, image_id_for, ingest, public_file_name, write_outputs
from tests.raw_exports import BO_SUNG_COLS, ENTITY_NEW_COLS, PHAN_CONG_COLS, row, write_csv


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
    assert entities["e2"]["leaf_label"] == "Hang động e2"


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


def test_disagreement_inside_the_newest_family_is_a_conservative_conflict(raw_dir: Path) -> None:
    write_csv(
        raw_dir / "Cuong_entity_new_11-20_verified.csv", ENTITY_NEW_COLS, [row("e1", "u6", "no")]
    )

    label = labels_by_pair(ingest(raw_dir).labels)[("u6", "e1")]

    assert label["value"] == "no"
    assert label["label_conflict"] is True


def test_blank_entity_id_is_rejected(raw_dir: Path) -> None:
    write_csv(raw_dir / "bo_sung_y_verified.csv", BO_SUNG_COLS, [row("  ", "u9", "yes")])

    with pytest.raises(IngestError, match="without image_url/entity_id"):
        ingest(raw_dir)


def test_crawl_metadata_is_kept_on_images(raw_dir: Path) -> None:
    images = {r["image_url"]: r for r in ingest(raw_dir).images.iter_rows(named=True)}

    assert images["u5"]["crawl_batch"] == "50k"
    assert images["u5"]["crawled_at"] == "2026-01-01T10:00:00"
    assert images["u1"]["crawl_batch"] is None


def test_entities_carry_description_note_and_relevant_image_count(raw_dir: Path) -> None:
    entities = {r["entity_id"]: r for r in ingest(raw_dir).entities.iter_rows(named=True)}

    assert entities["e2"]["description_note"] == "Typo in description"
    # e1: u1 yes, u6 yes; u2 is a conflict and does not count.
    assert entities["e1"]["n_yes_images"] == 2
    assert entities["e4"]["n_yes_images"] == 0


def test_public_export_names_keep_the_same_precedence(tmp_path: Path) -> None:
    raw = tmp_path / "public"
    raw.mkdir()
    write_csv(raw / "assignment_verified.csv", PHAN_CONG_COLS, [row("e1", "u1", "yes")])
    write_csv(raw / "supplement_verified.csv", BO_SUNG_COLS, [row("e1", "u1", "yes")])
    write_csv(raw / "entity_new_1-10_verified.csv", ENTITY_NEW_COLS, [row("e1", "u1", "no")])

    label = ingest(raw).labels.row(0, named=True)

    assert label["value"] == "no"
    assert label["source"] == "entity_new_1-10_verified.csv"
    assert label["label_conflict"] is True


@pytest.mark.parametrize(
    ("raw_name", "public_name"),
    [
        ("Cuong_entity_new_1-1313_verified.csv", "entity_new_1-1313_verified.csv"),
        ("bo_sung_anh_cho_entities_nhom_1_Cuong_verified.csv", "supplement_verified.csv"),
        ("Cuong_phan_cong_verified.csv", "assignment_verified.csv"),
        ("assignment_verified.csv", "assignment_verified.csv"),
    ],
)
def test_public_file_name(raw_name: str, public_name: str) -> None:
    assert public_file_name(raw_name) == public_name
