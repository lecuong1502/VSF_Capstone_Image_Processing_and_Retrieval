from pathlib import Path

import pytest

from tests.raw_exports import BO_SUNG_COLS, ENTITY_NEW_COLS, PHAN_CONG_COLS, row, write_csv


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
            row("e2", "u3", "", ghi_chu="Typo in description"),  # unlabeled everywhere
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
        [row("e3", "u5", "yes", nguon="50k", crawled_at="2026-01-01T10:00:00")],
    )
    (raw / "Cuong_phan_cong.csv").write_text("not a verified export, must be ignored\n")
    return raw
