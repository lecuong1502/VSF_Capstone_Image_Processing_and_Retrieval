"""data/sample builder and the committed sample (public repo: landmarks only, no annotators)."""

import unicodedata
from pathlib import Path

import polars as pl
import pytest

from pipeline.ingest import ingest
from pipeline.sample import ANNOTATOR_COLUMNS, Quotas, build_sample, is_landmark
from tests.raw_exports import BO_SUNG_COLS, row, write_csv

REPO_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
SMALL = Quotas(yes=2, no=1, unlabeled=1, conflicts=1)


def annotator_columns_in(csv: Path) -> set[str]:
    return set(pl.read_csv(csv, n_rows=0).columns) & set(ANNOTATOR_COLUMNS)


@pytest.mark.parametrize(
    ("name", "leaf"),
    [
        ("Vịnh Hạ Long", "Vịnh & đảo đá vôi — Đông Bắc"),
        ("Hồ Ia Ring", "Hồ & đầm Trung Bộ – Tây Nguyên"),
        ("Phó Bảng", "Cao nguyên đá (Đồng Văn)"),
        ("Sông Hương", "Sông & suối — miền Trung"),
        ("Sơn Trà", "Bán đảo & mũi đất (Sơn Trà, Cà Mau)"),
        ("Động Hương Tích", "Hang động khô (hang thạch nhũ)"),
        # Decomposed (NFD) Unicode must behave like the composed form.
        (
            unicodedata.normalize("NFD", "Vịnh Hạ Long"),
            unicodedata.normalize("NFD", "Vịnh & đảo"),
        ),
    ],
)
def test_landmarks_are_accepted(name: str, leaf: str) -> None:
    assert is_landmark(name, leaf)


@pytest.mark.parametrize(
    ("name", "leaf"),
    [
        ("Mộ và khu tưởng niệm Tổng Bí thư", "Lăng tẩm & mộ chí nguyên thủ"),
        ("Đền thờ Bác Hồ", "Hồ & đầm Trung Bộ"),  # landmark leaf, but about a person
        ("Tượng Phúc Lộc Thọ tam đa", "Tượng thần linh & thổ thần"),
        ("Đèn lồng", "Đèn lồng & đèn trang trí lễ hội"),  # not a landmark category
        # Landmark words only inside the place list / as a qualifier, not the category itself:
        ("Hạt tiêu đen Chư Sê", "Hạt tiêu & vỏ-quả thơm gia vị (hồ tiêu Phú Quốc, Chư Sê)"),
        ("Rìu đồng Quỳ Chử", "Rìu · lưỡi cày · công cụ đồng Đông Sơn (An Khê, Núi Nưa)"),
        ("Rong sụn Ninh Thuận", "Trại nuôi rong / tảo biển & thủy sinh thực vật"),
        ("Bia chùa Sùng Thiện", "Bia chùa · bia tháp Phật giáo"),
        ("Voọc chà vá", "Động vật hoang dã — linh trưởng"),
    ],
)
def test_people_and_non_landmarks_are_rejected(name: str, leaf: str) -> None:
    assert not is_landmark(name, leaf)


@pytest.fixture
def raw_with_person(raw_dir: Path) -> Path:
    write_csv(
        raw_dir / "bo_sung_people_verified.csv",
        BO_SUNG_COLS,
        [
            row(
                "e5",
                "u7",
                "yes",
                entity_name="Mộ Tổng Bí thư",
                leaf_label="Lăng tẩm & mộ chí",
                assigned_member="Someone",
            )
        ],
    )
    return raw_dir


def test_sample_ingests_with_every_label_kind(raw_with_person: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"

    build_sample(raw_with_person, out, SMALL)
    result = ingest(out)

    values = set(result.labels["value"])
    assert values == {"yes", "no", "unlabeled"}
    assert result.conflicts.height == 1
    assert "e5" not in set(result.labels["entity_id"])


def test_sample_drops_annotator_columns(raw_with_person: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"

    build_sample(raw_with_person, out, SMALL)

    for csv in out.glob("*.csv"):
        assert not annotator_columns_in(csv), csv.name
        assert "Someone" not in csv.read_text()


def test_sample_is_deterministic_and_replaces_old_files(raw_dir: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    second.mkdir()
    (second / "stale_verified.csv").write_text("old")

    build_sample(raw_dir, first, SMALL)
    build_sample(raw_dir, second, SMALL)

    names = sorted(p.name for p in first.glob("*.csv"))
    assert names == sorted(p.name for p in second.glob("*.csv"))
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()


# --- the committed sample (runs in CI) ----------------------------------------------------


def test_committed_sample_ingests_end_to_end() -> None:
    result = ingest(REPO_SAMPLE)

    assert 30 <= result.labels.height <= 80
    assert set(result.labels["value"]) == {"yes", "no", "unlabeled"}
    assert result.conflicts.height >= 1
    assert len(result.source_files) >= 3


def test_committed_sample_has_only_landmarks_and_no_annotator_data() -> None:
    result = ingest(REPO_SAMPLE)

    for entity in result.entities.iter_rows(named=True):
        assert is_landmark(entity["entity_name"], entity["leaf_label"]), entity["entity_name"]
    for csv in REPO_SAMPLE.glob("*.csv"):
        assert not annotator_columns_in(csv), csv.name


def test_sample_uses_public_file_names(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"

    # One "yes" per export family, so every family's file is written.
    build_sample(raw_dir, out, Quotas(yes=3, no=1, unlabeled=1, conflicts=1))

    assert sorted(p.name for p in out.glob("*.csv")) == [
        "assignment_verified.csv",
        "entity_new_1-10_verified.csv",
        "supplement_verified.csv",
    ]


def test_committed_sample_file_names_contain_no_person_names() -> None:
    names = [p.name for p in REPO_SAMPLE.glob("*.csv")]

    assert names and not [n for n in names if "cuong" in n.lower()]
