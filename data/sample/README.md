# data/sample

A small, public-safe slice of the private dataset, used by tests and CI.

- Same CSV layouts as the raw `*_verified.csv` exports, so `pipeline.ingest` runs on it end-to-end (`ingest(Path("data/sample"))`). Files use neutral names (`entity_new_*`, `supplement_*`, `assignment_*`) instead of the raw names, which contain the annotator's name; ingest gives them the same precedence.
- 61 raw rows = 41 (image, entity) pairs covering `yes`, `no`, `unlabeled` and cross-export label conflicts, from every export family.
- **Landmark / scenery entities only** (pagodas, old quarters, bridges, lakes, mountains, terraced fields, caves, …) – nothing about people.
- **No annotator data**: `assigned_member`, `verified_at`, `verification`, `ghi_chu` and `note` are removed.
- Images are referenced by URL only; they remain the property of their publishers.

Regenerate (deterministic – same input gives identical files):

```bash
make sample   # python -m pipeline.sample, reads RAW_DATA_DIR
```

`tests/test_sample.py` checks the committed files: they ingest cleanly, contain every label kind, only landmark entities and no annotator columns.
