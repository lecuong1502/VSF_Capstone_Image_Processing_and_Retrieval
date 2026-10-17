"""Embedding with tiny hand-built ONNX models (combined and split layouts) and fake shards."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import onnx
import polars as pl
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from pipeline.embed import bench, embedding_path, load_embeddings, run_embedding
from pipeline.embedder import Embedder, Preprocessor, model_dir_name

DIM = 4


def _tiny_vision(inputs: list[onnx.ValueInfoProto], out_name: str) -> list[onnx.NodeProto]:
    # pixel_values (b,3,H,W) -> mean over H,W -> (b,3) @ W (3,DIM)
    return [
        helper.make_node("ReduceMean", ["pixel_values", "axes_hw"], ["pooled"], keepdims=0),
        helper.make_node("MatMul", ["pooled", "w_img"], [out_name]),
    ]


def _tiny_text(out_name: str) -> list[onnx.NodeProto]:
    # input_ids (b,s) -> float mean over s -> (b,1) @ W (1,DIM)
    return [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["ids_f", "axes_s"], ["ids_mean"], keepdims=1),
        helper.make_node("MatMul", ["ids_mean", "w_txt"], ["txt_scaled"]),
        # A bias makes the direction depend on the mean, so padding changes the embedding.
        helper.make_node("Add", ["txt_scaled", "b_txt"], [out_name]),
    ]


INITS = [
    numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes_hw"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_s"),
    numpy_helper.from_array(
        np.arange(3 * DIM, dtype=np.float32).reshape(3, DIM) / 10 + 0.1, "w_img"
    ),
    numpy_helper.from_array(np.ones((1, DIM), dtype=np.float32), "w_txt"),
    numpy_helper.from_array(np.arange(DIM, dtype=np.float32), "b_txt"),
]
PIXELS = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["b", 3, "h", "w"])
IDS = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])


def _save(graph: onnx.GraphProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)


def _tokenizer(path: Path, fixed_length: int | None = None) -> None:
    vocab = {"[PAD]": 0, "[UNK]": 1, "hồ": 2, "gươm": 3, "vịnh": 4}
    padding = (
        {
            "strategy": {"Fixed": fixed_length},
            "direction": "Right",
            "pad_to_multiple_of": None,
            "pad_id": 0,
            "pad_type_id": 0,
            "pad_token": "[PAD]",
        }
        if fixed_length
        else None
    )
    truncation = (
        {"direction": "Right", "max_length": fixed_length, "strategy": "LongestFirst", "stride": 0}
        if fixed_length
        else None
    )
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": truncation,
                "padding": padding,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def combined_model(tmp_path: Path) -> Path:
    root = tmp_path / "models" / "acme--tiny-combined"
    out_img = helper.make_tensor_value_info(
        "l2norm_image_embeddings", TensorProto.FLOAT, ["b", DIM]
    )
    out_txt = helper.make_tensor_value_info("l2norm_text_embeddings", TensorProto.FLOAT, ["b", DIM])
    nodes = _tiny_vision([PIXELS], "img_raw") + _tiny_text("txt_raw")
    nodes += [
        helper.make_node("LpNormalization", ["img_raw"], ["l2norm_image_embeddings"], axis=1),
        helper.make_node("LpNormalization", ["txt_raw"], ["l2norm_text_embeddings"], axis=1),
    ]
    _save(
        helper.make_graph(nodes, "combined", [IDS, PIXELS], [out_txt, out_img], INITS),
        root / "onnx" / "model_fp16.onnx",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": 32,
                "resize_mode": "shortest",
                "interpolation": "bicubic",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            }
        )
    )
    _tokenizer(root / "tokenizer.json")
    return root


@pytest.fixture
def split_model(tmp_path: Path) -> Path:
    root = tmp_path / "models" / "acme--tiny-split"
    pooled_img = helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["b", DIM])
    _save(
        helper.make_graph(
            _tiny_vision([PIXELS], "pooler_output"),
            "vision",
            [PIXELS],
            [pooled_img],
            INITS[:1] + INITS[2:3],
        ),
        root / "onnx" / "vision_model_fp16.onnx",
    )
    pooled_txt = helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["b", DIM])
    _save(
        helper.make_graph(
            _tiny_text("pooler_output"), "text", [IDS], [pooled_txt], INITS[1:2] + INITS[3:]
        ),
        root / "onnx" / "text_model_fp16.onnx",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_resize": True,
                "size": {"height": 24, "width": 24},
                "resample": 2,
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    # Like onnx-community SigLIP2: fixed padding lives in tokenizer.json, lowercasing in the config.
    (root / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}))
    _tokenizer(root / "tokenizer.json", fixed_length=8)
    return root


def jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


# --- preprocessing -------------------------------------------------------------------------------


def test_jina_style_preprocess_resizes_shortest_side_then_center_crops() -> None:
    pre = Preprocessor.from_config(
        {
            "size": 512,
            "resize_mode": "shortest",
            "interpolation": "bicubic",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
    )

    x = pre(jpeg((1024, 768), (255, 255, 255)))

    assert x.shape == (3, 512, 512) and x.dtype == np.float32
    assert x[0, 0, 0] == pytest.approx((1 - 0.48145466) / 0.26862954, rel=1e-3)


def test_hf_style_preprocess_resizes_to_fixed_size() -> None:
    pre = Preprocessor.from_config(
        {
            "do_resize": True,
            "size": {"height": 256, "width": 256},
            "resample": 2,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )

    x = pre(jpeg((1000, 300), (0, 0, 0)))

    assert x.shape == (3, 256, 256)
    assert x.min() == pytest.approx(-1.0, abs=1e-2)


def test_preprocess_handles_grayscale_and_alpha() -> None:
    pre = Preprocessor.from_config(
        {"size": 16, "resize_mode": "shortest", "mean": [0, 0, 0], "std": [1, 1, 1]}
    )
    buf = io.BytesIO()
    Image.new("LA", (40, 40), (255, 0)).save(buf, "PNG")

    x = pre(buf.getvalue())

    assert x.shape == (3, 16, 16) and x.dtype == np.float32


def test_hf_shortest_edge_resizes_then_center_crops() -> None:
    pre = Preprocessor.from_config(
        {
            "size": {"shortest_edge": 40},
            "do_center_crop": True,
            "crop_size": {"height": 32, "width": 32},
            "image_mean": [0, 0, 0],
            "image_std": [1, 1, 1],
            "do_rescale": True,
            "rescale_factor": 1 / 255,
        }
    )

    prepared = pre.prepare(jpeg((80, 60), (10, 20, 30)))

    assert prepared.shape == (32, 32, 3) and prepared.dtype == np.uint8


def test_workers_send_uint8_and_normalize_happens_in_batch() -> None:
    pre = Preprocessor.from_config(
        {"size": 16, "resize_mode": "shortest", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    )
    image = jpeg((40, 30), (255, 255, 255))

    batch = pre.normalize(np.stack([pre.prepare(image)] * 2))

    assert batch.shape == (2, 3, 16, 16) and batch.dtype == np.float32
    assert np.allclose(batch, pre(image)[None], atol=1e-6)


# --- embedder ------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["combined_model", "split_model"])
def test_embedder_returns_unit_vectors(fixture: str, request: pytest.FixtureRequest) -> None:
    embedder = Embedder.load(request.getfixturevalue(fixture), providers=["CPUExecutionProvider"])
    images = [jpeg((60, 40), (200, 30, 30)), jpeg((40, 60), (30, 200, 30))]

    vectors = embedder.embed_images(images)
    texts = embedder.embed_texts(["hồ gươm", "vịnh"])

    assert vectors.shape == (2, DIM) and texts.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(texts, axis=1), 1, atol=1e-5)
    assert embedder.dim == DIM
    assert not np.allclose(vectors[0], vectors[1])


def test_fixed_padding_and_lowercasing_from_the_model_files(split_model: Path) -> None:
    embedder = Embedder.load(split_model, providers=["CPUExecutionProvider"])

    ids = embedder.tokenize(["hồ", "HỒ GƯƠM vịnh"])

    assert [len(row) for row in ids] == [8, 8]
    assert ids[1][:3] == [2, 3, 4]


def test_text_embedding_does_not_depend_on_the_batch(combined_model: Path) -> None:
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])

    alone = embedder.embed_texts(["hồ"])
    batched = embedder.embed_texts(["hồ", "hồ gươm vịnh vịnh"])

    assert np.allclose(alone[0], batched[0], atol=1e-6)


def test_model_dir_name() -> None:
    assert model_dir_name("jinaai/jina-clip-v2") == "jinaai--jina-clip-v2"


# --- pipeline ------------------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    run = data / "download" / "runs" / "r1"
    run.mkdir(parents=True)
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
    for shard, keys in (("00000", [0, 1]), ("00001", [2])):
        with tarfile.open(run / f"{shard}.tar", "w") as tar:
            for k in keys:
                data_bytes = jpeg((50, 40), colors[k])
                info = tarfile.TarInfo(f"{k:09d}.jpg")
                info.size = len(data_bytes)
                tar.addfile(info, io.BytesIO(data_bytes))
    pl.DataFrame(
        {
            "image_id": [1, 2, 3, 4],
            "download_status": ["ok", "ok", "ok", "dead"],
            "sha256": ["a", "b", "c", None],
            "shard_path": [
                "download/runs/r1/00000.tar#000000000",
                "download/runs/r1/00000.tar#000000001",
                "download/runs/r1/00001.tar#000000002",
                None,
            ],
        }
    ).write_parquet(data / "download" / "image_assets.parquet")
    return data


def test_run_embedding_embeds_every_ok_image_once(data_dir: Path, combined_model: Path) -> None:
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])

    first = run_embedding(data_dir, embedder, "acme/tiny", workers=1)
    second = run_embedding(data_dir, embedder, "acme/tiny", workers=1)

    table = load_embeddings(embedding_path(data_dir, "acme/tiny"))
    assert sorted(table["image_id"]) == [1, 2, 3]
    assert table.schema["embedding"] == pl.Array(pl.Float32, DIM)
    assert first.embedded == 3 and second.embedded == 0
    assert first.images_per_s > 0


def test_changed_image_is_reembedded(data_dir: Path, combined_model: Path) -> None:
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])
    run_embedding(data_dir, embedder, "acme/tiny", workers=1)
    assets_file = data_dir / "download" / "image_assets.parquet"
    pl.read_parquet(assets_file).with_columns(
        pl.when(pl.col("image_id") == 2)
        .then(pl.lit("b2"))
        .otherwise(pl.col("sha256"))
        .alias("sha256")
    ).write_parquet(assets_file)

    again = run_embedding(data_dir, embedder, "acme/tiny", workers=1)

    table = load_embeddings(embedding_path(data_dir, "acme/tiny"))
    assert again.embedded == 1
    assert table.height == 3
    assert dict(table.select("image_id", "sha256").rows())[2] == "b2"


def test_bench_reports_images_per_second(combined_model: Path) -> None:
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])

    rows = bench(embedder, [jpeg((50, 40), (1, 2, 3))] * 4, batch_sizes=[1, 2], repeats=1)

    assert [r["batch_size"] for r in rows] == [1, 2]
    assert all((r["images_per_s"] or 0) > 0 for r in rows)


class FlakyMemory:
    """Session stand-in that fails like onnxruntime when a batch is larger than `limit`."""

    def __init__(self, inner: object, limit: int) -> None:
        self.inner, self.limit = inner, limit
        self.sizes: list[int] = []

    def run(self, outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        size = len(feed["pixel_values"])
        self.sizes.append(size)
        if size > self.limit:
            raise RuntimeError("Failed to allocate memory for requested buffer")
        return self.inner.run(outputs, feed)  # type: ignore[attr-defined, no-any-return]

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def test_embedder_batches_and_splits_on_out_of_memory(combined_model: Path) -> None:
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"], batch_size=4)
    flaky = FlakyMemory(embedder._combined, limit=2)
    embedder._combined = flaky

    vectors = embedder.embed_images([jpeg((50, 40), (i * 20, 10, 10)) for i in range(10)])

    assert vectors.shape == (10, DIM)
    assert max(flaky.sizes) == 4 and all(s <= 4 for s in flaky.sizes)
    assert flaky.sizes.count(2) >= 4  # each 4-batch was split in two


def test_unreadable_images_are_recorded_and_not_retried(
    data_dir: Path, combined_model: Path
) -> None:
    run = data_dir / "download" / "runs" / "r1"
    with tarfile.open(run / "00002.tar", "w") as tar:
        info = tarfile.TarInfo("000000009.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    assets_file = data_dir / "download" / "image_assets.parquet"
    pl.concat(
        [
            pl.read_parquet(assets_file),
            pl.DataFrame(
                {
                    "image_id": [9],
                    "download_status": ["ok"],
                    "sha256": ["z"],
                    "shard_path": ["download/runs/r1/00002.tar#000000009"],
                }
            ),
        ]
    ).write_parquet(assets_file)
    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])

    first = run_embedding(data_dir, embedder, "acme/tiny", workers=1)
    second = run_embedding(data_dir, embedder, "acme/tiny", workers=1)

    assert first.failed == 1 and first.embedded == 3
    assert second.failed == 0 and second.embedded == 0
    failed = pl.read_parquet(embedding_path(data_dir, "acme/tiny") / "failed.parquet")
    assert failed["image_id"].to_list() == [9]


def test_embedding_takes_the_data_lock(data_dir: Path, combined_model: Path) -> None:
    from pipeline.download import lock_path
    from pipeline.files import LockedError, exclusive_lock

    embedder = Embedder.load(combined_model, providers=["CPUExecutionProvider"])
    with exclusive_lock(lock_path(data_dir)), pytest.raises(LockedError):
        run_embedding(data_dir, embedder, "acme/tiny", workers=1)


def test_load_embeddings_of_an_empty_dir_has_the_embedding_column(tmp_path: Path) -> None:
    assert "embedding" in load_embeddings(tmp_path).columns
