"""Image/text embedding with any CLIP-style model that ships FP16 ONNX weights on Hugging Face.

`EMBED_MODEL` names a Hugging Face repo; switching models needs no code change:
- combined layout (e.g. `jinaai/jina-clip-v2`): `onnx/model_fp16.onnx` takes both `input_ids`
  and `pixel_values` and returns text and image embeddings;
- split layout (e.g. `onnx-community/siglip2-base-patch16-256-ONNX`):
  `onnx/vision_model_fp16.onnx` + `onnx/text_model_fp16.onnx` returning `pooler_output`.
Preprocessing follows the repo's `preprocessor_config.json` (Jina's "shortest side + center crop"
or the Hugging Face fixed-size style); tokenization uses its `tokenizer.json`. Every embedding is
returned L2-normalized as float32.
"""

from __future__ import annotations

import io
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer, normalizers

log = logging.getLogger(__name__)

COMBINED = "onnx/model_fp16.onnx"
VISION = "onnx/vision_model_fp16.onnx"
TEXT = "onnx/text_model_fp16.onnx"
IMAGE_OUTPUTS = ("l2norm_image_embeddings", "image_embeds", "image_embeddings", "pooler_output")
TEXT_OUTPUTS = ("l2norm_text_embeddings", "text_embeds", "text_embeddings", "pooler_output")
DEFAULT_TEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 8  # jina-clip-v2 FP16 at 512 px: batch 16 does not fit in 6 GB
# Pillow resampling filters by name (Jina configs) and by Hugging Face's integer codes.
RESAMPLE = {"nearest": 0, "bilinear": 2, "bicubic": 3, "lanczos": 1}


def model_dir_name(repo: str) -> str:
    """Local directory name for a Hugging Face repo id."""
    return repo.replace("/", "--")


def resolve_model(repo: str, models_dir: Path) -> Path:
    """Local copy of `repo`'s FP16 ONNX weights and configs, downloaded on first use."""
    from huggingface_hub import list_repo_files, snapshot_download

    local = models_dir / model_dir_name(repo)
    if (local / COMBINED).exists() or (local / VISION).exists():
        return local
    files = set(list_repo_files(repo))
    weights = [VISION, TEXT] if VISION in files else [COMBINED]
    missing = [w for w in weights if w not in files]
    if missing:
        raise FileNotFoundError(f"{repo} has no FP16 ONNX export ({', '.join(missing)})")
    log.info("Downloading %s (%s) to %s", repo, ", ".join(weights), local)
    snapshot_download(
        repo,
        local_dir=local,
        allow_patterns=[*weights, "*.json", "tokenizer*"],
    )
    return local


@dataclass(frozen=True, eq=False)
class Preprocessor:
    """Image preprocessing from a model's preprocessor_config.json.

    `prepare` (decode, RGB, resize, crop → uint8 HWC) runs in worker processes; `normalize`
    (rescale, mean/std → float32 NCHW) runs on whole batches, so workers ship 4× less data.
    """

    height: int
    width: int
    resize_short: int | None  # resize the shortest side to this, then center crop; else squash
    resample: int
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Preprocessor:
        size = cfg.get("size", 224)
        resize_short: int | None
        if isinstance(size, dict) and "shortest_edge" in size:
            resize_short = int(size["shortest_edge"])
            height = width = resize_short
        elif isinstance(size, dict):
            height, width = int(size["height"]), int(size["width"])
            resize_short = None
        else:  # Jina: an int plus resize_mode="shortest"
            height = width = int(size)
            resize_short = height if cfg.get("resize_mode", "shortest") == "shortest" else None
        if cfg.get("do_center_crop") and "crop_size" in cfg:
            crop = cfg["crop_size"]
            height, width = (
                (int(crop["height"]), int(crop["width"]))
                if isinstance(crop, dict)
                else (int(crop), int(crop))
            )
        resample = cfg.get("resample", cfg.get("interpolation", "bicubic"))
        if isinstance(resample, str):
            resample = RESAMPLE.get(resample, 3)
        mean = cfg.get("image_mean", cfg.get("mean", [0.5, 0.5, 0.5]))
        std = cfg.get("image_std", cfg.get("std", [0.5, 0.5, 0.5]))
        return cls(
            height,
            width,
            resize_short,
            int(resample),
            np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1),
            np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1),
        )

    def prepare(self, image_bytes: bytes) -> np.ndarray:
        """Encoded image → uint8 (height, width, 3), resized and cropped like the reference."""
        with Image.open(io.BytesIO(image_bytes)) as image:
            # JPEG DCT downscale only while it stays ≥ 2× the target, so quality is unaffected.
            image.draft("RGB", (self.width * 2, self.height * 2))
            rgb = image.convert("RGB")
        if self.resize_short is not None:
            scale = self.resize_short / min(rgb.width, rgb.height)
            size = (
                max(self.width, round(rgb.width * scale)),
                max(self.height, round(rgb.height * scale)),
            )
            rgb = rgb.resize(size, resample=self.resample)
            left, top = (rgb.width - self.width) // 2, (rgb.height - self.height) // 2
            rgb = rgb.crop((left, top, left + self.width, top + self.height))
        else:
            rgb = rgb.resize((self.width, self.height), resample=self.resample)
        return np.asarray(rgb, dtype=np.uint8)

    def normalize(self, prepared: np.ndarray) -> np.ndarray:
        """uint8 (n, H, W, 3) → normalized float32 (n, 3, H, W)."""
        pixels = prepared.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        normalized: np.ndarray = (pixels - self.mean) / self.std
        return normalized

    def __call__(self, image_bytes: bytes) -> np.ndarray:
        """Encoded image → normalized float32 (3, height, width)."""
        single: np.ndarray = self.normalize(self.prepare(image_bytes)[None])[0]
        return single


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit: np.ndarray = x / np.maximum(norms, 1e-12)
    return unit


def _pick_output(session: ort.InferenceSession, names: Sequence[str]) -> str:
    available = [o.name for o in session.get_outputs()]
    for name in names:
        if name in available:
            return name
    raise ValueError(f"none of {names} in model outputs {available}")


def is_out_of_memory(exc: BaseException) -> bool:
    """onnxruntime reports GPU allocation failures as a generic Fail with this message."""
    return "Failed to allocate" in str(exc)


def default_providers() -> list[Any]:
    ort.preload_dlls()
    cuda = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
    return [cuda, "CPUExecutionProvider"]


class Embedder:
    """Loaded ONNX model(s), preprocessor and tokenizer for one EMBED_MODEL."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        tokenizer: Tokenizer,
        *,
        combined: ort.InferenceSession | None = None,
        vision: ort.InferenceSession | None = None,
        text: ort.InferenceSession | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self._combined, self._vision, self._text = combined, vision, text
        image_session = combined or vision
        text_session = combined or text
        if image_session is None or text_session is None:
            raise ValueError("need a combined model or both vision and text models")
        self._image_out = _pick_output(image_session, IMAGE_OUTPUTS)
        self._text_out = _pick_output(text_session, TEXT_OUTPUTS)
        self.dim = self._probe_dim()  # also warms the session up

    @classmethod
    def load(
        cls,
        model_dir: Path,
        providers: list[Any] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Embedder:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = providers or default_providers()

        def session(path: str) -> ort.InferenceSession:
            return ort.InferenceSession(str(model_dir / path), options, providers=providers)

        pre_cfg = json.loads((model_dir / "preprocessor_config.json").read_text())
        tokenizer = _load_tokenizer(model_dir)
        if (model_dir / COMBINED).exists():
            return cls(
                Preprocessor.from_config(pre_cfg),
                tokenizer,
                combined=session(COMBINED),
                batch_size=batch_size,
            )
        return cls(
            Preprocessor.from_config(pre_cfg),
            tokenizer,
            vision=session(VISION),
            text=session(TEXT),
            batch_size=batch_size,
        )

    def _probe_dim(self) -> int:
        dummy = np.zeros((1, 3, self.preprocessor.height, self.preprocessor.width), np.float32)
        return int(self.run_pixels(dummy).shape[1])

    def run_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Preprocessed (n, 3, H, W) images → L2-normalized (n, dim), in batches of batch_size."""
        return self._batched(pixels, self._run_image_batch)

    def forward_images(self, pixels: np.ndarray) -> np.ndarray:
        """One forward pass of exactly this batch (no splitting) – for benchmarking."""
        return _normalize(self._run_image_batch(pixels))

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        return self.run_pixels(np.stack([self.preprocessor(b) for b in images]))

    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(list(texts))]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Texts → L2-normalized (n, dim). Texts are batched only with texts of the same token
        length, so no padding is added beyond what the model's tokenizer itself prescribes (the
        combined ONNX graph takes no attention mask). With a combined model the vision tower also
        runs on blank images, so this costs about as much as embedding images."""
        ids = self.tokenize(texts)
        by_length: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(ids):
            by_length[len(row)].append(i)
        out = np.zeros((len(ids), self.dim), np.float32)
        for rows in by_length.values():
            batch = np.asarray([ids[i] for i in rows], np.int64)
            out[rows] = self._batched(batch, self._run_text_batch)
        return out

    def _batched(self, inputs: np.ndarray, run: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        parts = [
            self._split_on_oom(inputs[i : i + self.batch_size], run)
            for i in range(0, len(inputs), self.batch_size)
        ]
        return _normalize(np.concatenate(parts))

    def _split_on_oom(
        self, batch: np.ndarray, run: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Run a batch, halving it when the GPU runs out of memory."""
        try:
            return run(batch)
        except Exception as exc:  # onnxruntime raises its own Fail type on allocation errors
            if not is_out_of_memory(exc) or len(batch) == 1:
                raise
            log.warning("GPU out of memory at batch %d, splitting", len(batch))
            half = len(batch) // 2
            return np.concatenate(
                [self._split_on_oom(batch[:half], run), self._split_on_oom(batch[half:], run)]
            )

    def _run_image_batch(self, pixels: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            ids = np.zeros((len(pixels), 1), dtype=np.int64)  # text tower input is required
            feed = {"pixel_values": pixels, "input_ids": ids}
            out: np.ndarray = self._combined.run([self._image_out], feed)[0]
            return out
        assert self._vision is not None
        out = self._vision.run([self._image_out], {"pixel_values": pixels})[0]
        return out

    def _run_text_batch(self, ids: np.ndarray) -> np.ndarray:
        if self._combined is not None:
            shape = (len(ids), 3, self.preprocessor.height, self.preprocessor.width)
            feed = {"pixel_values": np.zeros(shape, np.float32), "input_ids": ids}
            out: np.ndarray = self._combined.run([self._text_out], feed)[0]
            return out
        assert self._text is not None
        out = self._text.run([self._text_out], {"input_ids": ids})[0]
        return out


def _load_tokenizer(model_dir: Path) -> Tokenizer:
    """tokenizer.json as shipped (SigLIP2's already pads to a fixed 64 tokens), plus lowercasing
    when tokenizer_config.json asks for it and a truncation limit if none is set."""
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    config_path = model_dir / "tokenizer_config.json"
    tok_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if tok_config.get("do_lower_case"):
        lowercase = normalizers.Lowercase()
        existing = tokenizer.normalizer
        tokenizer.normalizer = (
            normalizers.Sequence([lowercase, existing]) if existing is not None else lowercase
        )
    if tokenizer.truncation is None:
        limit = tok_config.get("model_max_length", DEFAULT_TEXT_LENGTH)
        tokenizer.enable_truncation(int(min(limit, DEFAULT_TEXT_LENGTH)))
    return tokenizer
