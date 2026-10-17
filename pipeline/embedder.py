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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

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


@dataclass(frozen=True)
class Preprocessor:
    height: int
    width: int
    shortest_then_crop: bool  # Jina: resize shortest side, center crop; HF: resize to (h, w)
    resample: int
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Preprocessor:
        size = cfg.get("size", 224)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                height = width = int(size["shortest_edge"])
                crop = True
            else:
                height, width = int(size["height"]), int(size["width"])
                crop = False
        else:
            height = width = int(size)
            crop = cfg.get("resize_mode", "shortest") == "shortest"
        if "crop_size" in cfg and cfg.get("do_center_crop"):
            crop_size = cfg["crop_size"]
            height, width = (
                (int(crop_size["height"]), int(crop_size["width"]))
                if isinstance(crop_size, dict)
                else (int(crop_size), int(crop_size))
            )
            crop = True
        resample = cfg.get("resample", cfg.get("interpolation", "bicubic"))
        if isinstance(resample, str):
            resample = RESAMPLE.get(resample, 3)
        mean = cfg.get("image_mean", cfg.get("mean", [0.5, 0.5, 0.5]))
        std = cfg.get("image_std", cfg.get("std", [0.5, 0.5, 0.5]))
        return cls(
            height,
            width,
            crop,
            int(resample),
            np.asarray(mean, dtype=np.float32).reshape(3, 1, 1),
            np.asarray(std, dtype=np.float32).reshape(3, 1, 1),
        )

    def __call__(self, image_bytes: bytes) -> np.ndarray:
        """Encoded image → normalized float32 array (3, height, width)."""
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.draft("RGB", (self.width * 2, self.height * 2))
            rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        if self.shortest_then_crop:
            scale = max(self.width / rgb.width, self.height / rgb.height)
            size = (
                max(self.width, round(rgb.width * scale)),
                max(self.height, round(rgb.height * scale)),
            )
            rgb = rgb.resize(size, resample=self.resample)
            left, top = (rgb.width - self.width) // 2, (rgb.height - self.height) // 2
            rgb = rgb.crop((left, top, left + self.width, top + self.height))
        else:
            rgb = rgb.resize((self.width, self.height), resample=self.resample)
        pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        normalized: np.ndarray = (pixels - self.mean) / self.std
        return normalized


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

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Texts → L2-normalized (n, dim). With a combined model this also runs the vision tower
        on blank images, so it costs about as much as embedding images."""
        ids = np.asarray([e.ids for e in self.tokenizer.encode_batch(list(texts))], np.int64)
        return self._batched(ids, self._run_text_batch)

    def _batched(self, inputs: np.ndarray, run: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((0, getattr(self, "dim", 0)), np.float32)
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
            if "Failed to allocate" not in str(exc) or len(batch) == 1:
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
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    special = model_dir / "special_tokens_map.json"
    pad_token = None
    if special.exists():
        value = json.loads(special.read_text()).get("pad_token")
        pad_token = value.get("content") if isinstance(value, dict) else value
    pad_id = tokenizer.token_to_id(pad_token) if pad_token else None
    pad_id = pad_id if pad_id is not None else 0
    config = model_dir / "config.json"
    fixed = None
    if config.exists():
        fixed = json.loads(config.read_text()).get("text_config", {}).get("max_position_embeddings")
    if fixed:  # SigLIP-style text towers expect exactly this many tokens
        tokenizer.enable_truncation(int(fixed))
        tokenizer.enable_padding(length=int(fixed), pad_id=pad_id, pad_token=pad_token or "")
    else:
        tokenizer.enable_truncation(DEFAULT_TEXT_LENGTH)
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token or "")
    return tokenizer
