"""Relevance features without domain knowledge (spec §4.1):

1. `s_text`: cosine(image, text embedding of "ảnh {entity_name}, {leaf_label}").
2. `s_proto`: cosine(image, centroid of the entity's accepted images), leaving the image itself
   out; entities with fewer than `min_proto` other accepted images fall back to `s_text`.
3. `neighbor_rank`: rank of the paired entity when the image is compared with every entity
   centroid (1 = closest); `suggested_entity_id` is the closest one – a different entity flags a
   possible mislabel.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

MIN_PROTO = 5
CHUNK = 4096


def relevance_features(
    vectors: np.ndarray,
    index: Mapping[int, int],
    pairs: pl.DataFrame,
    text: Mapping[str, np.ndarray],
    *,
    min_proto: int = MIN_PROTO,
) -> pl.DataFrame:
    """One row per (image_id, entity_id) of `pairs` whose image has a vector. `value == "yes"`
    pairs build the entity centroids; `vectors` rows are L2-normalized."""
    pairs = pairs.filter(pl.col("image_id").is_in(list(index)))
    rows = np.array([index[i] for i in pairs["image_id"]], dtype=np.int64)
    accepted = (pairs["value"] == "yes").to_numpy()

    entities = sorted(set(pairs.filter(pl.col("value") == "yes")["entity_id"]))
    position = {e: k for k, e in enumerate(entities)}
    sums = np.zeros((len(entities), vectors.shape[1]), np.float64)
    counts = np.zeros(len(entities), np.int64)
    for row, entity, yes in zip(rows, pairs["entity_id"], accepted, strict=True):
        if yes:
            sums[position[entity]] += vectors[row]
            counts[position[entity]] += 1
    centroids = sums / np.maximum(counts, 1)[:, None]
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    entity_ids = pairs["entity_id"].to_list()
    own = np.array([position.get(e, -1) for e in entity_ids])
    s_text = np.array(
        [
            float(vectors[r] @ text[e]) if e in text else np.nan
            for r, e in zip(rows, entity_ids, strict=True)
        ]
    )
    s_proto = s_text.copy()
    fallback = np.ones(len(rows), bool)
    ranks = np.zeros(len(rows), np.int64)
    suggested: list[str | None] = [None] * len(rows)
    for start in range(0, len(rows), CHUNK):
        chunk = slice(start, start + CHUNK)
        v = vectors[rows[chunk]].astype(np.float64)
        sims = v @ centroids.T if entities else np.zeros((len(v), 0))
        for k, (vec, e, yes) in enumerate(zip(v, own[chunk], accepted[chunk], strict=True)):
            i = start + k
            if e >= 0:
                n = counts[e] - int(yes)
                if n >= min_proto:
                    loo = sums[e] - vec * int(yes)
                    s_proto[i] = float(vec @ loo / max(np.linalg.norm(loo), 1e-12))
                    fallback[i] = False
                sims[k, e] = -np.inf  # compared through s_proto, never its own full centroid
            mine = s_proto[i]
            ranks[i] = 1 + int(np.sum(sims[k] > mine))
            best = int(np.argmax(sims[k])) if entities else -1
            suggested[i] = entity_ids[i] if best < 0 or sims[k, best] <= mine else entities[best]
    return pl.DataFrame(
        {
            "image_id": pairs["image_id"],
            "entity_id": pairs["entity_id"],
            "s_text": s_text,
            "s_proto": s_proto,
            "proto_fallback": fallback,
            "neighbor_rank": ranks,
            "suggested_entity_id": suggested,
        }
    )
