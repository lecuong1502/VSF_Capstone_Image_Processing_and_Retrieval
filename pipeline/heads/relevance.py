"""Relevance features without domain knowledge (spec §4.1):

1. `s_text`: cosine(image, text embedding of "ảnh {entity_name}, {leaf_label}").
2. `s_proto`: cosine(image, centroid of the entity's accepted images), leaving the image itself
   out; entities with at most `min_proto` accepted images fall back to `s_text`.
3. `neighbor_rank`: rank of the paired entity when the image is compared with every entity
   centroid (1 = closest), null when the entity has no other accepted image;
   `suggested_entity_id` is the closest entity – a different one flags a possible mislabel.

Every centroid the image belongs to is computed without it, and whether a pair falls back or has
a rank depends only on the entity's number of accepted images – never on the pair's own label,
which would leak it into the features.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import polars as pl

MIN_PROTO = 5
CHUNK = 4096


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v / max(float(np.linalg.norm(v)), 1e-12))


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
    entity_ids = pairs["entity_id"].to_list()
    accepted = (pairs["value"] == "yes").to_numpy()

    entities = sorted({e for e, yes in zip(entity_ids, accepted, strict=True) if yes})
    position = {e: k for k, e in enumerate(entities)}
    sums = np.zeros((len(entities), vectors.shape[1]), np.float64)
    counts = np.zeros(len(entities), np.int64)
    member_of: dict[int, set[int]] = defaultdict(set)  # image row -> centroids it is part of
    for row, entity, yes in zip(rows, entity_ids, accepted, strict=True):
        if yes and position[entity] not in member_of[int(row)]:
            sums[position[entity]] += vectors[row]
            counts[position[entity]] += 1
            member_of[int(row)].add(position[entity])
    centroids = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)

    own = np.array([position.get(e, -1) for e in entity_ids])
    s_text = np.array(
        [
            float(vectors[r] @ text[e]) if e in text else np.nan
            for r, e in zip(rows, entity_ids, strict=True)
        ]
    )
    s_proto = s_text.copy()
    fallback = np.ones(len(rows), bool)
    ranks = np.full(len(rows), -1, np.int64)
    suggested: list[str | None] = [None] * len(rows)
    for start in range(0, len(rows), CHUNK):
        chunk = slice(start, start + CHUNK)
        vecs = vectors[rows[chunk]].astype(np.float64)
        sims = vecs @ centroids.T if entities else np.zeros((len(vecs), 0))
        for k, (row, vec) in enumerate(zip(rows[chunk], vecs, strict=True)):
            i = start + k
            # leave the image out of every centroid it belongs to
            for e in member_of.get(int(row), ()):
                sims[k, e] = float(vec @ _unit(sums[e] - vec)) if counts[e] > 1 else -np.inf
            e = own[i]
            mine = -np.inf
            if e >= 0 and counts[e] > 1:
                mine = sims[k, e]
                if counts[e] > min_proto:
                    s_proto[i] = mine
                    fallback[i] = False
                sims[k, e] = -np.inf
                ranks[i] = 1 + int(np.sum(sims[k] > mine))
            elif e >= 0:
                sims[k, e] = -np.inf
            best = int(np.argmax(sims[k])) if entities else -1
            suggested[i] = entity_ids[i] if best < 0 or sims[k, best] <= mine else entities[best]
    return pl.DataFrame(
        {
            "image_id": pairs["image_id"],
            "entity_id": pairs["entity_id"],
            "s_text": s_text,
            "s_proto": s_proto,
            "proto_fallback": fallback,
            "neighbor_rank": pl.Series(ranks).replace(-1, None),
            "suggested_entity_id": suggested,
        }
    )
