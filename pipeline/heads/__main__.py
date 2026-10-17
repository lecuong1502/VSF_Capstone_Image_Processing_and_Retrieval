"""`python -m pipeline.heads [--seed 0] [--n-boot 1000]` – train, evaluate and apply every
moderation head on EMBED_MODEL's embeddings (text prompts are embedded on the GPU)."""

from __future__ import annotations

import argparse
import logging

import polars as pl

from pipeline import config
from pipeline.embedder import Embedder, resolve_model
from pipeline.heads.train import train_heads

log = logging.getLogger("pipeline.heads")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.heads")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    args = parser.parse_args()

    model = config.embed_model()
    embedder = Embedder.load(resolve_model(model, config.models_dir()))
    report = train_heads(
        config.data_dir(),
        model,
        lambda texts: embedder.embed_texts(texts),
        seed=args.seed,
        n_boot=args.n_boot,
    )
    rows = [
        {
            "head": name,
            "truth": r["truth"],
            "f1@0.5": r.get("test", {}).get("f1"),
            "precision@reject": r.get("reject", {}).get("precision"),
            "recall@reject": r.get("reject", {}).get("recall"),
            "pr_auc": r.get("pr_auc"),
            "ece": r.get("ece"),
            "recall_kept@low": r.get("recall_kept_by_low"),
            "escalation": r.get("escalation_share"),
        }
        for name, r in report["heads"].items()
    ]
    with pl.Config(
        tbl_hide_dataframe_shape=True, tbl_rows=-1, fmt_str_lengths=40, tbl_width_chars=200
    ):
        log.info("%s\n%s", report["model_version"], pl.DataFrame(rows, strict=False))
    log.info("Images escalated by any head: %.2f%%", 100 * report["escalation_rate"])


if __name__ == "__main__":
    main()
