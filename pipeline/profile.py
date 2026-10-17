"""Data profiling report (spec §2.2, §10 task 1): `verify` rates by declared size, image source,
query_type, branch and priority, and entities with few relevant images.

Rates are computed over labeled (image, entity) pairs, excluding `unlabeled` and conflicting
pairs, with 95% Wilson intervals so small groups are not over-read. Rule thresholds for the
cascade are meant to be chosen from these numbers, not guessed.

Usage: python -m pipeline.profile [--ingest-dir DIR] [--out reports/data_profile.html]
"""

from __future__ import annotations

import argparse
import html
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pipeline import config

log = logging.getLogger(__name__)

SIZE_BUCKETS = ((300, "<300"), (500, "300–499"), (800, "500–799"), (1200, "800–1199"))
SIZE_ORDER = [label for _, label in SIZE_BUCKETS] + ["≥1200", "unknown"]
FEW_YES_THRESHOLD = 5  # spec §2.1: entities with < 5 relevant images fall back to text-only
TOP_SOURCES = 25
MIN_SOURCE_N = 100  # only rank sources with enough labeled pairs for a stable rate
Z_95 = 1.959964


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def size_bucket(width: int | None, height: int | None) -> str:
    """Bucket by the declared short side (what matters for thumbnails and embeddings)."""
    if width is None or height is None:
        return "unknown"
    short = min(width, height)
    for upper, label in SIZE_BUCKETS:
        if short < upper:
            return label
    return "≥1200"


def verify_rates(labeled: pl.DataFrame, group: str) -> pl.DataFrame:
    """n, yes count, yes rate and Wilson interval per `group`, over clean yes/no labels."""
    counts = (
        labeled.filter(pl.col("value").is_in(["yes", "no"]) & ~pl.col("label_conflict"))
        .group_by(pl.col(group).fill_null("unknown"))
        .agg(pl.len().alias("n"), (pl.col("value") == "yes").sum().alias("yes"))
    )
    intervals = [wilson(y, n) for y, n in zip(counts["yes"], counts["n"], strict=True)]
    return counts.with_columns(
        (pl.col("yes") / pl.col("n")).alias("yes_rate"),
        pl.Series("ci_low", [lo for lo, _ in intervals]),
        pl.Series("ci_high", [hi for _, hi in intervals]),
    ).sort("n", descending=True)


@dataclass(frozen=True)
class Profile:
    overview: dict[str, int]
    rates: dict[str, pl.DataFrame]
    few_yes: pl.DataFrame


def build_profile(images: pl.DataFrame, labels: pl.DataFrame, entities: pl.DataFrame) -> Profile:
    sized = images.with_columns(
        pl.struct("declared_w", "declared_h")
        .map_elements(
            lambda s: size_bucket(s["declared_w"], s["declared_h"]), return_dtype=pl.String
        )
        .alias("size")
    )
    labeled = labels.join(
        sized.select("image_id", "size", "image_source"), on="image_id", how="left"
    ).join(entities.select("entity_id", "branch", "priority"), on="entity_id", how="left")

    rates = {
        "size": _order(verify_rates(labeled, "size"), "size", SIZE_ORDER),
        "source": verify_rates(labeled, "image_source"),
        "query_type": verify_rates(labeled, "query_type"),
        "branch": verify_rates(labeled, "branch").sort("branch"),
        "priority": verify_rates(labeled, "priority").sort("priority"),
    }
    clean = labels.filter(pl.col("value").is_in(["yes", "no"]) & ~pl.col("label_conflict"))
    per_entity = clean.group_by("entity_id").agg(pl.len().alias("n_labeled"))
    few_yes = (
        entities.filter(pl.col("n_yes_images") < FEW_YES_THRESHOLD)
        .join(per_entity, on="entity_id", how="left")
        .select(
            "entity_id",
            "entity_name",
            "branch",
            "priority",
            "n_yes_images",
            pl.col("n_labeled").fill_null(0),
        )
        .sort("priority", "n_yes_images", "entity_name")
    )
    value_counts = dict(labels.group_by("value").len().rows())
    overview = {
        "images": images.height,
        "entities": entities.height,
        "pairs": labels.height,
        "labeled_pairs": labels.height - value_counts.get("unlabeled", 0),
        "yes": value_counts.get("yes", 0),
        "no": value_counts.get("no", 0),
        "unlabeled": value_counts.get("unlabeled", 0),
        "conflicts": int(labels["label_conflict"].sum()),
        "images_multi_entity": int(
            labels.group_by("image_id").len().filter(pl.col("len") > 1).height
        ),
        "entities_few_yes": few_yes.height,
    }
    return Profile(overview=overview, rates=rates, few_yes=few_yes)


def _order(frame: pl.DataFrame, column: str, order: list[str]) -> pl.DataFrame:
    rank = {value: i for i, value in enumerate(order)}
    return (
        frame.with_columns(pl.col(column).replace_strict(rank, default=len(order)).alias("_o"))
        .sort("_o")
        .drop("_o")
    )


# --- HTML -------------------------------------------------------------------------------------

CSS = """
.viz-root {
  color-scheme: light;
  --surface-0: #f4f4f2; --surface-1: #fcfcfb; --rule: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --series-1: #2a78d6; --whisker: #0b0b0b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --rule: #2e2e2c;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --series-1: #3987e5; --whisker: #f0efec;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --rule: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --series-1: #3987e5; --whisker: #f0efec;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); }
.viz-root { font: 14px/1.45 system-ui, sans-serif; color: var(--text-primary);
  background: var(--surface-0); padding: 24px 16px 48px; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 4px; }
p.note, .meta { color: var(--text-secondary); margin: 0 0 12px; }
section { background: var(--surface-1); border: 1px solid var(--rule); border-radius: 8px;
  padding: 16px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .k { color: var(--text-secondary); font-size: 12px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 500; color: var(--text-secondary); font-size: 12px;
  border-bottom: 1px solid var(--rule); padding: 4px 8px 4px 0; }
td { padding: 4px 8px 4px 0; border-bottom: 1px solid var(--rule); vertical-align: middle; }
td.num { text-align: right; white-space: nowrap; }
tr.bar-row:hover td { background: color-mix(in srgb, var(--series-1) 8%, transparent); }
.label { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.track { position: relative; height: 14px; min-width: 160px; }
.bar { position: absolute; left: 0; top: 1px; height: 12px; background: var(--series-1);
  border-radius: 0 4px 4px 0; }
.ci { position: absolute; top: 6px; height: 2px; background: var(--whisker); opacity: .55; }
.wrap { overflow-x: auto; }
"""


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _rate_table(frame: pl.DataFrame, key: str, title: str, note: str) -> str:
    rows = []
    for r in frame.iter_rows(named=True):
        label = html.escape(str(r[key]))
        tip = (
            f"{label}: {_pct(r['yes_rate'])} yes ({r['yes']}/{r['n']}), "
            f"95% CI {_pct(r['ci_low'])}–{_pct(r['ci_high'])}"
        )
        rows.append(
            f'<tr class="bar-row" title="{tip}"><td class="label">{label}</td>'
            f'<td><div class="track"><div class="bar" style="width:{100 * r["yes_rate"]:.2f}%">'
            f'</div><div class="ci" style="left:{100 * r["ci_low"]:.2f}%;'
            f'width:{100 * (r["ci_high"] - r["ci_low"]):.2f}%"></div></div></td>'
            f'<td class="num">{_pct(r["yes_rate"])}</td>'
            f'<td class="num">{_pct(r["ci_low"])}–{_pct(r["ci_high"])}</td>'
            f'<td class="num">{r["n"]:,}</td></tr>'
        )
    return (
        f"<section><h2>{html.escape(title)}</h2><p class=note>{html.escape(note)}</p>"
        '<div class="wrap"><table><thead><tr><th>Group</th><th>yes rate (bar) · 95% CI (line)</th>'
        "<th>yes rate</th><th>95% CI</th><th>n</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def render_html(profile: Profile) -> str:
    o = profile.overview
    tiles = [
        ("Images (unique URLs)", o["images"]),
        ("Entities", o["entities"]),
        ("(image, entity) pairs", o["pairs"]),
        ("Labeled pairs", o["labeled_pairs"]),
        ("yes", o["yes"]),
        ("no", o["no"]),
        ("Unlabeled", o["unlabeled"]),
        ("Conflicting pairs", o["conflicts"]),
        ("Images in > 1 entity", o["images_multi_entity"]),
        (f"Entities with < {FEW_YES_THRESHOLD} yes", o["entities_few_yes"]),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v:,}</div><div class="k">{html.escape(k)}</div></div>'
        for k, v in tiles
    )
    sources = profile.rates["source"].head(TOP_SOURCES)
    rates_note = "Share of yes among labeled pairs (unlabeled and conflicting pairs excluded)."
    sections = [
        _rate_table(
            profile.rates["size"],
            "size",
            "Declared size (short side, px)",
            rates_note + " Declared size from the crawl; re-measured after download (#9).",
        ),
        _rate_table(
            sources,
            "image_source",
            f"Image source (top {len(sources)} of {profile.rates['source'].height} by volume)",
            rates_note,
        ),
        _rate_table(
            profile.rates["source"]
            .filter(pl.col("n") >= MIN_SOURCE_N)
            .sort("yes_rate", "image_source")
            .head(TOP_SOURCES),
            "image_source",
            f"Lowest yes rate sources (n ≥ {MIN_SOURCE_N})",
            rates_note + " Candidates for URL/domain rules in the cascade's first stage.",
        ),
        _rate_table(profile.rates["query_type"], "query_type", "Query type", rates_note),
        _rate_table(profile.rates["branch"], "branch", "Branch", rates_note),
        _rate_table(profile.rates["priority"], "priority", "Priority", rates_note),
    ]
    few = profile.few_yes
    by_group = (
        few.group_by("branch", "priority").len().sort("branch", "priority").iter_rows(named=True)
    )
    group_rows = "".join(
        f"<tr><td>{html.escape(str(r['branch']))}</td><td>{html.escape(str(r['priority']))}</td>"
        f'<td class="num">{r["len"]:,}</td></tr>'
        for r in by_group
    )
    entity_rows = "".join(
        f'<tr><td class="label">{html.escape(str(r["entity_name"]))}</td>'
        f"<td>{html.escape(str(r['branch']))}</td><td>{html.escape(str(r['priority']))}</td>"
        f'<td class="num">{r["n_yes_images"]}</td><td class="num">{r["n_labeled"]}</td></tr>'
        for r in few.head(50).iter_rows(named=True)
    )
    sections.append(
        f"<section><h2>Entities with fewer than {FEW_YES_THRESHOLD} relevant images</h2>"
        f"<p class=note>{few.height:,} entities fall back to text-only matching (spec §2.1). "
        "Counts by branch × priority, then the first 50 by priority.</p>"
        '<div class="wrap"><table><thead><tr><th>Branch</th><th>Priority</th><th>Entities</th>'
        f"</tr></thead><tbody>{group_rows}</tbody></table></div><br>"
        '<div class="wrap"><table><thead><tr><th>Entity</th><th>Branch</th><th>Priority</th>'
        "<th>yes images</th><th>labeled pairs</th></tr></thead>"
        f"<tbody>{entity_rows}</tbody></table></div></section>"
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Data Profile</title><style>{CSS}</style></head>"
        '<body><div class="viz-root"><main>'
        "<h1>Data profile – verified travel images</h1>"
        f'<p class="meta">Generated {generated} by <code>python -m pipeline.profile</code> from '
        "the ingest tables. Dead-link rate is added after download (#9).</p>"
        f'<section><div class="tiles">{tiles_html}</div></section>'
        + "".join(sections)
        + "</main></div></body></html>"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m pipeline.profile")
    parser.add_argument("--ingest-dir", type=Path, default=config.data_dir() / "ingest")
    parser.add_argument(
        "--out", type=Path, default=config.REPO_ROOT / "reports" / "data_profile.html"
    )
    args = parser.parse_args()

    tables = {n: pl.read_parquet(args.ingest_dir / f"{n}.parquet") for n in ("images", "labels")}
    entities = pl.read_parquet(args.ingest_dir / "entities.parquet")
    profile = build_profile(tables["images"], tables["labels"], entities)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(profile), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
