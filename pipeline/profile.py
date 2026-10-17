"""Data profiling report (spec §2.2, §10 task 1): `verify` rates by declared size, image source,
query_type, branch and priority, and entities with few relevant images.

Rates are computed over labeled (image, entity) pairs, excluding `unlabeled` and conflicting
pairs, with 95% Wilson intervals so small groups are not over-read. Rule thresholds for the
cascade are meant to be chosen from these numbers, not guessed.

The dead-link rate (also asked for in §2.2) needs the downloads and is added to this report by
the download step (#9).

Usage: python -m pipeline.profile [--ingest-dir DIR] [--out reports/data_profile.html]
"""

from __future__ import annotations

import argparse
import html
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pipeline import config

log = logging.getLogger(__name__)

SIZE_BUCKETS = ((300, "<300"), (500, "300–499"), (800, "500–799"), (1200, "800–1199"))
SIZE_ORDER = [label for _, label in SIZE_BUCKETS] + ["≥1200", "unknown"]
# width / height
ASPECT_BUCKETS = ((0.5, "tall <1:2"), (0.8, "portrait"), (1.25, "square-ish"), (2.0, "landscape"))
ASPECT_ORDER = [label for _, label in ASPECT_BUCKETS] + ["wide >2:1", "unknown"]
FEW_YES_THRESHOLD = 5  # spec §2.1: entities with < 5 relevant images fall back to text-only
YES_PER_ENTITY_BUCKETS = ((1, "0"), (5, "1–4"), (10, "5–9"), (20, "10–19"), (50, "20–49"))
YES_PER_ENTITY_ORDER = [label for _, label in YES_PER_ENTITY_BUCKETS] + ["≥50"]
TOP_SOURCES = 25
MIN_SOURCE_N = 100  # only rank domains with enough labeled pairs for a stable rate
Z_95 = 1.959964

# A label usable as ground truth: yes/no and not conflicting between exports.
CLEAN_LABEL = pl.col("value").is_in(["yes", "no"]) & ~pl.col("label_conflict")


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bucket(value: float, buckets: tuple[tuple[float, str], ...], last: str) -> str:
    for upper, label in buckets:
        if value < upper:
            return label
    return last


def size_bucket(width: int | None, height: int | None) -> str:
    """Bucket by the declared short side (what matters for thumbnails and embeddings)."""
    if width is None or height is None:
        return "unknown"
    return _bucket(min(width, height), SIZE_BUCKETS, "≥1200")


def aspect_bucket(width: int | None, height: int | None) -> str:
    """Bucket by width / height; extreme ratios are typical of banners and screenshots."""
    if not width or not height:
        return "unknown"
    return _bucket(width / height, ASPECT_BUCKETS, "wide >2:1")


def normalize_domain(source: str | None) -> str | None:
    """Merge `www.` variants and case so one site is one group."""
    if source is None:
        return None
    domain = source.strip().lower()
    return domain.removeprefix("www.")


def verify_rates(labeled: pl.DataFrame, group: str) -> pl.DataFrame:
    """n, yes count, yes rate and Wilson interval per `group`, over clean yes/no labels."""
    counts = (
        labeled.filter(CLEAN_LABEL)
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
    distributions: dict[str, pl.DataFrame]
    short_side_quantiles: dict[str, float]
    few_yes: pl.DataFrame


def _apply(fn: Callable[[int | None, int | None], str], *columns: str) -> pl.Expr:
    """Row-wise `fn` over `columns` (keeps the tested pure bucket functions the single source)."""
    return pl.struct(*columns).map_elements(
        lambda row: fn(*(row[c] for c in columns)), return_dtype=pl.String
    )


def build_profile(images: pl.DataFrame, labels: pl.DataFrame, entities: pl.DataFrame) -> Profile:
    images = images.with_columns(
        _apply(size_bucket, "declared_w", "declared_h").alias("size"),
        _apply(aspect_bucket, "declared_w", "declared_h").alias("aspect"),
        pl.col("image_source")
        .map_elements(normalize_domain, return_dtype=pl.String)
        .alias("domain"),
    )
    labeled = labels.join(
        images.select("image_id", "size", "aspect", "domain"), on="image_id", how="left"
    ).join(entities.select("entity_id", "branch", "priority"), on="entity_id", how="left")

    rates = {
        "size": _order(verify_rates(labeled, "size"), "size", SIZE_ORDER),
        "aspect": _order(verify_rates(labeled, "aspect"), "aspect", ASPECT_ORDER),
        "domain": verify_rates(labeled, "domain"),
        "query_type": verify_rates(labeled, "query_type"),
        "branch": verify_rates(labeled, "branch").sort("branch"),
        "priority": verify_rates(labeled, "priority").sort("priority"),
    }

    yes_bucket = (
        entities["n_yes_images"]
        .map_elements(lambda n: _bucket(n, YES_PER_ENTITY_BUCKETS, "≥50"), return_dtype=pl.String)
        .alias("bucket")
    )
    distributions = {
        "size": _counts(images["size"], "images", SIZE_ORDER),
        "aspect": _counts(images["aspect"], "images", ASPECT_ORDER),
        "yes_per_entity": _counts(yes_bucket, "entities", YES_PER_ENTITY_ORDER),
    }
    short_side = images.select(pl.min_horizontal("declared_w", "declared_h")).to_series()
    short_side_quantiles = {
        f"p{int(q * 100):02d}": float(short_side.quantile(q) or 0)
        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)
    }

    per_entity = labels.filter(CLEAN_LABEL).group_by("entity_id").agg(pl.len().alias("n_labeled"))
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
    return Profile(overview, rates, distributions, short_side_quantiles, few_yes)


def _counts(values: pl.Series, unit: str, order: list[str]) -> pl.DataFrame:
    """Count of `unit` per bucket, in `order`, including empty buckets."""
    counts = dict(values.value_counts().rows())
    return pl.DataFrame({"bucket": order, unit: [counts.get(b, 0) for b in order]})


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


def _count_table(frame: pl.DataFrame, unit: str, title: str, note: str) -> str:
    total = max(int(frame[unit].sum()), 1)
    peak = max(int(frame.select(pl.col(unit).max()).item() or 0), 1)
    rows = "".join(
        f'<tr class="bar-row" title="{html.escape(str(r["bucket"]))}: {r[unit]:,} {unit} '
        f'({_pct(r[unit] / total)})"><td class="label">{html.escape(str(r["bucket"]))}</td>'
        f'<td><div class="track"><div class="bar" style="width:{100 * r[unit] / peak:.2f}%">'
        f'</div></div></td><td class="num">{r[unit]:,}</td>'
        f'<td class="num">{_pct(r[unit] / total)}</td></tr>'
        for r in frame.iter_rows(named=True)
    )
    return (
        f"<section><h2>{html.escape(title)}</h2><p class=note>{html.escape(note)}</p>"
        f'<div class="wrap"><table><thead><tr><th>Bucket</th><th>{unit} (bar)</th>'
        f"<th>{unit}</th><th>share</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
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
    domains = profile.rates["domain"]
    rates_note = "Share of yes among labeled pairs (unlabeled and conflicting pairs excluded)."
    q = profile.short_side_quantiles
    quantiles = ", ".join(f"{k} {v:,.0f}px" for k, v in q.items())
    sections = [
        _count_table(
            profile.distributions["size"],
            "images",
            "Image size distribution (declared short side, px)",
            f"Every image once (labeled or not). Short-side quantiles: {quantiles}. "
            "Declared sizes come from the crawl and are re-measured after download (#9).",
        ),
        _rate_table(profile.rates["size"], "size", "Declared size (short side, px)", rates_note),
        _count_table(
            profile.distributions["aspect"],
            "images",
            "Aspect ratio distribution (width / height)",
            "Every image once. Extreme ratios are typical of banners and screenshots.",
        ),
        _rate_table(profile.rates["aspect"], "aspect", "Aspect ratio", rates_note),
        _rate_table(
            domains.head(TOP_SOURCES),
            "domain",
            f"Domain (top {min(TOP_SOURCES, domains.height)} of {domains.height} by volume)",
            rates_note + " image_source lower-cased with www. removed.",
        ),
        _rate_table(
            domains.filter(pl.col("n") >= MIN_SOURCE_N)
            .sort("yes_rate", "domain")
            .head(TOP_SOURCES),
            "domain",
            f"Lowest yes rate domains (n ≥ {MIN_SOURCE_N})",
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
        _count_table(
            profile.distributions["yes_per_entity"],
            "entities",
            "Relevant images per entity",
            "Number of non-conflicting yes images per entity (all entities).",
        )
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

    images = pl.read_parquet(args.ingest_dir / "images.parquet")
    labels = pl.read_parquet(args.ingest_dir / "labels.parquet")
    entities = pl.read_parquet(args.ingest_dir / "entities.parquet")
    profile = build_profile(images, labels, entities)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(profile), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
