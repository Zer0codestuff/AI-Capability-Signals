"""Write the Markdown and interactive HTML report from analysis tables.

The renderer never recomputes a statistic. Every number is read from a CSV written by an analysis
module. The HTML dashboard is intentionally dense — many charts, sortable tables, sidebar nav —
so it stays comparable to the previous deep report, without restoring indefensible composites.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import pandas as pd

from ..analysis import TableWriter
from ..provenance import RunContext

PACKAGE_DIR = Path(__file__).resolve().parent


def write_report(writer: TableWriter, context: RunContext, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name in ("report.css", "report.js"):
        src = PACKAGE_DIR / name
        if src.exists():
            shutil.copy2(src, assets_dir / name)

    tables = _load_tables(writer)
    markdown = _compose(context, tables)
    md_path = report_dir / "frontier_signals.md"
    md_path.write_text(markdown, encoding="utf-8")

    figures_dir = report_dir.parent / "figures"
    html_path = report_dir / "frontier_signals.html"
    html_path.write_text(_build_dashboard(context, tables, figures_dir), encoding="utf-8")

    (report_dir.parent / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=report/frontier_signals.html">'
        '<link rel="canonical" href="report/frontier_signals.html">'
        '<title>AI Capability Signals</title>'
        '<p>Redirecting to <a href="report/frontier_signals.html">the report</a>.</p>',
        encoding="utf-8",
    )
    return md_path


def _load_tables(writer: TableWriter) -> dict[str, pd.DataFrame]:
    names = (
        "source_freshness",
        "refusals",
        "disclosure_open_vs_closed",
        "disclosure_by_year",
        "disclosure_by_vendor",
        "compute_trend_estimates",
        "compute_forecast_backtest",
        "compute_frontier_forecast",
        "compute_selection_sensitivity",
        "cheapest_at_quality_threshold",
        "price_spread_at_matched_quality",
        "price_quality_regression",
        "price_tier_impact",
        "price_distribution_by_weights",
        "benchmark_pair_agreement",
        "benchmark_rank_instability",
        "composite_score_verdict",
        "open_closed_gap_trend",
        "open_weights_lag_summary",
        "open_closed_gap_by_category",
        "usage_surface_contrast",
        "usage_surface_summary",
        "usage_top_occupations",
        "usage_major_group_composition",
        "calendar_release_tests",
        "calendar_release_counts",
        "data_quality_findings",
        "swebench_top_systems",
        "swebench_scaffold_spread",
        "swebench_frontier_history",
        "lmarena_duplicate_report",
    )
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        path = writer.directory / f"{name}.csv"
        out[name] = writer.read(name) if path.exists() else pd.DataFrame()
    return out


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    if df.empty:
        return "_Unavailable._"
    view = df if cols is None else df[[c for c in cols if c in df.columns]]
    if n is not None:
        view = view.head(n)
    return view.to_markdown(index=False)


def _fig_md(name: str, caption: str) -> str:
    return f"![{caption}](../figures/{name}.png)\n\n*{caption}*"


def _compose(context: RunContext, t: dict[str, pd.DataFrame]) -> str:
    sections = [
        _header(context, t["source_freshness"]),
        _executive(t),
        _section_quality(t),
        _section_disclosure(t),
        _section_compute(t),
        _section_prices(t),
        _section_agreement(t),
        _section_open_weights(t),
        _section_usage(t),
        _section_swebench(t),
        _section_calendar(t),
        _section_methods(context),
    ]
    return "\n\n".join(section for section in sections if section)


def _header(context: RunContext, freshness: pd.DataFrame) -> str:
    rows = freshness.to_markdown(index=False) if not freshness.empty else "_Freshness table unavailable._"
    return f"""# AI Capability Signals

**Reference date:** {context.reference_date} (freshest source horizon; lagging sources are listed below).
**Generated:** {context.started_at}.
**Version:** 2.0.

This report answers questions that public data can actually support, each with a stated estimator
and an uncertainty interval. Claims the data cannot support are listed as refusals rather than
estimated. The HTML dashboard embeds every analysis chart for fast visual reading.

## Source freshness

{_fig_md("source_freshness", "Source freshness — days behind the reference date")}

{rows}
"""


def _executive(t: dict[str, pd.DataFrame]) -> str:
    bullets: list[str] = []
    disclosure_cmp = t["disclosure_open_vs_closed"]
    compute_trends = t["compute_trend_estimates"]
    verdict = t["composite_score_verdict"]
    gap_trend = t["open_closed_gap_trend"]
    lag_summary = t["open_weights_lag_summary"]
    usage_contrast = t["usage_surface_contrast"]
    refusals = t["refusals"]

    if not disclosure_cmp.empty:
        significant = disclosure_cmp[disclosure_cmp["significant_after_bh"]]
        if not significant.empty:
            best = significant.iloc[0]
            bullets.append(
                f"Open-weight releases disclose more than closed-weight releases. "
                f"The largest corrected difference is for **{best['field_label']}**: "
                f"{best['difference']:.0%} points "
                f"(95% CI {best['ci_low']:.0%} to {best['ci_high']:.0%}; BH q={best['q_value_bh']:.3f})."
            )

    if not compute_trends.empty:
        p90 = compute_trends[
            (compute_trends["specification"] == "frontier_p90_by_year")
            & (compute_trends["estimator"] == "ols_hc3")
        ]
        if not p90.empty:
            row = p90.iloc[0]
            bullets.append(
                f"The disclosed training-compute frontier grows at "
                f"**{row['log10_flop_per_year']:.2f} log10 FLOP per year** "
                f"(HC3 95% CI {row['ci_low']:.2f}–{row['ci_high']:.2f}), "
                f"a doubling time of about **{row['doubling_time_months']:.1f} months**. "
                "This is a lower bound: compute disclosure is incomplete and voluntary."
            )

    if not verdict.empty:
        bullets.append(str(verdict.iloc[0]["verdict"]))

    if not gap_trend.empty:
        latest = gap_trend.sort_values("window_end").iloc[-1]
        bullets.append(
            f"Inside the current arena rating regime ({latest['methodology_regime']}), "
            f"the open–closed gap changes by "
            f"**{latest['gap_change_per_year_ols']:+.1f} rating points per year** "
            f"(HC3 CI {latest['ci_low']:+.1f} to {latest['ci_high']:+.1f}). "
            "A crossing date is not published."
        )

    if not lag_summary.empty:
        row = lag_summary.iloc[0]
        median = row["kaplan_meier_median_lag_days"]
        if pd.notna(median):
            bullets.append(
                f"Kaplan–Meier median lag for open weights to reach a historical closed-frontier "
                f"level: **{median:.0f} days** "
                f"({int(row['levels_still_unmatched'])} of {int(row['levels_tracked'])} levels "
                "still unmatched and right-censored)."
            )

    if not usage_contrast.empty:
        row = usage_contrast.iloc[0]
        bullets.append(
            f"On the same occupations, the consumer product and the first-party API disagree about "
            f"automation share by **{row['mean_difference_a_minus_b']:+.1f} percentage points** "
            f"(95% CI {row['ci_low']:+.1f} to {row['ci_high']:+.1f}). "
            "No single 'share of work automated' number is published."
        )

    bullets.append(
        f"The pipeline records **{len(refusals)} refused claims** — quantities the available "
        "public data cannot support. They are listed below rather than estimated."
    )
    body = "\n".join(f"- {item}" for item in bullets)
    return f"## Findings that survive the audit\n\n{body}"


def _section_quality(t: dict[str, pd.DataFrame]) -> str:
    quality = t["data_quality_findings"]
    refusals = t["refusals"]
    blocking = quality[quality["severity"] == "blocking"] if not quality.empty else quality
    return f"""## Data quality and refusals

Before any estimate, the pipeline reports the properties of the data that constrain what can be
claimed. Blocking findings are paired with an entry in the refusals ledger.

{_fig_md("refusals_overview", "Where the pipeline refuses to invent a number")}
{_fig_md("source_freshness", "Source freshness")}

### Blocking findings

{_md_table(blocking, ["source_id", "check", "value", "interpretation"])}

### Refused claims

{_md_table(refusals)}
"""


def _section_disclosure(t: dict[str, pd.DataFrame]) -> str:
    disclosure_cmp = t["disclosure_open_vs_closed"]
    if disclosure_cmp.empty:
        return ""
    return f"""## Disclosure completeness

Disclosure is a property of the record: either a field is populated or it is not. The comparison
below is a difference of proportions with a two-sample bootstrap interval, corrected across fields
by Benjamini–Hochberg.

{_fig_md("disclosure_open_vs_closed", "Open vs closed disclosure rates")}
{_fig_md("disclosure_by_year", "Disclosure rates over time (Wilson intervals)")}
{_fig_md("disclosure_by_vendor", "Top vendors by disclosure completeness")}

{_md_table(disclosure_cmp, ["field_label", "open_disclosure_rate", "closed_disclosure_rate", "difference", "ci_low", "ci_high", "q_value_bh", "significant_after_bh"])}

**Confound, stated rather than adjusted:** open-weight releases skew academic and closed releases
skew commercial. Part of the difference is publication culture.
"""


def _section_compute(t: dict[str, pd.DataFrame]) -> str:
    trends = t["compute_trend_estimates"]
    if trends.empty:
        return ""
    return f"""## Disclosed training-compute growth

Training compute is a physical, unbounded quantity. Its logarithm can be extrapolated without
hitting a ceiling — unlike a benchmark percentage capped at 100.

{_fig_md("compute_frontier", "Disclosed training-compute frontier")}
{_fig_md("compute_sensitivity", "Compute-growth slope under selection restrictions")}
{_fig_md("compute_backtest_skill", "Backtest: trend MAE vs last-value baseline")}

### Trend estimates

{_md_table(trends, ["specification", "estimator", "n", "log10_flop_per_year", "ci_low", "ci_high", "doubling_time_months"])}

### Backtest against a last-value baseline

{_md_table(t["compute_forecast_backtest"])}

### Selection sensitivity

{_md_table(t["compute_selection_sensitivity"], ["restriction", "n", "log10_flop_per_year", "doubling_time_months", "relative_change_vs_baseline", "status"])}

### Expectations for provisional and future years

{_md_table(t["compute_frontier_forecast"])}
"""


def _section_prices(t: dict[str, pd.DataFrame]) -> str:
    if t["cheapest_at_quality_threshold"].empty and t["price_spread_at_matched_quality"].empty:
        return ""
    return f"""## Price structure in the current catalogue

Quality and price come from the **same catalogue record**, so there is no cross-source name join.
The catalogue is a cross-section of currently listed models: it cannot support a price history.

{_fig_md("price_quality_pareto", "Price–quality Pareto frontier")}
{_fig_md("price_spread_bands", "Price spread at matched quality")}
{_fig_md("price_cheapest_ladder", "Cheapest listed model at each quality threshold")}
{_fig_md("price_by_weights", "Catalogue price levels by weight availability")}

### Cheapest listed model at each quality threshold

{_md_table(t["cheapest_at_quality_threshold"])}

### Price spread among models of comparable quality

{_md_table(t["price_spread_at_matched_quality"], ["quality_band", "models", "cheapest_usd_per_call", "dearest_usd_per_call", "max_min_price_ratio", "ratio_ci_low", "ratio_ci_high"])}

### Weight availability and price, at equal measured quality

{_md_table(t["price_quality_regression"], ["term", "price_factor", "ci_low", "ci_high", "n", "causal_status"])}

### Prompt-length price tiers

{_md_table(t["price_tier_impact"])}
"""


def _section_agreement(t: dict[str, pd.DataFrame]) -> str:
    agreement = t["benchmark_pair_agreement"]
    if agreement.empty:
        return ""
    return f"""## Do public benchmarks agree?

A composite is treated as defensible only if every measured pair of independent benchmarks has a
Kendall τ lower bound above 0.8.

{_fig_md("benchmark_agreement", "Benchmark pair agreement (Kendall τ-b)")}
{_fig_md("benchmark_rank_instability", "Models whose standing depends on the benchmark")}

{_md_table(t["composite_score_verdict"])}

{_md_table(agreement, ["benchmark_a", "benchmark_b", "common_models", "kendall_tau_b", "tau_ci_low", "tau_ci_high", "top_10_overlap", "both_vendor_composites"])}

Because the threshold is not met, **no composite capability score is published**.
"""


def _section_open_weights(t: dict[str, pd.DataFrame]) -> str:
    if t["open_closed_gap_trend"].empty and t["open_weights_lag_summary"].empty:
        return ""
    return f"""## Open-weight lag on the arena frontier

Trends are estimated inside a single rating-methodology regime.

{_fig_md("open_closed_gap", "Open-weight vs closed frontier")}
{_fig_md("open_closed_gap_trend", "Is the open–closed gap closing?")}
{_fig_md("open_closed_gap_by_category", "Open–closed gap by category")}
{_fig_md("open_weights_lag_km", "Open-weight catch-up lag (right-censored)")}

### Gap trend by regime

{_md_table(t["open_closed_gap_trend"])}

### Catch-up lag (Kaplan–Meier, right-censored)

{_md_table(t["open_weights_lag_summary"])}

### Gap by category in the latest publication

{_md_table(t["open_closed_gap_by_category"], ["arena", "category", "open_best_model", "closed_best_model", "gap_rating", "gap_ci_low", "gap_distinguishable_from_zero"], n=12)}
"""


def _section_usage(t: dict[str, pd.DataFrame]) -> str:
    if t["usage_surface_summary"].empty:
        return ""
    return f"""## Observed usage composition (not employment impact)

These tables describe the composition of one vendor's observed conversations. They are not a sample
of the workforce and carry no information about employment outcomes.

{_fig_md("usage_surface_contrast", "Automation share by surface")}
{_fig_md("usage_top_occupations", "Top occupations by usage share")}
{_fig_md("usage_major_groups", "Usage composition by SOC major group")}

### Surface medians

{_md_table(t["usage_surface_summary"])}

### Surface contrast on the same occupations

{_md_table(t["usage_surface_contrast"])}
"""


def _section_swebench(t: dict[str, pd.DataFrame]) -> str:
    refusals = t["refusals"]
    stale_refused = (
        not refusals.empty
        and refusals["claim"].astype(str).str.contains("SWE-bench", case=False).any()
    )
    note = (
        "The public submission directory is stale relative to the reference date, so only the "
        "historical series is shown. Present-tense claims are refused."
        if stale_refused
        else "Scores describe an agent scaffold plus a model, not a model alone."
    )
    return f"""## SWE-bench Verified: historical record

{note}

{_fig_md("swebench_frontier", "SWE-bench Verified public frontier")}
{_fig_md("swebench_top_systems", "Top SWE-bench Verified public submissions")}

### Top public submissions

{_md_table(t["swebench_top_systems"], n=12)}

### Within-model scaffold spread

{_md_table(t["swebench_scaffold_spread"], n=10)}
"""


def _section_calendar(t: dict[str, pd.DataFrame]) -> str:
    calendar = t["calendar_release_tests"]
    if calendar.empty:
        return ""
    surviving = int(calendar["surviving_after_correction"].iloc[0])
    note = (
        "No feature survives correction — apparent clusters were multiple-testing artifacts."
        if surviving == 0
        else (
            "Clusters survive correction. That is consistent with real vendor scheduling "
            "(weekdays, conference seasons), not astrology, and not a finding beyond calendars."
        )
    )
    return f"""## Calendar negative control

A small, pre-registered family of calendar features (weekday, month, quarter) is tested against a
year-preserving random-date null, then corrected by Benjamini–Hochberg.

**Results surviving BH correction:** {surviving} of {len(calendar)}. {note}

{_fig_md("calendar_control", "Calendar clustering of releases")}
{_fig_md("calendar_weekday_counts", "Notable model releases by weekday")}

{_md_table(calendar, ["feature", "top_bucket", "top_count", "share", "permutation_p_value", "q_value_bh", "significant_after_bh"])}
"""


def _section_methods(context: RunContext) -> str:
    return f"""## Methods in brief

- **Reference date** `{context.reference_date}` is the freshest source horizon.
- **Intervals** are Wilson, HC3, Theil–Sen, and percentile bootstraps that resample the unit of analysis.
- **Multiple comparisons** inside a pre-registered family are corrected by Benjamini–Hochberg.
- **Forecasts** are published only when a rolling-origin backtest beats a last-value baseline.
- **Refusals** replace capped extrapolations.
- Audit of the previous version: [`docs/audit_of_previous_version.md`](../docs/audit_of_previous_version.md).

## Reproduce

```bash
uv sync
uv run aicap                # full refresh
uv run aicap --from-interim # re-analyse cached frames
uv run python -m unittest discover -s tests
```
"""


# ---------------------------------------------------------------------------
# Interactive HTML dashboard
# ---------------------------------------------------------------------------


FIGURE_CATALOG: list[tuple[str, str, str]] = [
    ("quality", "source_freshness", "Freschezza delle fonti"),
    ("quality", "refusals_overview", "Refusals per analisi"),
    ("disclosure", "disclosure_open_vs_closed", "Disclosure open vs closed"),
    ("disclosure", "disclosure_by_year", "Disclosure nel tempo"),
    ("disclosure", "disclosure_by_vendor", "Disclosure per vendor"),
    ("compute", "compute_frontier", "Frontiera compute dichiarato"),
    ("compute", "compute_sensitivity", "Sensibilità della slope"),
    ("compute", "compute_backtest_skill", "Backtest vs baseline"),
    ("pricing", "price_quality_pareto", "Pareto prezzo–qualità"),
    ("pricing", "price_spread_bands", "Spread di prezzo a qualità pari"),
    ("pricing", "price_cheapest_ladder", "Ladder del più economico"),
    ("pricing", "price_by_weights", "Prezzi per disponibilità pesi"),
    ("benchmarks", "benchmark_agreement", "Accordo tra benchmark"),
    ("benchmarks", "benchmark_rank_instability", "Instabilità di rango"),
    ("lag", "open_closed_gap", "Frontiera open vs closed"),
    ("lag", "open_closed_gap_trend", "Trend del gap"),
    ("lag", "open_closed_gap_by_category", "Gap per categoria"),
    ("lag", "open_weights_lag_km", "Lag Kaplan–Meier"),
    ("usage", "usage_surface_contrast", "Automazione per superficie"),
    ("usage", "usage_top_occupations", "Top occupation"),
    ("usage", "usage_major_groups", "Gruppi SOC"),
    ("swebench", "swebench_frontier", "Frontiera SWE-bench"),
    ("swebench", "swebench_top_systems", "Top sistemi SWE-bench"),
    ("calendar", "calendar_control", "Controllo calendario"),
    ("calendar", "calendar_weekday_counts", "Rilasci per giorno"),
]


def _metric_cards(t: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    disclosure = t["disclosure_open_vs_closed"]
    if not disclosure.empty:
        significant = disclosure[disclosure["significant_after_bh"]]
        row = significant.iloc[0] if not significant.empty else disclosure.iloc[0]
        cards.append(
            {
                "label": "Δ disclosure",
                "value": f"{float(row['difference']) * 100:+.0f} pp",
                "detail": str(row["field_label"]),
                "tone": "good" if float(row["difference"]) > 0 else "warn",
            }
        )
    trends = t["compute_trend_estimates"]
    if not trends.empty:
        p90 = trends[
            (trends["specification"] == "frontier_p90_by_year") & (trends["estimator"] == "ols_hc3")
        ]
        if not p90.empty:
            slope = float(p90.iloc[0]["log10_flop_per_year"])
            cards.append(
                {
                    "label": "Crescita compute",
                    "value": f"{slope:.2f} log₁₀/anno",
                    "detail": f"doubling ≈ {float(p90.iloc[0]['doubling_time_months']):.1f} mesi",
                    "tone": "neutral",
                }
            )
    lag = t["open_weights_lag_summary"]
    if not lag.empty and pd.notna(lag.iloc[0].get("kaplan_meier_median_lag_days")):
        cards.append(
            {
                "label": "Lag open-weights",
                "value": f"{float(lag.iloc[0]['kaplan_meier_median_lag_days']):.0f} giorni",
                "detail": "mediana Kaplan–Meier",
                "tone": "neutral",
            }
        )
    agreement = t["benchmark_pair_agreement"]
    if not agreement.empty:
        indep = agreement[~agreement["both_vendor_composites"]] if "both_vendor_composites" in agreement else agreement
        if indep.empty:
            indep = agreement
        tau = float(indep["kendall_tau_b"].min())
        cards.append(
            {
                "label": "Accordo benchmark",
                "value": f"τ ≥ {tau:.2f}",
                "detail": "min Kendall τ-b (coppie indipendenti)",
                "tone": "warn" if tau < 0.8 else "good",
            }
        )
    usage = t["usage_surface_contrast"]
    if not usage.empty:
        diff = float(usage.iloc[0]["mean_difference_a_minus_b"])
        cards.append(
            {
                "label": "API − consumer",
                "value": f"{diff:+.1f} pp",
                "detail": "Δ automazione (stesse occupation)",
                "tone": "neutral",
            }
        )
    cards.append(
        {
            "label": "Refusals",
            "value": str(len(t["refusals"])),
            "detail": "claim rifiutati esplicitamente",
            "tone": "warn",
        }
    )
    return cards


def _fig_html(figures_dir: Path, stem: str, caption: str) -> str:
    path = figures_dir / f"{stem}.png"
    if not path.exists():
        return f'<p class="missing-fig">Figura mancante: <code>{html.escape(stem)}.png</code></p>'
    src = html.escape(f"../figures/{stem}.png")
    cap = html.escape(caption)
    return (
        f'<figure class="chart" data-lightbox>'
        f'<img src="{src}" alt="{cap}" loading="lazy" />'
        f"<figcaption>{cap}</figcaption>"
        f"</figure>"
    )


def _df_html(df: pd.DataFrame, table_id: str, cols: list[str] | None = None, n: int = 30) -> str:
    if df.empty:
        return '<p class="muted">Nessun dato.</p>'
    view = df if cols is None else df[[c for c in cols if c in df.columns]]
    view = view.head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    headers = "".join(f"<th data-sort>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, r in view.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in r.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap" id="{html.escape(table_id)}">'
        f'<div class="table-tools"><input type="search" class="table-filter" '
        f'placeholder="Filtra tabella…" aria-label="Filtra" /></div>'
        f'<table class="data-table sortable"><thead><tr>{headers}</tr></thead>'
        f"<tbody>\n" + "\n".join(rows) + "\n</tbody></table></div>"
    )


def _findings_html(t: dict[str, pd.DataFrame]) -> str:
    # Reuse executive markdown bullets as HTML list via a light pass.
    md = _executive(t)
    items = []
    for line in md.splitlines():
        if line.startswith("- "):
            items.append(f"<li>{_inline(line[2:])}</li>")
    return "<ul class='findings-list'>" + "".join(items) + "</ul>"


def _build_dashboard(context: RunContext, t: dict[str, pd.DataFrame], figures_dir: Path) -> str:
    cards = _metric_cards(t)
    card_html = "".join(
        f'<article class="metric-card tone-{html.escape(c["tone"])}">'
        f'<p class="metric-label">{html.escape(c["label"])}</p>'
        f'<p class="metric-value">{html.escape(c["value"])}</p>'
        f'<p class="metric-detail">{html.escape(c["detail"])}</p>'
        f"</article>"
        for c in cards
    )
    n_figs = sum(1 for _, stem, _ in FIGURE_CATALOG if (figures_dir / f"{stem}.png").exists())

    sections: list[tuple[str, str, str]] = [
        (
            "findings",
            "Findings",
            f"<p>Risultati che sopravvivono all'audit metodologico. I claim non supportati sono "
            f"nel ledger refusals, non stimati.</p>{_findings_html(t)}",
        ),
        (
            "quality",
            "Qualità dati & refusals",
            f"{_fig_html(figures_dir, 'refusals_overview', 'Refusals per analisi')}"
            f"{_fig_html(figures_dir, 'source_freshness', 'Freschezza fonti')}"
            f"<h3>Blocking findings</h3>"
            f"{_df_html(t['data_quality_findings'], 'tbl-quality', ['source_id', 'check', 'severity', 'value', 'interpretation'])}"
            f"<h3>Refusals</h3>"
            f"{_df_html(t['refusals'], 'tbl-refusals')}",
        ),
        (
            "disclosure",
            "Disclosure",
            f"<p>Trasparenza dei record Epoch — non un indice di capability.</p>"
            f"{_fig_html(figures_dir, 'disclosure_open_vs_closed', 'Open vs closed disclosure')}"
            f"{_fig_html(figures_dir, 'disclosure_by_year', 'Disclosure nel tempo')}"
            f"{_fig_html(figures_dir, 'disclosure_by_vendor', 'Disclosure per vendor')}"
            f"{_df_html(t['disclosure_open_vs_closed'], 'tbl-disc')}",
        ),
        (
            "compute",
            "Compute scaling",
            f"<p>Trend in log₁₀ FLOP con sensibilità e backtest. Nessuna proiezione di score misti.</p>"
            f"{_fig_html(figures_dir, 'compute_frontier', 'Frontiera compute')}"
            f"{_fig_html(figures_dir, 'compute_sensitivity', 'Sensibilità slope')}"
            f"{_fig_html(figures_dir, 'compute_backtest_skill', 'Backtest skill')}"
            f"{_df_html(t['compute_trend_estimates'], 'tbl-compute')}"
            f"{_df_html(t['compute_selection_sensitivity'], 'tbl-sens')}",
        ),
        (
            "pricing",
            "Prezzi (catalogo corrente)",
            f"<p><strong>Non</strong> è una serie storica: è la struttura del catalogo al pull.</p>"
            f"{_fig_html(figures_dir, 'price_quality_pareto', 'Pareto prezzo–qualità')}"
            f"{_fig_html(figures_dir, 'price_spread_bands', 'Spread a qualità pari')}"
            f"{_fig_html(figures_dir, 'price_cheapest_ladder', 'Ladder più economico')}"
            f"{_fig_html(figures_dir, 'price_by_weights', 'Prezzi per pesi')}"
            f"{_df_html(t['cheapest_at_quality_threshold'], 'tbl-cheap')}"
            f"{_df_html(t['price_spread_at_matched_quality'], 'tbl-spread')}",
        ),
        (
            "benchmarks",
            "Accordo benchmark",
            f"<p>Premessa di ogni composito: i segnali misurano la stessa cosa. Qui si testa.</p>"
            f"{_fig_html(figures_dir, 'benchmark_agreement', 'Accordo tra benchmark')}"
            f"{_fig_html(figures_dir, 'benchmark_rank_instability', 'Instabilità di rango')}"
            f"{_df_html(t['composite_score_verdict'], 'tbl-verdict')}"
            f"{_df_html(t['benchmark_pair_agreement'], 'tbl-agree')}",
        ),
        (
            "lag",
            "Lag open-weights",
            f"<p>Gap e catching-up dentro un singolo regime di rating; KM con censoring.</p>"
            f"{_fig_html(figures_dir, 'open_closed_gap', 'Frontiera open vs closed')}"
            f"{_fig_html(figures_dir, 'open_closed_gap_trend', 'Trend del gap')}"
            f"{_fig_html(figures_dir, 'open_closed_gap_by_category', 'Gap per categoria')}"
            f"{_fig_html(figures_dir, 'open_weights_lag_km', 'Lag Kaplan–Meier')}"
            f"{_df_html(t['open_closed_gap_trend'], 'tbl-gap')}"
            f"{_df_html(t['open_weights_lag_summary'], 'tbl-km')}"
            f"{_df_html(t['open_closed_gap_by_category'], 'tbl-cat', n=20)}",
        ),
        (
            "usage",
            "Uso osservato",
            f"<p>Composizione di traffico documentato — non impatto sull'occupazione.</p>"
            f"{_fig_html(figures_dir, 'usage_surface_contrast', 'Automazione per superficie')}"
            f"{_fig_html(figures_dir, 'usage_top_occupations', 'Top occupation')}"
            f"{_fig_html(figures_dir, 'usage_major_groups', 'Gruppi SOC')}"
            f"{_df_html(t['usage_surface_summary'], 'tbl-usage')}"
            f"{_df_html(t['usage_surface_contrast'], 'tbl-contrast')}",
        ),
        (
            "swebench",
            "SWE-bench",
            f"<p>Serie storica pubblica; lo score è scaffold+modello, non solo modello.</p>"
            f"{_fig_html(figures_dir, 'swebench_frontier', 'Frontiera SWE-bench')}"
            f"{_fig_html(figures_dir, 'swebench_top_systems', 'Top sistemi')}"
            f"{_df_html(t['swebench_top_systems'], 'tbl-swe', n=15)}"
            f"{_df_html(t['swebench_scaffold_spread'], 'tbl-scaffold', n=15)}",
        ),
        (
            "calendar",
            "Controllo calendario",
            f"<p>Negative control: clustering di rilasci per weekday/mese/trimestre.</p>"
            f"{_fig_html(figures_dir, 'calendar_control', 'Test calendario')}"
            f"{_fig_html(figures_dir, 'calendar_weekday_counts', 'Rilasci per weekday')}"
            f"{_df_html(t['calendar_release_tests'], 'tbl-cal')}",
        ),
        (
            "methods",
            "Metodi",
            f"""
            <ul>
              <li><strong>Reference date</strong> <code>{html.escape(str(context.reference_date))}</code>
                  = orizzonte della fonte più fresca.</li>
              <li>Intervalli Wilson / HC3 / Theil–Sen / bootstrap sul unit of analysis.</li>
              <li>Confronti multipli corretti con Benjamini–Hochberg.</li>
              <li>Forecast solo se il backtest batte la baseline last-value.</li>
              <li>Refusals al posto di estrapolazioni capped.</li>
            </ul>
            <p>Audit v1: <a href="../docs/audit_of_previous_version.md">docs/audit_of_previous_version.md</a>
            · Markdown: <a href="frontier_signals.md">frontier_signals.md</a></p>
            """,
        ),
    ]

    nav = "".join(
        f'<a href="#{html.escape(sid)}">{html.escape(title)}</a>' for sid, title, _ in sections
    )
    body = "".join(
        f'<section id="{html.escape(sid)}" class="report-section">'
        f"<h2>{html.escape(title)}</h2>{content}</section>"
        for sid, title, content in sections
    )

    gallery = "".join(
        f'<a class="gallery-item" href="#{sec}">{html.escape(cap)} <span>{html.escape(stem)}</span></a>'
        for sec, stem, cap in FIGURE_CATALOG
        if (figures_dir / f"{stem}.png").exists()
    )

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Capability Signals — Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="assets/report.css" />
</head>
<body>
  <div class="shell">
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        <p class="brand-kicker">AI Capability Signals</p>
        <h1>Frontier report</h1>
        <p class="brand-meta">ref {html.escape(str(context.reference_date))} · {n_figs} figure · v2</p>
      </div>
      <nav class="side-nav" aria-label="Sezioni">{nav}</nav>
      <div class="gallery-nav" aria-label="Figure">
        <p class="gallery-label">Figure</p>
        {gallery}
      </div>
      <p class="side-note">Tabelle ordinabili · click sulle figure per ingrandire</p>
    </aside>
    <main class="main">
      <header class="hero">
        <p class="eyebrow">Analisi empirica · senza compositi indefendibili</p>
        <h1>Segnali di capability frontier</h1>
        <p class="lede">
          Dashboard densificata come il report precedente: disclosure, compute, prezzi, accordo
          tra benchmark, lag open-weights, uso documentato, SWE-bench e controlli calendario —
          con refusals espliciti dove i dati non supportano la domanda.
        </p>
      </header>
      <section class="metrics" aria-label="Snapshot">{card_html}</section>
      {body}
      <footer class="footer">
        <p>Generato {html.escape(str(context.started_at))} ·
           <code>aicap --from-interim</code> · dati in <code>data/analysis/</code></p>
      </footer>
    </main>
  </div>
  <div id="lightbox" class="lightbox" hidden>
    <button type="button" class="lightbox-close" aria-label="Chiudi">×</button>
    <img alt="" />
  </div>
  <script src="assets/report.js"></script>
</body>
</html>
"""


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text
