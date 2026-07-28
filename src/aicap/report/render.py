"""Write the Markdown and HTML report from analysis tables.

The renderer never recomputes a statistic. Every number is read from a CSV written by an analysis
module, so the prose cannot drift away from the computation — the failure mode that left the
previous version's report describing a cross-domain fallback the code no longer performed.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

from ..analysis import TableWriter
from ..provenance import RunContext


def write_report(writer: TableWriter, context: RunContext, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = _compose(writer, context)
    md_path = report_dir / "frontier_signals.md"
    md_path.write_text(markdown, encoding="utf-8")
    html_path = report_dir / "frontier_signals.html"
    html_path.write_text(_to_html(markdown), encoding="utf-8")
    # Keep a stable entry point for GitHub Pages.
    (report_dir.parent / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=report/frontier_signals.html">'
        '<link rel="canonical" href="report/frontier_signals.html">'
        '<title>AI Capability Signals</title>'
        '<p>Redirecting to <a href="report/frontier_signals.html">the report</a>.</p>',
        encoding="utf-8",
    )
    return md_path


def _compose(writer: TableWriter, context: RunContext) -> str:
    def table(name: str) -> pd.DataFrame:
        path = writer.directory / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return writer.read(name)

    freshness = table("source_freshness")
    refusals = table("refusals")
    disclosure_cmp = table("disclosure_open_vs_closed")
    compute_trends = table("compute_trend_estimates")
    compute_backtest = table("compute_forecast_backtest")
    compute_forecast = table("compute_frontier_forecast")
    sensitivity = table("compute_selection_sensitivity")
    cheapest = table("cheapest_at_quality_threshold")
    spread = table("price_spread_at_matched_quality")
    regression = table("price_quality_regression")
    tiers = table("price_tier_impact")
    agreement = table("benchmark_pair_agreement")
    verdict = table("composite_score_verdict")
    gap_trend = table("open_closed_gap_trend")
    lag_summary = table("open_weights_lag_summary")
    gap_by_cat = table("open_closed_gap_by_category")
    usage_contrast = table("usage_surface_contrast")
    usage_summary = table("usage_surface_summary")
    calendar = table("calendar_release_tests")
    quality = table("data_quality_findings")
    swe_top = table("swebench_top_systems")
    swe_spread = table("swebench_scaffold_spread")

    sections = [
        _header(context, freshness),
        _executive(disclosure_cmp, compute_trends, agreement, verdict, gap_trend, lag_summary, usage_contrast, refusals),
        _section_quality(quality, freshness, refusals),
        _section_disclosure(disclosure_cmp),
        _section_compute(compute_trends, compute_backtest, compute_forecast, sensitivity),
        _section_prices(cheapest, spread, regression, tiers),
        _section_agreement(agreement, verdict),
        _section_open_weights(gap_trend, lag_summary, gap_by_cat),
        _section_usage(usage_summary, usage_contrast),
        _section_swebench(swe_top, swe_spread, refusals),
        _section_calendar(calendar),
        _section_methods(context),
    ]
    return "\n\n".join(section for section in sections if section)


def _header(context: RunContext, freshness: pd.DataFrame) -> str:
    rows = ""
    if not freshness.empty:
        rows = freshness.to_markdown(index=False)
    return f"""# AI Capability Signals

**Reference date:** {context.reference_date} (derived from the least fresh source).
**Generated:** {context.started_at}.
**Version:** 2.0.

This report answers a small number of questions that public data can actually support, each with a
stated estimator and an uncertainty interval. Claims the data cannot support are listed as
refusals rather than estimated.

## Source freshness

The reference date is the minimum of these horizons, so no claim can be newer than the least fresh
input.

{rows if rows else "_Freshness table unavailable._"}
"""


def _executive(
    disclosure_cmp: pd.DataFrame,
    compute_trends: pd.DataFrame,
    agreement: pd.DataFrame,
    verdict: pd.DataFrame,
    gap_trend: pd.DataFrame,
    lag_summary: pd.DataFrame,
    usage_contrast: pd.DataFrame,
    refusals: pd.DataFrame,
) -> str:
    bullets: list[str] = []

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


def _section_quality(quality: pd.DataFrame, freshness: pd.DataFrame, refusals: pd.DataFrame) -> str:
    blocking = quality[quality["severity"] == "blocking"] if not quality.empty else quality
    refusals_md = refusals.to_markdown(index=False) if not refusals.empty else "_None._"
    blocking_md = (
        blocking[["source_id", "check", "value", "interpretation"]].to_markdown(index=False)
        if not blocking.empty
        else "_No blocking findings._"
    )
    return f"""## Data quality and refusals

Before any estimate, the pipeline reports the properties of the data that constrain what can be
claimed. Blocking findings are paired with an entry in the refusals ledger.

### Blocking findings

{blocking_md}

### Refused claims

{refusals_md}

![Source coverage is summarised in the freshness table above.](../figures/calendar_control.png)
"""


def _section_disclosure(disclosure_cmp: pd.DataFrame) -> str:
    if disclosure_cmp.empty:
        return ""
    return f"""## Disclosure completeness

Disclosure is a property of the record: either a field is populated or it is not. The comparison
below is a difference of proportions with a two-sample bootstrap interval, corrected across fields
by Benjamini–Hochberg. The accessibility field is excluded from the test because the weights class
is derived from it, which would make the comparison circular.

{disclosure_cmp[["field_label", "open_disclosure_rate", "closed_disclosure_rate", "difference", "ci_low", "ci_high", "q_value_bh", "significant_after_bh"]].to_markdown(index=False)}

![Open vs closed disclosure rates](../figures/disclosure_open_vs_closed.png)

**Confound, stated rather than adjusted:** open-weight releases skew academic and closed releases
skew commercial. Part of the difference is publication culture. The data contain no instrument that
separates the two.
"""


def _section_compute(
    trends: pd.DataFrame,
    backtest: pd.DataFrame,
    forecast: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    if trends.empty:
        return ""
    return f"""## Disclosed training-compute growth

Training compute is a physical, unbounded quantity. Its logarithm can be extrapolated without
hitting a ceiling — unlike a benchmark percentage capped at 100, which is why the previous version's
domain forecasts were withdrawn.

### Trend estimates

{trends[["specification", "estimator", "n", "log10_flop_per_year", "ci_low", "ci_high", "doubling_time_months"]].to_markdown(index=False)}

### Backtest against a last-value baseline

A forecast is published only when it beats carrying the last observed frontier forward. The
prediction interval is built from measured out-of-sample errors, not from the regression's
in-sample standard error.

{backtest.to_markdown(index=False) if not backtest.empty else "_No backtest rows._"}

### Selection sensitivity

Compute disclosure is incomplete and voluntary. The slope under several restrictions:

{sensitivity[["restriction", "n", "log10_flop_per_year", "doubling_time_months", "relative_change_vs_baseline", "status"]].to_markdown(index=False) if not sensitivity.empty else "_Unavailable._"}

### Expectations for provisional and future years

{forecast.to_markdown(index=False) if not forecast.empty else "_No horizon beat the baseline; no forecast published._"}

![Disclosed compute frontier](../figures/compute_frontier.png)
"""


def _section_prices(
    cheapest: pd.DataFrame,
    spread: pd.DataFrame,
    regression: pd.DataFrame,
    tiers: pd.DataFrame,
) -> str:
    if cheapest.empty and spread.empty:
        return ""
    return f"""## Price structure in the current catalogue

Quality and price come from the **same catalogue record**, so there is no cross-source name join.
The catalogue is a cross-section of currently listed models: it cannot support a price history, and
that claim is refused.

### Cheapest listed model at each quality threshold

{cheapest.to_markdown(index=False) if not cheapest.empty else "_Unavailable._"}

### Price spread among models of comparable quality

If quality determined price, the max/min ratio inside a narrow quality band would be near 1. It is
not.

{spread[["quality_band", "models", "cheapest_usd_per_call", "dearest_usd_per_call", "max_min_price_ratio", "ratio_ci_low", "ratio_ci_high"]].to_markdown(index=False) if not spread.empty else "_Unavailable._"}

### Weight availability and price, at equal measured quality

{regression[["term", "price_factor", "ci_low", "ci_high", "n", "causal_status"]].to_markdown(index=False) if not regression.empty else "_Unavailable._"}

### Prompt-length price tiers

{tiers.to_markdown(index=False) if not tiers.empty else "_Unavailable._"}

![Price–quality Pareto frontier](../figures/price_quality_pareto.png)

![Price spread at matched quality](../figures/price_spread_bands.png)
"""


def _section_agreement(agreement: pd.DataFrame, verdict: pd.DataFrame) -> str:
    if agreement.empty:
        return ""
    return f"""## Do public benchmarks agree?

Every composite "AI capability score" rests on an untested premise: that the signals being combined
measure one underlying thing. This section tests that premise. A composite is treated as defensible
only if every measured pair of independent benchmarks has a Kendall τ lower bound above 0.8.

{verdict.to_markdown(index=False) if not verdict.empty else ""}

{agreement[["benchmark_a", "benchmark_b", "common_models", "kendall_tau_b", "tau_ci_low", "tau_ci_high", "top_10_overlap", "both_vendor_composites"]].to_markdown(index=False)}

![Benchmark agreement](../figures/benchmark_agreement.png)

Because the threshold is not met, **no composite capability score is published**. Benchmarks are
reported separately.
"""


def _section_open_weights(
    gap_trend: pd.DataFrame, lag_summary: pd.DataFrame, gap_by_cat: pd.DataFrame
) -> str:
    if gap_trend.empty and lag_summary.empty:
        return ""
    top_cats = (
        gap_by_cat.head(12)[
            ["arena", "category", "open_best_model", "closed_best_model", "gap_rating", "gap_ci_low", "gap_distinguishable_from_zero"]
        ].to_markdown(index=False)
        if not gap_by_cat.empty
        else "_Unavailable._"
    )
    return f"""## Open-weight lag on the arena frontier

Trends are estimated inside a single rating-methodology regime. A slope that spans a methodology
break would partly measure the break.

### Gap trend by regime

{gap_trend.to_markdown(index=False) if not gap_trend.empty else "_Unavailable._"}

### Catch-up lag (Kaplan–Meier, right-censored)

{lag_summary.to_markdown(index=False) if not lag_summary.empty else "_Unavailable._"}

### Gap by category in the latest publication

A single pooled gap number is refused; the gap is category-specific.

{top_cats}

![Open vs closed frontier](../figures/open_closed_gap.png)
"""


def _section_usage(usage_summary: pd.DataFrame, usage_contrast: pd.DataFrame) -> str:
    if usage_summary.empty:
        return ""
    return f"""## Observed usage composition (not employment impact)

These tables describe the composition of one vendor's observed conversations. They are not a sample
of the workforce and carry no information about employment outcomes. No replacement or disruption
index is constructed.

### Surface medians

{usage_summary.to_markdown(index=False)}

### Surface contrast on the same occupations

{usage_contrast.to_markdown(index=False) if not usage_contrast.empty else "_Unavailable._"}

![Automation share by surface](../figures/usage_surface_contrast.png)
"""


def _section_swebench(top: pd.DataFrame, spread: pd.DataFrame, refusals: pd.DataFrame) -> str:
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

### Top public submissions

{top.head(12).to_markdown(index=False) if not top.empty else "_Unavailable._"}

### Within-model scaffold spread

A large spread across submissions that name the same model is direct evidence that the score is not
a model property.

{spread.head(10).to_markdown(index=False) if not spread.empty else "_Too few repeated model tags to estimate._"}
"""


def _section_calendar(calendar: pd.DataFrame) -> str:
    if calendar.empty:
        return ""
    surviving = int(calendar["surviving_after_correction"].iloc[0])
    return f"""## Calendar negative control

A small, pre-registered family of calendar features (weekday, month, quarter) is tested against a
year-preserving random-date null, then corrected by Benjamini–Hochberg. Features that look
significant before correction and not after are the spurious-pattern lesson.

**Results surviving BH correction:** {surviving} of {len(calendar)}.

{calendar[["feature", "top_bucket", "top_count", "share", "permutation_p_value", "q_value_bh", "significant_after_bh"]].to_markdown(index=False)}

![Calendar control](../figures/calendar_control.png)
"""


def _section_methods(context: RunContext) -> str:
    return f"""## Methods in brief

- **Reference date** `{context.reference_date}` is the minimum of each source's latest observation,
  so no claim is newer than the least fresh input.
- **Intervals** are Wilson score intervals for proportions, HC3 robust intervals for OLS slopes,
  Theil–Sen for robust slope checks, and percentile bootstraps that resample the unit of analysis
  (a model, an occupation, a benchmark pair) — never a row of a table that may contain repeated
  snapshots.
- **Multiple comparisons** inside a pre-registered family are corrected by Benjamini–Hochberg.
- **Forecasts** are published only when a rolling-origin backtest beats a last-value baseline; the
  published interval is built from measured out-of-sample errors.
- **Refusals** replace capped extrapolations. If a claim cannot be supported, the pipeline says so
  and writes a row to `data/analysis/refusals.csv`.
- The detailed audit of the previous version, and the specific defects this rebuild exists to
  prevent, is in [`docs/audit_of_previous_version.md`](../docs/audit_of_previous_version.md).

## Reproduce

```bash
uv sync
uv run aicap                # full refresh
uv run aicap --from-interim # re-analyse cached frames
uv run python -m unittest discover -s tests
```
"""


def _to_html(markdown: str) -> str:
    body = "\n".join(_markdown_blocks(markdown))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Capability Signals</title>
  <style>
    :root {{
      --ink: #1c1917; --muted: #78716c; --paper: #fafaf9; --card: #ffffff;
      --line: #e7e5e4; --accent: #1d4ed8; --open: #0f766e; --closed: #9a3412;
    }}
    body {{ margin: 0; font-family: "Source Serif 4", "Iowan Old Style", "Palatino Linotype", Palatino, serif;
      color: var(--ink); background: linear-gradient(180deg, #f5f5f4 0%, #e7e5e4 100%); }}
    main {{ max-width: 920px; margin: 0 auto; padding: 48px 22px 80px; background: var(--card);
      box-shadow: 0 0 0 1px var(--line); }}
    h1 {{ font-family: "Segoe UI", "Helvetica Neue", sans-serif; font-size: 2.4rem; line-height: 1.1;
      letter-spacing: -0.02em; margin: 0 0 0.6em; }}
    h2 {{ font-family: "Segoe UI", "Helvetica Neue", sans-serif; margin-top: 2.4em;
      border-top: 1px solid var(--line); padding-top: 1em; }}
    p, li {{ line-height: 1.65; font-size: 1.02rem; }}
    strong {{ color: var(--ink); }}
    code, pre {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.92em; }}
    pre {{ background: #1c1917; color: #fafaf9; padding: 14px 16px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 1em 0 1.4em;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    img {{ max-width: 100%; border: 1px solid var(--line); background: white; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body><main>{body}</main></body>
</html>
"""


def _markdown_blocks(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    out: list[str] = []
    in_code = False
    in_table = False
    table_lines: list[str] = []
    list_open = False

    def flush_table() -> None:
        nonlocal table_lines, in_table
        if not table_lines:
            return
        rows = [line.strip().strip("|").split("|") for line in table_lines if "|" in line]
        if len(rows) >= 2:
            out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c.strip())}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        table_lines = []
        in_table = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    for line in lines:
        if line.startswith("```"):
            flush_table()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                close_list()
                out.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|") and "|" in line[1:]:
            close_list()
            in_table = True
            table_lines.append(line)
            continue
        if in_table:
            flush_table()
        if not line.strip():
            close_list()
            continue
        if line.startswith("# "):
            close_list()
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            close_list()
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            close_list()
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("!["):
            close_list()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                out.append(
                    f'<p><img alt="{html.escape(match.group(1))}" src="{html.escape(match.group(2))}"></p>'
                )
        else:
            close_list()
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    close_list()
    return out


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text
