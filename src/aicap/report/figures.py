"""Figures for the published report.

Each figure is a direct plot of an analysis table. No computation happens here beyond what is
needed to draw: reading a CSV, sorting, and labelling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis import TableWriter

PALETTE = {
    "ink": "#1c1917",
    "muted": "#78716c",
    "open": "#0f766e",
    "closed": "#9a3412",
    "accent": "#1d4ed8",
    "grid": "#e7e5e4",
    "band": "#dbeafe",
    "gold": "#b45309",
    "violet": "#6d28d9",
    "paper": "#fafaf9",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _safe_read(writer: TableWriter, name: str) -> pd.DataFrame:
    path = writer.directory / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(name)
    return writer.read(name)


def render_all(writer: TableWriter, figures_dir: Path) -> list[Path]:
    _style()
    written: list[Path] = []
    renderers = (
        ("source_freshness", _source_freshness),
        ("disclosure_open_vs_closed", _disclosure),
        ("disclosure_by_year", _disclosure_by_year),
        ("disclosure_by_vendor", _disclosure_by_vendor),
        ("compute_frontier", _compute_frontier),
        ("compute_sensitivity", _compute_sensitivity),
        ("compute_backtest_skill", _compute_backtest),
        ("price_quality_pareto", _price_pareto),
        ("price_spread_bands", _price_spread),
        ("price_cheapest_ladder", _price_ladder),
        ("price_by_weights", _price_by_weights),
        ("benchmark_agreement", _benchmark_agreement),
        ("benchmark_rank_instability", _rank_instability),
        ("open_closed_gap", _open_closed_gap),
        ("open_closed_gap_trend", _gap_trend),
        ("open_closed_gap_by_category", _gap_by_category),
        ("open_weights_lag_km", _lag_km),
        ("usage_surface_contrast", _usage_contrast),
        ("usage_top_occupations", _usage_top),
        ("usage_major_groups", _usage_major),
        ("swebench_frontier", _swebench_frontier),
        ("swebench_top_systems", _swebench_top),
        ("calendar_control", _calendar),
        ("calendar_weekday_counts", _calendar_counts),
        ("refusals_overview", _refusals),
    )
    for name, renderer in renderers:
        try:
            path = figures_dir / f"{name}.png"
            renderer(writer, path)
            if path.exists():
                written.append(path)
        except FileNotFoundError:
            continue
        except Exception as exc:  # keep the report useful even if one chart fails
            print(f"figure {name} skipped: {exc}", flush=True)
            plt.close("all")
    return written


def _source_freshness(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "source_freshness").sort_values("days_behind_reference")
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    colors = [PALETTE["accent"] if flag else PALETTE["gold"] for flag in frame["is_freshest"]]
    ax.barh(frame["source_id"], frame["days_behind_reference"], color=colors)
    ax.set_xlabel("Days behind reference date")
    ax.set_title("Source freshness — how far each input lags the as-of date")
    _save(path)


def _disclosure(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "disclosure_open_vs_closed")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(frame))
    ax.barh(y - 0.18, frame["open_disclosure_rate"], height=0.35, color=PALETTE["open"], label="Open weights")
    ax.barh(y + 0.18, frame["closed_disclosure_rate"], height=0.35, color=PALETTE["closed"], label="Closed weights")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["field_label"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of models disclosing the field")
    ax.set_title("Open-weight releases disclose more than closed-weight releases")
    ax.legend(loc="lower right")
    _save(path)


def _disclosure_by_year(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "disclosure_by_year")
    fields = ["parameters", "training_compute_flop", "training_tokens", "training_hardware"]
    labels = {
        "parameters": "Parameters",
        "training_compute_flop": "Training compute",
        "training_tokens": "Training tokens",
        "training_hardware": "Hardware",
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for field in fields:
        subset = frame[frame["field"] == field].sort_values("publication_year")
        if subset.empty:
            continue
        ax.plot(subset["publication_year"], subset["disclosure_rate"], marker="o", label=labels[field])
        ax.fill_between(subset["publication_year"], subset["ci_low"], subset["ci_high"], alpha=0.12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Disclosure rate")
    ax.set_title("Disclosure rates over time (Wilson intervals shaded)")
    ax.legend(loc="lower right", ncol=2)
    _save(path)


def _disclosure_by_vendor(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "disclosure_by_vendor").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(frame["vendor"], frame["disclosure_completeness"], color=PALETTE["accent"], alpha=0.9)
    ax.errorbar(
        frame["disclosure_completeness"],
        np.arange(len(frame)),
        xerr=[
            frame["disclosure_completeness"] - frame["ci_low"],
            frame["ci_high"] - frame["disclosure_completeness"],
        ],
        fmt="none",
        ecolor=PALETTE["ink"],
        capsize=2,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Disclosure completeness")
    ax.set_title("Top vendors by disclosure completeness")
    _save(path)


def _compute_frontier(writer: TableWriter, path: Path) -> None:
    frontier = _safe_read(writer, "compute_frontier_by_year")
    forecast = (
        _safe_read(writer, "compute_frontier_forecast")
        if (writer.directory / "compute_frontier_forecast.csv").exists()
        else pd.DataFrame()
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(
        frontier["publication_year"],
        frontier["log10_compute_p90"],
        marker="o",
        color=PALETTE["accent"],
        label="Observed p90 of disclosed compute",
    )
    ax.plot(
        frontier["publication_year"],
        frontier["log10_compute_max"],
        marker=".",
        linestyle=":",
        color=PALETTE["muted"],
        label="Observed max",
    )
    if not forecast.empty:
        ax.fill_between(
            forecast["target_year"], forecast["pi_low"], forecast["pi_high"], color=PALETTE["band"], label="Backtested PI"
        )
        ax.plot(
            forecast["target_year"],
            forecast["log10_compute_p90_forecast"],
            marker="s",
            linestyle="--",
            color=PALETTE["accent"],
            label="Backtested expectation",
        )
    ax.set_xlabel("Publication year")
    ax.set_ylabel("log10 training compute (FLOP)")
    ax.set_title("Disclosed training-compute frontier")
    ax.legend(loc="upper left")
    _save(path)


def _compute_sensitivity(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "compute_selection_sensitivity")
    frame = frame[frame["status"] == "estimated"].copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(frame))
    ax.hlines(y, frame["ci_low"], frame["ci_high"], color=PALETTE["accent"])
    ax.plot(frame["log10_flop_per_year"], y, "o", color=PALETTE["accent"])
    ax.set_yticks(y)
    ax.set_yticklabels(frame["restriction"].str.replace("_", " "))
    ax.set_xlabel("log10 FLOP per year")
    ax.set_title("Compute-growth slope under selection restrictions")
    _save(path)


def _compute_backtest(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "compute_forecast_backtest")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(frame))
    width = 0.35
    ax.bar(x - width / 2, frame["mae_log10"], width, label="Trend MAE", color=PALETTE["accent"])
    ax.bar(x + width / 2, frame["baseline_mae_log10"], width, label="Last-value MAE", color=PALETTE["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}y" for h in frame["horizon_years"]])
    ax.set_ylabel("MAE (log10 FLOP)")
    ax.set_title("Does the compute trend beat a last-value baseline?")
    ax.legend()
    _save(path)


def _price_pareto(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "price_quality_frontier")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    dominated = frame[~frame["is_pareto_efficient"]]
    efficient = frame[frame["is_pareto_efficient"]]
    ax.scatter(
        dominated["blended_usd_per_call"],
        dominated["aa_intelligence_index"],
        s=28,
        alpha=0.4,
        color=PALETTE["muted"],
        label="Dominated",
    )
    colors = [PALETTE["open"] if flag else PALETTE["closed"] for flag in efficient["weights_available"]]
    ax.scatter(
        efficient["blended_usd_per_call"],
        efficient["aa_intelligence_index"],
        s=60,
        color=colors,
        edgecolors=PALETTE["ink"],
        linewidths=0.6,
        label="Pareto efficient",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Blended list price per call (USD, log)")
    ax.set_ylabel("Vendor intelligence index")
    ax.set_title("Price–quality Pareto frontier (same-record quality and price)")
    ax.legend(loc="lower right")
    _save(path)


def _price_spread(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "price_spread_at_matched_quality")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    x = np.arange(len(frame))
    ax.bar(x, frame["max_min_price_ratio"], color=PALETTE["accent"], alpha=0.85)
    ax.errorbar(
        x,
        frame["max_min_price_ratio"],
        yerr=[
            frame["max_min_price_ratio"] - frame["ratio_ci_low"],
            frame["ratio_ci_high"] - frame["max_min_price_ratio"],
        ],
        fmt="none",
        ecolor=PALETTE["ink"],
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["quality_band"], rotation=30, ha="right")
    ax.set_ylabel("Max / min list-price ratio")
    ax.set_title("Price spread among models of comparable measured quality")
    _save(path)


def _price_ladder(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "cheapest_at_quality_threshold")
    frame = frame[frame["status"] == "estimated"].copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(frame["quality_threshold"], frame["cheapest_blended_usd_per_call"], marker="o", color=PALETTE["accent"])
    for _, row in frame.iterrows():
        ax.annotate(
            str(row["cheapest_model_id"]).split("/")[-1][:18],
            (row["quality_threshold"], row["cheapest_blended_usd_per_call"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=PALETTE["muted"],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Minimum vendor intelligence index")
    ax.set_ylabel("Cheapest blended list price per call (USD, log)")
    ax.set_title("Cheapest listed model at each quality threshold")
    _save(path)


def _price_by_weights(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "price_distribution_by_weights")
    subset = frame[frame["metric"] == "blended_usd_per_call"].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(subset))
    colors = {
        "weights_published": PALETTE["open"],
        "weights_not_published": PALETTE["closed"],
        "all_listed": PALETTE["accent"],
    }
    ax.bar(x, subset["median"], color=[colors.get(g, PALETTE["muted"]) for g in subset["group"]])
    ax.errorbar(
        x,
        subset["median"],
        yerr=[subset["median"] - subset["ci_low"], subset["ci_high"] - subset["median"]],
        fmt="none",
        ecolor=PALETTE["ink"],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["group"].str.replace("_", "\n"))
    ax.set_ylabel("Median blended list price per call (USD)")
    ax.set_title("Catalogue price levels by weight availability")
    _save(path)


def _benchmark_agreement(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "benchmark_pair_agreement")
    if frame.empty:
        return
    independent = frame[~frame["both_vendor_composites"]].copy()
    if independent.empty:
        independent = frame.copy()
    independent = independent.sort_values("kendall_tau_b")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(independent))
    ax.hlines(y, independent["tau_ci_low"], independent["tau_ci_high"], color=PALETTE["accent"])
    ax.plot(independent["kendall_tau_b"], y, "o", color=PALETTE["accent"])
    ax.axvline(0.8, color=PALETTE["closed"], linestyle="--", label="Composite threshold (0.8)")
    labels = [
        f"{a.replace('lmarena_', '').replace('_rating', '').replace('_index', '')} vs "
        f"{b.replace('lmarena_', '').replace('_rating', '').replace('_index', '')}"
        for a, b in zip(independent["benchmark_a"], independent["benchmark_b"], strict=True)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Kendall τ-b (model-level bootstrap CI)")
    ax.set_title("Do public benchmarks agree about model ordering?")
    ax.legend(loc="lower right")
    _save(path)


def _rank_instability(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "benchmark_rank_instability").head(15).iloc[::-1]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.hlines(np.arange(len(frame)), frame["worst_percentile"], frame["best_percentile"], color=PALETTE["accent"])
    ax.plot(frame["worst_percentile"], np.arange(len(frame)), "o", color=PALETTE["closed"], label="Worst percentile")
    ax.plot(frame["best_percentile"], np.arange(len(frame)), "o", color=PALETTE["open"], label="Best percentile")
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["model_id"].str.replace("openai/", "").str.replace("anthropic/", "").str.slice(0, 28))
    ax.set_xlabel("Percentile rank across benchmarks")
    ax.set_title("Models whose standing depends most on which benchmark you pick")
    ax.legend(loc="lower right")
    _save(path)


def _open_closed_gap(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "arena_frontier_by_date")
    if frame.empty:
        return
    latest_regime = frame.sort_values("publish_date")["methodology_regime"].iloc[-1]
    window = frame[frame["methodology_regime"] == latest_regime]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(window["publish_date"], window["closed_best_rating"], color=PALETTE["closed"], label="Closed frontier")
    ax.plot(window["publish_date"], window["open_best_rating"], color=PALETTE["open"], label="Open-weight frontier")
    ax.fill_between(
        window["publish_date"],
        window["open_best_rating"],
        window["closed_best_rating"],
        color=PALETTE["band"],
        alpha=0.7,
        label="Gap",
    )
    ax.set_xlabel("Leaderboard publication date")
    ax.set_ylabel("Arena rating (Bradley–Terry)")
    ax.set_title(f"Open-weight vs closed frontier — {latest_regime}")
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    _save(path)


def _gap_trend(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "open_closed_gap_trend")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(frame))
    ax.hlines(y, frame["ci_low"], frame["ci_high"], color=PALETTE["accent"])
    ax.plot(frame["gap_change_per_year_ols"], y, "o", color=PALETTE["accent"], label="OLS HC3")
    ax.plot(frame["gap_change_per_year_theil_sen"], y, "s", color=PALETTE["gold"], label="Theil–Sen")
    ax.axvline(0, color=PALETTE["muted"], linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["methodology_regime"].str.replace("bradley_terry_", "").str.replace("_", " "))
    ax.set_xlabel("Gap change per year (rating points)")
    ax.set_title("Is the open–closed gap closing? (within rating regimes)")
    ax.legend(loc="best")
    _save(path)


def _gap_by_category(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "open_closed_gap_by_category")
    text = frame[frame["arena"] == "text"].sort_values("gap_rating", ascending=False).head(12).iloc[::-1]
    if text.empty:
        text = frame.sort_values("gap_rating", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    colors = [PALETTE["closed"] if flag else PALETTE["muted"] for flag in text["gap_distinguishable_from_zero"]]
    ax.barh(text["category"].str.replace("_", " ").str.slice(0, 42), text["gap_rating"], color=colors)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlabel("Closed − open best rating")
    ax.set_title("Open–closed gap by category (latest publication)")
    _save(path)


def _lag_km(writer: TableWriter, path: Path) -> None:
    lag = _safe_read(writer, "open_weights_lag")
    summary = _safe_read(writer, "open_weights_lag_summary")
    if lag.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    completed = lag[~lag["is_right_censored"]]["lag_days"]
    censored = lag[lag["is_right_censored"]]["lag_days"]
    bins = np.linspace(0, max(lag["lag_days"].max(), 1), 20)
    ax.hist(completed, bins=bins, color=PALETTE["open"], alpha=0.85, label="Reached")
    ax.hist(censored, bins=bins, color=PALETTE["closed"], alpha=0.45, label="Still unmatched (censored)")
    if not summary.empty and pd.notna(summary.iloc[0]["kaplan_meier_median_lag_days"]):
        ax.axvline(
            summary.iloc[0]["kaplan_meier_median_lag_days"],
            color=PALETTE["accent"],
            linestyle="--",
            label=f"KM median = {summary.iloc[0]['kaplan_meier_median_lag_days']:.0f}d",
        )
        ax.axvline(
            summary.iloc[0]["naive_median_of_completed_lags_days"],
            color=PALETTE["gold"],
            linestyle=":",
            label=f"Naive median = {summary.iloc[0]['naive_median_of_completed_lags_days']:.0f}d",
        )
    ax.set_xlabel("Days for open weights to reach a closed-frontier level")
    ax.set_ylabel("Closed-frontier levels")
    ax.set_title("Open-weight catch-up lag (right-censored levels retained)")
    ax.legend(loc="upper right")
    _save(path)


def _usage_contrast(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "usage_surface_summary")
    auto = frame[frame["metric"] == "automation_share_pct"]
    if auto.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(auto))
    ax.bar(x, auto["median"], color=[PALETTE["closed"], PALETTE["open"]][: len(auto)], alpha=0.9)
    ax.errorbar(
        x,
        auto["median"],
        yerr=[auto["median"] - auto["ci_low"], auto["ci_high"] - auto["median"]],
        fmt="none",
        ecolor=PALETTE["ink"],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(auto["surface"])
    ax.set_ylabel("Median automation share across occupations (%)")
    ax.set_title("Same vendor, two surfaces: automation share disagrees")
    _save(path)


def _usage_top(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "usage_top_occupations")
    # Prefer the consumer surface for the headline chart; fall back to whichever exists.
    preferred = frame[frame["surface"] == "claude_ai"]
    if preferred.empty:
        preferred = frame
    top = preferred.sort_values("usage_share_pct", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.barh(top["occupation"].str.slice(0, 40), top["usage_share_pct"], color=PALETTE["accent"])
    ax.set_xlabel("Share of observed usage (%)")
    ax.set_title(f"Top occupations by usage share — {top['surface'].iloc[0]}")
    _save(path)


def _usage_major(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "usage_major_group_composition")
    preferred = frame[frame["surface"] == "claude_ai"]
    if preferred.empty:
        preferred = frame
    top = preferred.sort_values("usage_share_pct", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(top["soc_major_group"].astype(str), top["usage_share_pct"], color=PALETTE["violet"], alpha=0.85)
    ax.set_xlabel("Usage share (%)")
    ax.set_title("Usage composition by SOC major group")
    _save(path)


def _swebench_frontier(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "swebench_frontier_history")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.scatter(
        pd.to_datetime(frame["submission_date"]),
        frame["resolve_rate_pct"],
        s=18,
        alpha=0.35,
        color=PALETTE["muted"],
        label="Submission",
    )
    ax.plot(
        pd.to_datetime(frame["submission_date"]),
        frame["running_max_resolve_rate_pct"],
        color=PALETTE["accent"],
        label="Running max",
    )
    ax.set_ylabel("Resolve rate (%)")
    ax.set_xlabel("Submission date")
    ax.set_title("SWE-bench Verified public frontier (historical)")
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    _save(path)


def _swebench_top(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "swebench_top_systems").head(12).iloc[::-1]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    labels = frame["system_label"].astype(str).str.slice(0, 42)
    colors = [
        PALETTE["open"] if flag is True else PALETTE["closed"] if flag is False else PALETTE["muted"]
        for flag in frame["uses_open_weights_model"]
    ]
    ax.barh(labels, frame["resolve_rate_pct"], color=colors)
    ax.set_xlabel("Resolve rate (%)")
    ax.set_title("Top SWE-bench Verified public submissions")
    _save(path)


def _calendar(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "calendar_release_tests")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    y = np.arange(len(frame))
    colors = [PALETTE["closed"] if flag else PALETTE["accent"] for flag in frame["significant_after_bh"]]
    ax.barh(y, frame["permutation_p_value"], color=colors)
    ax.axvline(0.05, color=PALETTE["muted"], linestyle="--", label="Nominal α = 0.05")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.feature}: {row.top_bucket}" for row in frame.itertuples()])
    ax.set_xlabel("Permutation p-value")
    ax.set_title("Calendar clustering of releases (filled = survives BH)")
    ax.legend(loc="lower right")
    _save(path)


def _calendar_counts(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "calendar_release_counts")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    subset = frame[frame["feature"] == "weekday"].set_index("bucket").reindex(weekdays).fillna(0)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(weekdays, subset["releases"], color=PALETTE["accent"])
    ax.set_ylabel("Releases")
    ax.set_title("Notable model releases by weekday")
    plt.xticks(rotation=20, ha="right")
    _save(path)


def _refusals(writer: TableWriter, path: Path) -> None:
    frame = _safe_read(writer, "refusals")
    if frame.empty:
        return
    counts = frame["analysis"].value_counts().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.barh(counts.index, counts.values, color=PALETTE["gold"])
    ax.set_xlabel("Refused claims")
    ax.set_title("Where the pipeline refuses to invent a number")
    _save(path)
