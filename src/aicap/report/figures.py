"""Figures for the published report.

Each figure is a direct plot of an analysis table. No computation happens here beyond what is
needed to draw: reading a CSV, sorting, and labelling. That keeps the report layer from drifting
away from the analysis layer — a figure that cannot be regenerated from a table is a figure that
should not exist.
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
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafaf9",
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


def render_all(writer: TableWriter, figures_dir: Path) -> list[Path]:
    _style()
    written: list[Path] = []
    for name, renderer in (
        ("disclosure_open_vs_closed", _disclosure),
        ("compute_frontier", _compute_frontier),
        ("price_quality_pareto", _price_pareto),
        ("price_spread_bands", _price_spread),
        ("benchmark_agreement", _benchmark_agreement),
        ("open_closed_gap", _open_closed_gap),
        ("usage_surface_contrast", _usage_contrast),
        ("calendar_control", _calendar),
    ):
        try:
            path = figures_dir / f"{name}.png"
            renderer(writer, path)
            if path.exists():
                written.append(path)
        except FileNotFoundError:
            # A missing table means the analysis refused; skip the figure rather than inventing one.
            continue
    return written


def _disclosure(writer: TableWriter, path: Path) -> None:
    frame = writer.read("disclosure_open_vs_closed")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(frame))
    ax.barh(
        y - 0.18,
        frame["open_disclosure_rate"],
        height=0.35,
        color=PALETTE["open"],
        label="Open weights",
    )
    ax.barh(
        y + 0.18,
        frame["closed_disclosure_rate"],
        height=0.35,
        color=PALETTE["closed"],
        label="Closed weights",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["field_label"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of models disclosing the field")
    ax.set_title("Open-weight releases disclose more than closed-weight releases")
    ax.legend(loc="lower right")
    _save(path)


def _compute_frontier(writer: TableWriter, path: Path) -> None:
    frontier = writer.read("compute_frontier_by_year")
    forecast = writer.read("compute_frontier_forecast") if (writer.directory / "compute_frontier_forecast.csv").exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(
        frontier["publication_year"],
        frontier["log10_compute_p90"],
        marker="o",
        color=PALETTE["accent"],
        label="Observed p90 of disclosed compute",
    )
    if not forecast.empty:
        ax.fill_between(
            forecast["target_year"],
            forecast["pi_low"],
            forecast["pi_high"],
            color=PALETTE["band"],
            label="Backtested prediction interval",
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


def _price_pareto(writer: TableWriter, path: Path) -> None:
    frame = writer.read("price_quality_frontier")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    dominated = frame[~frame["is_pareto_efficient"]]
    efficient = frame[frame["is_pareto_efficient"]]
    ax.scatter(
        dominated["blended_usd_per_call"],
        dominated["aa_intelligence_index"],
        s=28,
        alpha=0.45,
        color=PALETTE["muted"],
        label="Dominated",
    )
    colors = [PALETTE["open"] if flag else PALETTE["closed"] for flag in efficient["weights_available"]]
    ax.scatter(
        efficient["blended_usd_per_call"],
        efficient["aa_intelligence_index"],
        s=55,
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
    frame = writer.read("price_spread_at_matched_quality")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
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


def _benchmark_agreement(writer: TableWriter, path: Path) -> None:
    frame = writer.read("benchmark_pair_agreement")
    if frame.empty:
        return
    independent = frame[~frame["both_vendor_composites"]].copy()
    if independent.empty:
        independent = frame.copy()
    independent = independent.sort_values("kendall_tau_b")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
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


def _open_closed_gap(writer: TableWriter, path: Path) -> None:
    frame = writer.read("arena_frontier_by_date")
    if frame.empty:
        return
    # Plot only the most recent methodology regime so the visual does not cross a break.
    latest_regime = frame.sort_values("publish_date")["methodology_regime"].iloc[-1]
    window = frame[frame["methodology_regime"] == latest_regime]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
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


def _usage_contrast(writer: TableWriter, path: Path) -> None:
    frame = writer.read("usage_surface_summary")
    if frame.empty:
        return
    auto = frame[frame["metric"] == "automation_share_pct"]
    if auto.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
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


def _calendar(writer: TableWriter, path: Path) -> None:
    frame = writer.read("calendar_release_tests")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(frame))
    colors = [PALETTE["closed"] if flag else PALETTE["accent"] for flag in frame["significant_after_bh"]]
    ax.barh(y, frame["permutation_p_value"], color=colors)
    ax.axvline(0.05, color=PALETTE["muted"], linestyle="--", label="Nominal α = 0.05")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.feature}: {row.top_bucket}" for row in frame.itertuples()]
    )
    ax.set_xlabel("Permutation p-value")
    ax.set_title("Calendar clustering of releases (red = survives BH correction)")
    ax.legend(loc="lower right")
    _save(path)
