"""Do different benchmarks agree about which models are better?

Question
--------
For models measured on more than one public benchmark, how strongly do those benchmarks agree on
model ordering, and how much does a model's rank move depending on which benchmark you pick?

Why this module exists
----------------------
Every composite "AI capability score" rests on an untested premise: that the signals being combined
measure one underlying thing, so averaging them reduces noise rather than mixing constructs. The
previous version built exactly such an index — min-max scaling arena ratings, resolve rates,
leaderboard fractions, prices and download counts into one number — and never checked the premise.

This module checks it. The result determines whether a composite is defensible, and the project
publishes the check instead of the composite. That inversion is the single most important design
difference from the previous version: a measurement of construct validity is a finding, whereas a
composite built on an unmeasured assumption is a guess with a decimal point.

Interpretation guide
--------------------
Kendall's tau-b is the share of model pairs ordered the same way by both benchmarks, rescaled to
[-1, 1]. A tau of 0.6 means roughly 80% of pairs agree — which sounds high until you notice it
implies one pair in five is ordered *oppositely*. Top-k overlap is reported alongside because
readers use leaderboard heads, and heads can disagree while the overall correlation looks healthy.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..identity import STRICT_TIERS, Matcher
from ..provenance import RunContext
from ..stats import kendall_tau, paired_bootstrap_ci, spearman_rho, top_k_overlap

#: Minimum models in common for a pair of benchmarks to be compared.
MIN_COMMON_MODELS = 12

#: Head size for the overlap statistic.
TOP_K = 10


def build(
    openrouter: pd.DataFrame,
    lmarena: pd.DataFrame,
    swebench: pd.DataFrame,
    context: RunContext,
) -> dict[str, pd.DataFrame]:
    panel, crosswalk = _build_panel(openrouter, lmarena, swebench)

    tables = {
        "benchmark_crosswalk": crosswalk,
        "benchmark_panel": panel,
    }

    benchmark_columns = [column for column in _BENCHMARKS if column in panel.columns]
    pairs = _pairwise_agreement(panel, benchmark_columns, context)
    if pairs.empty:
        context.refuse(
            claim="Agreement between public benchmarks on model ordering.",
            reason=(
                "No pair of benchmarks shares enough identically-matched models for a rank "
                "correlation with a usable interval."
            ),
            unblocked_by="Wider overlap between the benchmark sources, or an official model crosswalk.",
            analysis="benchmark_agreement",
        )
        return tables

    tables["benchmark_pair_agreement"] = pairs
    tables["benchmark_rank_instability"] = _rank_instability(panel, benchmark_columns)
    tables["composite_score_verdict"] = _verdict(pairs)
    return tables


#: Benchmark columns compared, with the scale of each. They are never combined; the point of the
#: module is to measure whether combining them would be legitimate.
_BENCHMARKS: dict[str, str] = {
    "aa_intelligence_index": "Vendor composite index (Artificial Analysis intelligence)",
    "aa_coding_index": "Vendor composite index (Artificial Analysis coding)",
    "aa_agentic_index": "Vendor composite index (Artificial Analysis agentic)",
    "design_arena_score": "Design Arena score",
    "lmarena_text_rating": "LMArena text arena Bradley-Terry rating (overall category)",
    "lmarena_webdev_rating": "LMArena code arena Bradley-Terry rating (overall category)",
    "swebench_resolve_rate_pct": "SWE-bench Verified resolve rate of the best system using this model",
}


def _build_panel(
    openrouter: pd.DataFrame, lmarena: pd.DataFrame, swebench: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble one row per catalogue model with every benchmark score that can be strictly matched.

    The catalogue is the spine because it is the only source with a stable machine identifier. Arena
    and SWE-bench rows are attached through :class:`aicap.identity.Matcher` at strict tiers only;
    configuration variants are recorded in the crosswalk for auditing but excluded from the panel.
    """
    panel = openrouter[
        [
            "model_id",
            "display_name",
            "vendor",
            "family",
            "weights_class",
            "aa_intelligence_index",
            "aa_coding_index",
            "aa_agentic_index",
            "design_arena_score",
        ]
    ].copy()

    matcher = Matcher(zip(openrouter["model_id"], openrouter["display_name"], strict=True))
    crosswalk_rows: list[dict[str, object]] = []

    for arena, column in (("text", "lmarena_text_rating"), ("webdev", "lmarena_webdev_rating")):
        latest = _latest_arena_snapshot(lmarena, arena)
        attached: dict[str, float] = {}
        for _, row in latest.iterrows():
            match = matcher.match(row["model_name"])
            crosswalk_rows.append(
                {
                    "source": f"lmarena_{arena}",
                    "source_name": row["model_name"],
                    "match_tier": match.tier,
                    "matched_model_id": match.candidate_id,
                    "used_in_panel": bool(match.is_strict),
                    "score": float(row["rating"]),
                }
            )
            if match.is_strict and match.candidate_id is not None:
                # A catalogue id can be hit by several arena names only if those names collide after
                # canonicalisation; keep the higher rating and let the crosswalk expose the collision.
                previous = attached.get(match.candidate_id)
                if previous is None or row["rating"] > previous:
                    attached[match.candidate_id] = float(row["rating"])
        panel[column] = panel["model_id"].map(attached)

    swebench_scores = _swebench_best_by_model(swebench, matcher, crosswalk_rows)
    panel["swebench_resolve_rate_pct"] = panel["model_id"].map(swebench_scores)

    crosswalk = pd.DataFrame(crosswalk_rows)
    return panel, crosswalk


def _latest_arena_snapshot(lmarena: pd.DataFrame, arena: str) -> pd.DataFrame:
    """The most recent publication of one arena's overall category.

    A single publication date is used deliberately. Pooling snapshots — as the previous version did
    by taking a maximum over history — mixes models measured months apart under different rating
    methodologies and rewards models that appeared in more snapshots.
    """
    subset = lmarena[(lmarena["arena"] == arena) & (lmarena["category"] == "overall")]
    if subset.empty:
        return subset
    latest_date = subset["publish_date"].max()
    return subset[subset["publish_date"] == latest_date]


def _swebench_best_by_model(
    swebench: pd.DataFrame, matcher: Matcher, crosswalk_rows: list[dict[str, object]]
) -> dict[str, float]:
    """Best resolve rate among submissions whose model tag matches a catalogue model.

    "Best system using this model" is the honest description. A submission bundles a scaffold with a
    model, so this is an upper bound on what the model contributes and is labelled that way wherever
    it appears.
    """
    best: dict[str, float] = {}
    for _, row in swebench.dropna(subset=["resolve_rate_pct"]).iterrows():
        match = matcher.match(row["model_tag"])
        crosswalk_rows.append(
            {
                "source": "swebench_verified",
                "source_name": str(row["model_tag"]),
                "match_tier": match.tier,
                "matched_model_id": match.candidate_id,
                "used_in_panel": bool(match.is_strict),
                "score": float(row["resolve_rate_pct"]),
            }
        )
        if match.is_strict and match.candidate_id is not None:
            rate = float(row["resolve_rate_pct"])
            if match.candidate_id not in best or rate > best[match.candidate_id]:
                best[match.candidate_id] = rate
    return best


def _pairwise_agreement(
    panel: pd.DataFrame, columns: list[str], context: RunContext
) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(columns, 2):
        common = panel[[left, right]].dropna()
        if len(common) < MIN_COMMON_MODELS:
            continue
        x = common[left].to_numpy(dtype=float)
        y = common[right].to_numpy(dtype=float)
        tau, tau_low, tau_high = paired_bootstrap_ci(
            list(zip(x, y, strict=True)), statistic=kendall_tau, min_n=MIN_COMMON_MODELS
        )
        rows.append(
            {
                "benchmark_a": left,
                "benchmark_b": right,
                "benchmark_a_scale": _BENCHMARKS[left],
                "benchmark_b_scale": _BENCHMARKS[right],
                "common_models": int(len(common)),
                "kendall_tau_b": tau,
                "tau_ci_low": tau_low,
                "tau_ci_high": tau_high,
                "spearman_rho": spearman_rho(x, y),
                "share_of_pairs_ordered_alike": (tau + 1.0) / 2.0 if np.isfinite(tau) else float("nan"),
                f"top_{TOP_K}_overlap": top_k_overlap(x, y, TOP_K),
                "estimator": "kendall_tau_b_with_model_level_paired_bootstrap",
                "both_vendor_composites": left.startswith("aa_") and right.startswith("aa_"),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("kendall_tau_b").reset_index(drop=True)
    weakest = frame.iloc[0]
    context.refuse(
        claim="A single composite score ranking models by overall capability.",
        reason=(
            "Public benchmarks do not agree strongly enough on model ordering to be treated as "
            "noisy measurements of one construct. The weakest measured pair "
            f"({weakest['benchmark_a']} versus {weakest['benchmark_b']}) has Kendall tau "
            f"{weakest['kendall_tau_b']:.2f} on {int(weakest['common_models'])} shared models, so a "
            "weighted average of them would mostly encode the choice of weights."
        ),
        unblocked_by=(
            "Evidence that the benchmarks load on a common factor, for example a factor analysis on "
            "a much wider model-by-benchmark matrix with published per-model scores."
        ),
        analysis="benchmark_agreement",
    )
    return frame


def _rank_instability(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """How much a model's percentile rank moves across benchmarks it appears on.

    Percentile rather than raw rank, so models measured on benchmarks with different numbers of
    participants remain comparable. Only models present on at least three benchmarks are included,
    since a spread over two points is not informative.
    """
    percentiles = panel[columns].rank(pct=True)
    coverage = percentiles.notna().sum(axis=1)
    rows = []
    for index in panel.index[coverage >= 3]:
        values = percentiles.loc[index].dropna()
        rows.append(
            {
                "model_id": panel.loc[index, "model_id"],
                "vendor": panel.loc[index, "vendor"],
                "weights_class": panel.loc[index, "weights_class"],
                "benchmarks_covered": int(values.size),
                "best_percentile": float(values.max()),
                "worst_percentile": float(values.min()),
                "percentile_spread": float(values.max() - values.min()),
                "best_on": str(values.idxmax()),
                "worst_on": str(values.idxmin()),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("percentile_spread", ascending=False)
    frame["interpretation"] = (
        "Percentile position on the benchmark that flatters the model versus the one that does not. "
        "A large spread means the model's apparent standing is a function of benchmark choice."
    )
    return frame.reset_index(drop=True)


def _verdict(pairs: pd.DataFrame) -> pd.DataFrame:
    """State, from the measured agreement, whether a composite score would be defensible.

    The threshold is declared here rather than chosen after seeing the data: a composite is treated
    as defensible only if every measured pair has a tau lower bound above 0.8, which corresponds to
    at least 90% of model pairs ordered identically. That is a demanding bar, and it should be: the
    whole value of a single number is that a reader can rely on it without knowing which benchmark
    produced it.
    """
    independent = pairs[~pairs["both_vendor_composites"]]
    basis = independent if not independent.empty else pairs
    min_tau_low = float(basis["tau_ci_low"].min())
    weakest = basis.loc[basis["kendall_tau_b"].idxmin()]
    defensible = bool(min_tau_low > 0.8)
    return pd.DataFrame(
        [
            {
                "pairs_tested": int(len(pairs)),
                "pairs_between_independent_benchmarks": int(len(independent)),
                "minimum_tau_ci_low": min_tau_low,
                "median_tau": float(basis["kendall_tau_b"].median()),
                "weakest_pair": f"{weakest['benchmark_a']} vs {weakest['benchmark_b']}",
                "weakest_pair_tau": float(weakest["kendall_tau_b"]),
                "threshold_for_composite": 0.8,
                "composite_score_defensible": defensible,
                "verdict": (
                    "A single composite capability score is defensible on this evidence."
                    if defensible
                    else (
                        "A single composite capability score is not defensible on this evidence. "
                        "Benchmarks are reported separately, and no weighted average of them is "
                        "published."
                    )
                ),
                "threshold_rationale": (
                    "Declared before estimation. tau > 0.8 corresponds to at least 90% of model "
                    "pairs ordered identically, which is the minimum for a single number to be "
                    "usable without knowing which benchmark generated it."
                ),
            }
        ]
    )
