"""What does the public SWE-bench Verified submission history show, and what does it not?

Question
--------
Across dated public submissions, how has the best resolve rate moved, and how much of the variation
is between scaffolds rather than between models?

Constraints that dominate the answer
------------------------------------
* A submission measures an agent scaffold plus a model. Two submissions using the same model can
  differ by tens of points. Scores are attributed to systems.
* Submission is voluntary, so the frontier is a lower bound on what exists.
* The directory can go long periods without new entries. When it is stale, present-tense claims are
  refused and only the historical series is published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..provenance import RunContext
from ..sources.swebench import staleness_days
from ..stats import ols_hc3

#: Days after which a dated leaderboard cannot support present-tense claims. Kept in sync with
#: :data:`aicap.analysis.data_quality.STALENESS_LIMIT_DAYS`.
STALENESS_LIMIT_DAYS = 90


def build(swebench: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    frame = swebench.dropna(subset=["submission_date", "resolve_rate_pct"]).copy()
    frame = frame.sort_values("submission_date").reset_index(drop=True)
    if frame.empty:
        context.refuse(
            claim="SWE-bench Verified progress.",
            reason="No dated submissions with resolve rates.",
            unblocked_by="Public submissions with parseable dates.",
            analysis="swebench_progress",
        )
        return {}

    stale = staleness_days(frame, context.reference_date)
    history = _frontier_history(frame)
    tables: dict[str, pd.DataFrame] = {
        "swebench_frontier_history": history,
        "swebench_top_systems": _top_systems(frame),
        "swebench_scaffold_spread": _scaffold_spread(frame),
    }

    if stale > STALENESS_LIMIT_DAYS:
        # Present-tense claims already refused in data_quality; here we additionally refuse a trend
        # extrapolation that would project a frozen leaderboard into the future.
        context.refuse(
            claim="Extrapolated future SWE-bench Verified resolve rates.",
            reason=(
                f"The public leaderboard has received no entry for {stale} days. A trend fitted to "
                "a frozen series projects silence, not progress."
            ),
            unblocked_by="New public submissions.",
            analysis="swebench_progress",
        )
    else:
        tables["swebench_trend"] = _trend(history)

    return tables


def _frontier_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Running maximum resolve rate, with the system that set each new record."""
    rows = []
    best = -1.0
    for _, row in frame.iterrows():
        rate = float(row["resolve_rate_pct"])
        is_record = rate > best
        if is_record:
            best = rate
        rows.append(
            {
                "submission_date": row["submission_date"],
                "submission": row["submission"],
                "system_label": row["system_label"],
                "model_tag": row["model_tag"],
                "resolve_rate_pct": rate,
                "running_max_resolve_rate_pct": best,
                "sets_new_record": is_record,
                "uses_open_weights_model": row["uses_open_weights_model"],
            }
        )
    return pd.DataFrame(rows)


def _top_systems(frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    top = frame.sort_values("resolve_rate_pct", ascending=False).head(top_n)
    return top[
        [
            "submission",
            "submission_date",
            "system_label",
            "model_tag",
            "vendor",
            "family",
            "resolve_rate_pct",
            "uses_open_weights_model",
            "source_url",
        ]
    ].reset_index(drop=True)


def _scaffold_spread(frame: pd.DataFrame, min_submissions: int = 3) -> pd.DataFrame:
    """Within-model spread across scaffolds, for models that appear in several submissions.

    A large within-model spread is direct evidence that the score is not a model property. This is
    the quantitative version of the attribution caveat.
    """
    # Group by the cleaned model tag. Empty tags are uninformative.
    usable = frame[frame["model_tag"].astype(str).str.len() > 0]
    rows = []
    for model_tag, group in usable.groupby("model_tag"):
        if len(group) < min_submissions:
            continue
        rates = group["resolve_rate_pct"].to_numpy(dtype=float)
        rows.append(
            {
                "model_tag": model_tag,
                "submissions": int(len(group)),
                "distinct_system_labels": int(group["system_label"].nunique()),
                "min_resolve_rate_pct": float(rates.min()),
                "max_resolve_rate_pct": float(rates.max()),
                "spread_pp": float(rates.max() - rates.min()),
                "median_resolve_rate_pct": float(np.median(rates)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_tag",
                "submissions",
                "distinct_system_labels",
                "min_resolve_rate_pct",
                "max_resolve_rate_pct",
                "spread_pp",
                "median_resolve_rate_pct",
            ]
        )
    out = pd.DataFrame(rows).sort_values("spread_pp", ascending=False).reset_index(drop=True)
    out["interpretation"] = (
        "Spread in resolve rate across submissions that name the same model. A large spread means "
        "the score is dominated by the agent scaffold, not by the model."
    )
    return out


def _trend(history: pd.DataFrame) -> pd.DataFrame:
    """OLS trend on the running maximum, reported only when the series is still being updated."""
    records = history[history["sets_new_record"]].copy()
    if len(records) < 4:
        return pd.DataFrame()
    days = (
        records["submission_date"] - records["submission_date"].min()
    ).dt.days.to_numpy(dtype=float)
    rates = records["running_max_resolve_rate_pct"].to_numpy(dtype=float)
    fit = ols_hc3(days, rates)
    return pd.DataFrame(
        [
            {
                "record_setting_submissions": int(len(records)),
                "window_start": records["submission_date"].min(),
                "window_end": records["submission_date"].max(),
                "points_per_year": fit.slope * 365.25,
                "ci_low": fit.slope_low * 365.25,
                "ci_high": fit.slope_high * 365.25,
                "r_squared": fit.r_squared,
                "estimator": "ols_hc3_on_record_setting_submissions",
                "caveat": (
                    "A running maximum is monotone by construction, so a positive slope is almost "
                    "guaranteed. The magnitude is informative; the sign is not a finding."
                ),
            }
        ]
    )
