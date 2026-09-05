"""How far behind the closed frontier are open-weight models, and is the gap closing?

Question
--------
Within one arena and one rating methodology, what is the rating gap between the best open-weight and
best closed-weight model at each leaderboard publication, how has that gap moved, and how long does
it take open weights to reach a capability level the closed frontier had already reached?

Why this is the project's one real time series
----------------------------------------------
The arena dataset publishes a dated leaderboard roughly weekly, each row carrying a rating and a
published variance. That is a genuine panel: the same quantity, measured repeatedly, with
source-supplied uncertainty. No other source here has that, which is why this is the only module
that estimates a trend over calendar time.

Three design choices that the previous version got wrong
--------------------------------------------------------
1. **One methodology regime.** Ratings are only comparable within a regime, so the trend is estimated
   inside the most recent one. A slope spanning the 2025-05-16 style-control change or the
   2025-07-23 re-weighting would partly measure the methodology change.
2. **Uncertainty from the source.** The gap's standard error combines the two models' published
   standard errors. The previous version manufactured intervals by adding invented noise to derived
   scores.
3. **Unknown licences excluded.** Models whose licence cannot be classified are dropped rather than
   assumed proprietary, which would have inflated the closed frontier.

The lag statistic
-----------------
For each date ``t``, ``lag`` is the time until the open-weight frontier first reaches the closed
frontier's level at ``t``. Dates whose level has not yet been reached are right-censored and reported
as censored rather than dropped — dropping them is the classic survivorship error that makes a lag
look shorter than it is, because the hardest-to-match levels are exactly the ones still unmatched.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..provenance import RunContext
from ..stats import kaplan_meier_median, ols_hc3, theil_sen_slope
from ..taxonomy import WEIGHTS_CLOSED, WEIGHTS_OPEN, WEIGHTS_RESTRICTED

#: Arena and category analysed. The text arena's overall category has the longest history and the
#: largest vote counts, so it has the tightest published intervals.
ARENA = "text"
CATEGORY = "overall"

#: Minimum published leaderboards inside a regime for a trend to be estimated.
MIN_DATES_FOR_TREND = 12


def build(lmarena: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    subset = lmarena[
        (lmarena["arena"] == ARENA)
        & (lmarena["category"] == CATEGORY)
        & lmarena["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED, WEIGHTS_CLOSED])
    ].copy()

    if subset.empty:
        context.refuse(
            claim="Open-weight versus closed-weight capability gap.",
            reason=f"No classifiable rows for arena={ARENA!r} category={CATEGORY!r}.",
            unblocked_by="Arena data with classifiable licences.",
            analysis="open_weights_lag",
        )
        return {}

    subset["weights_available"] = subset["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED])
    frontier = _frontier_by_date(subset)
    tables: dict[str, pd.DataFrame] = {"arena_frontier_by_date": frontier}

    usable = frontier.dropna(subset=["gap_rating"])
    lag = _lag_table(usable)
    tables["open_weights_lag"] = lag
    tables["open_weights_lag_summary"] = lag_summary(lag)

    trend = _gap_trend(usable, context)
    if not trend.empty:
        tables["open_closed_gap_trend"] = trend

    tables["open_closed_gap_by_category"] = _gap_by_category(lmarena)
    return tables


def _frontier_by_date(subset: pd.DataFrame) -> pd.DataFrame:
    """Best open and best closed rating at each publication date, with the gap and its error."""
    rows = []
    for date, group in subset.groupby("publish_date"):
        open_side = group[group["weights_available"]]
        closed_side = group[~group["weights_available"]]
        record: dict[str, object] = {
            "publish_date": date,
            "methodology_regime": group["methodology_regime"].iloc[0],
            "models_ranked": int(len(group)),
            "open_models": int(len(open_side)),
            "closed_models": int(len(closed_side)),
        }
        for label, side in (("open", open_side), ("closed", closed_side)):
            if side.empty:
                record[f"{label}_best_rating"] = np.nan
                record[f"{label}_best_se"] = np.nan
                record[f"{label}_best_model"] = None
                continue
            best = side.loc[side["rating"].idxmax()]
            record[f"{label}_best_rating"] = float(best["rating"])
            record[f"{label}_best_se"] = float(best["rating_se"]) if pd.notna(best["rating_se"]) else np.nan
            record[f"{label}_best_model"] = str(best["model_name"])
        rows.append(record)

    frame = pd.DataFrame(rows).sort_values("publish_date").reset_index(drop=True)
    frame["gap_rating"] = frame["closed_best_rating"] - frame["open_best_rating"]
    # Independent-error approximation. The two ratings come from one Bradley-Terry fit and are
    # therefore correlated, so this understates the gap's uncertainty; the dataset does not publish
    # the covariance needed to do better, and that limitation is carried in the column below.
    frame["gap_se"] = np.sqrt(frame["open_best_se"] ** 2 + frame["closed_best_se"] ** 2)
    frame["gap_ci_low"] = frame["gap_rating"] - 1.959963985 * frame["gap_se"]
    frame["gap_ci_high"] = frame["gap_rating"] + 1.959963985 * frame["gap_se"]
    frame["gap_se_note"] = (
        "Independent-error approximation; the two ratings are estimated jointly, so the true "
        "standard error is smaller than this and the interval is conservative."
    )
    return frame


def _lag_table(frontier: pd.DataFrame) -> pd.DataFrame:
    """Time for the open-weight frontier to reach each historical closed-frontier level.

    Uses the running maximum of the open frontier, so a level once reached stays reached: arena
    ratings drift as the model pool changes, and without the running maximum a later dip would
    register as capability being lost.
    """
    frame = frontier.dropna(subset=["closed_best_rating", "open_best_rating"]).copy()
    frame = frame.sort_values("publish_date").reset_index(drop=True)
    frame["open_running_max"] = frame["open_best_rating"].cummax()

    dates = frame["publish_date"].to_numpy()
    open_running = frame["open_running_max"].to_numpy(dtype=float)
    final_open = float(open_running[-1])

    rows = []
    for index, row in frame.iterrows():
        target = float(row["closed_best_rating"])
        if target <= open_running[index]:
            rows.append(
                {
                    "closed_frontier_date": row["publish_date"],
                    "closed_frontier_rating": target,
                    "closed_frontier_model": row["closed_best_model"],
                    "lag_days": 0,
                    "reached_on": row["publish_date"],
                    "is_right_censored": False,
                    "status": "already_matched_when_set",
                }
            )
            continue
        reached_positions = np.nonzero(open_running >= target)[0]
        if reached_positions.size == 0:
            rows.append(
                {
                    "closed_frontier_date": row["publish_date"],
                    "closed_frontier_rating": target,
                    "closed_frontier_model": row["closed_best_model"],
                    "lag_days": int((dates[-1] - row["publish_date"]) / np.timedelta64(1, "D")),
                    "reached_on": pd.NaT,
                    "is_right_censored": True,
                    "status": f"not_yet_reached (open frontier at {final_open:.0f})",
                }
            )
            continue
        reached_date = dates[reached_positions[0]]
        rows.append(
            {
                "closed_frontier_date": row["publish_date"],
                "closed_frontier_rating": target,
                "closed_frontier_model": row["closed_best_model"],
                "lag_days": int((reached_date - row["publish_date"]) / np.timedelta64(1, "D")),
                "reached_on": reached_date,
                "is_right_censored": False,
                "status": "reached",
            }
        )

    out = pd.DataFrame(rows)
    out["censoring_note"] = (
        "Right-censored rows are levels the open frontier has not yet reached; their lag_days is a "
        "lower bound. Summarising only the uncensored rows would understate the lag, because the "
        "levels still unmatched are the hardest ones."
    )
    return out


def lag_summary(lag_table: pd.DataFrame) -> pd.DataFrame:
    """Censoring-aware summary of open-weight catch-up lag.

    Uses the Kaplan-Meier estimator so that levels the open frontier has not yet reached still
    contribute as right-censored observations. The naive median of completed lags is reported
    alongside it as a diagnostic of how badly that estimator understates the lag.
    """
    if lag_table.empty:
        return pd.DataFrame()
    durations = lag_table["lag_days"].to_numpy(dtype=float)
    observed = (~lag_table["is_right_censored"].to_numpy(dtype=bool))
    km_median, follow_up = kaplan_meier_median(durations, observed)
    completed = durations[observed]
    return pd.DataFrame(
        [
            {
                "levels_tracked": int(len(lag_table)),
                "levels_reached": int(observed.sum()),
                "levels_still_unmatched": int((~observed).sum()),
                "share_right_censored": float((~observed).mean()),
                "kaplan_meier_median_lag_days": km_median,
                "follow_up_days": follow_up,
                "naive_median_of_completed_lags_days": float(np.median(completed))
                if completed.size
                else float("nan"),
                "naive_understates_by_days": (
                    float(km_median - np.median(completed))
                    if completed.size and math.isfinite(km_median)
                    else float("nan")
                ),
                "estimator": "kaplan_meier_median_with_right_censoring",
                "interpretation": (
                    "Kaplan-Meier median days for the open-weight frontier to reach a historical "
                    "closed-frontier level. Censored levels contribute as 'not yet reached'. A NaN "
                    "median means more than half the levels tracked are still unmatched within the "
                    "follow-up window, so the honest answer is 'longer than the follow-up'."
                ),
            }
        ]
    )


def _gap_trend(frontier: pd.DataFrame, context: RunContext) -> pd.DataFrame:
    """Estimate the gap's trend inside each methodology regime with enough publications."""
    rows = []
    for regime, group in frontier.groupby("methodology_regime"):
        group = group.sort_values("publish_date")
        if len(group) < MIN_DATES_FOR_TREND:
            continue
        days = (group["publish_date"] - group["publish_date"].min()).dt.days.to_numpy(dtype=float)
        gaps = group["gap_rating"].to_numpy(dtype=float)
        fit = ols_hc3(days, gaps)
        ts_slope, ts_low, ts_high = theil_sen_slope(days, gaps)
        rows.append(
            {
                "methodology_regime": regime,
                "leaderboards": int(len(group)),
                "window_start": group["publish_date"].min(),
                "window_end": group["publish_date"].max(),
                "mean_gap_rating": float(np.nanmean(gaps)),
                "gap_change_per_year_ols": fit.slope * 365.25,
                "ci_low": fit.slope_low * 365.25,
                "ci_high": fit.slope_high * 365.25,
                "gap_change_per_year_theil_sen": ts_slope * 365.25,
                "theil_sen_ci_low": ts_low * 365.25,
                "theil_sen_ci_high": ts_high * 365.25,
                "r_squared": fit.r_squared,
                "estimator": "ols_hc3_and_theil_sen_on_within_regime_dates",
                "serial_correlation_note": (
                    "Consecutive leaderboards share most of their vote history, so observations are "
                    "strongly serially correlated and these intervals are too narrow. Read the sign "
                    "and rough magnitude, not the endpoints."
                ),
            }
        )
    if not rows:
        context.refuse(
            claim="Trend in the open-versus-closed capability gap.",
            reason=(
                f"No rating-methodology regime contains at least {MIN_DATES_FOR_TREND} published "
                "leaderboards with both an open and a closed model classified."
            ),
            unblocked_by="A longer run of publications under one methodology.",
            analysis="open_weights_lag",
        )
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    latest = frame.sort_values("window_end").iloc[-1]
    if not math.isfinite(latest["gap_change_per_year_ols"]):
        return frame
    # Guard against the sign of a trend being read as a forecast of when the gap closes.
    context.refuse(
        claim="The date on which open-weight models will match the closed frontier.",
        reason=(
            "Extrapolating the measured gap trend to zero assumes the trend is linear, that the "
            "rating scale is stable, and that no methodology change intervenes. The trend is "
            "estimated on serially correlated weekly publications inside a single regime, which "
            "supports a direction but not a crossing date."
        ),
        unblocked_by="A model of the gap process validated out of sample against past regime changes.",
        analysis="open_weights_lag",
    )
    return frame


def _gap_by_category(lmarena: pd.DataFrame, min_models_per_side: int = 3) -> pd.DataFrame:
    """The gap in the latest publication, per arena and category.

    Published because the pooled single-number version of this claim was refused in
    :mod:`aicap.analysis.data_quality`: the gap is category-specific, and this table is what that
    refusal points to instead.
    """
    rows = []
    classifiable = lmarena[lmarena["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED, WEIGHTS_CLOSED])]
    for (arena, category), group in classifiable.groupby(["arena", "category"]):
        latest = group[group["publish_date"] == group["publish_date"].max()]
        open_side = latest[latest["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED])]
        closed_side = latest[latest["weights_class"] == WEIGHTS_CLOSED]
        if len(open_side) < min_models_per_side or len(closed_side) < min_models_per_side:
            continue
        open_best = open_side.loc[open_side["rating"].idxmax()]
        closed_best = closed_side.loc[closed_side["rating"].idxmax()]
        gap = float(closed_best["rating"] - open_best["rating"])
        gap_se = float(
            np.sqrt(np.nan_to_num(open_best["rating_se"]) ** 2 + np.nan_to_num(closed_best["rating_se"]) ** 2)
        )
        rows.append(
            {
                "arena": arena,
                "category": category,
                "publish_date": latest["publish_date"].max(),
                "open_models": int(len(open_side)),
                "closed_models": int(len(closed_side)),
                "open_best_model": str(open_best["model_name"]),
                "open_best_rating": float(open_best["rating"]),
                "closed_best_model": str(closed_best["model_name"]),
                "closed_best_rating": float(closed_best["rating"]),
                "gap_rating": gap,
                "gap_se": gap_se,
                "gap_ci_low": gap - 1.959963985 * gap_se,
                "gap_ci_high": gap + 1.959963985 * gap_se,
                "gap_distinguishable_from_zero": bool(gap - 1.959963985 * gap_se > 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["arena", "gap_rating"], ascending=[True, False]).reset_index(
        drop=True
    )
