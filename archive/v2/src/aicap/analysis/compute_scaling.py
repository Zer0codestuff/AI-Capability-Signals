"""How fast does disclosed training compute grow, and can that trend forecast anything?

Question
--------
Among models whose training compute is disclosed, how fast does the compute frontier grow, and does
a fitted trend forecast better than assuming no change?

Why this is the only forecast the project publishes
---------------------------------------------------
Training compute is measured in FLOP: a physical, unbounded, ratio-scale quantity. Its logarithm
can be extrapolated without hitting a ceiling, unlike a benchmark percentage capped at 100 — which
is what made the previous version's 2036 domain forecasts incoherent. It also has enough dated
observations for rolling-origin validation.

A forecast is published only if it beats a last-value baseline out of sample. The interval comes
from the measured backtest errors, not from the regression's in-sample standard error, because the
in-sample interval assumes the model is correct and answers the wrong question.

The selection problem, taken seriously
--------------------------------------
Compute is disclosed for a minority of models, and disclosure is not random: open research releases
document compute, frontier commercial systems often do not. The frontier estimated from disclosed
values is therefore a lower bound on the true frontier, and the growth rate is only unbiased if
disclosure propensity is unrelated to compute *within* year. That is not testable here, so
:func:`_selection_sensitivity` reports how much the slope moves under several restrictions. A
result that survives all of them is not proven, but a result that does not survive them is
withdrawn.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..provenance import RunContext
from ..sources.epoch import censoring_boundary
from ..stats import BacktestResult, ols_hc3, rolling_origin_backtest, theil_sen_slope

#: Compute disclosure is too sparse before this year to estimate an annual frontier.
FIRST_YEAR = 2012

#: Minimum disclosed models in a year for that year to contribute a frontier point.
MIN_MODELS_PER_YEAR = 5

#: Frontier definition: the quantile of log10 compute taken within each year. The 90th percentile
#: rather than the maximum, because a maximum is a single observation and inherits the full noise of
#: whichever lab happened to disclose. Both are reported so the choice is visible.
FRONTIER_QUANTILE = 0.9

#: Horizons backtested, in years.
BACKTEST_HORIZONS = (1, 2, 3)


def build(epoch: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    censored_from = censoring_boundary(epoch)
    disclosed = epoch[
        epoch["training_compute_flop"].notna()
        & (epoch["training_compute_flop"] > 0)
        & epoch["publication_year"].notna()
    ].copy()
    disclosed["publication_year"] = disclosed["publication_year"].astype(int)
    disclosed["log10_compute"] = np.log10(disclosed["training_compute_flop"])
    window = disclosed[
        (disclosed["publication_year"] >= FIRST_YEAR) & (disclosed["publication_year"] < censored_from)
    ]

    if window["publication_year"].nunique() < 6:
        context.refuse(
            claim="Growth rate of disclosed training compute.",
            reason="Fewer than six usable years of disclosed compute after the censoring cut.",
            unblocked_by="A dataset covering more complete years.",
            analysis="compute_scaling",
        )
        return {}

    frontier = _frontier_by_year(window)
    trends = _trend_estimates(window, frontier)
    sensitivity = _selection_sensitivity(window)
    backtests, backtest_frame = _backtests(frontier)
    forecast = _forecast(frontier, backtests, context)

    tables = {
        "compute_frontier_by_year": frontier,
        "compute_trend_estimates": trends,
        "compute_selection_sensitivity": sensitivity,
        "compute_forecast_backtest": backtest_frame,
    }
    if not forecast.empty:
        tables["compute_frontier_forecast"] = forecast
    return tables


def _frontier_by_year(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in window.groupby("publication_year"):
        if len(group) < MIN_MODELS_PER_YEAR:
            continue
        logs = group["log10_compute"]
        top = group.loc[logs.idxmax()]
        rows.append(
            {
                "publication_year": int(year),
                "models_with_disclosed_compute": int(len(group)),
                "log10_compute_p90": float(logs.quantile(FRONTIER_QUANTILE)),
                "log10_compute_max": float(logs.max()),
                "log10_compute_median": float(logs.median()),
                "top_model": str(top["model"]),
                "top_model_vendor": str(top["vendor"]),
                "distinct_vendors": int(group["vendor"].nunique()),
                "frontier_definition": f"p{int(FRONTIER_QUANTILE * 100)} of log10 FLOP within year",
            }
        )
    return pd.DataFrame(rows).sort_values("publication_year").reset_index(drop=True)


def _trend_estimates(window: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    """Estimate growth for several subsets and with two estimators each.

    Reporting OLS and Theil-Sen side by side is the robustness check: they answer the same question
    with different sensitivity to outliers, and a material disagreement means the OLS slope is being
    set by a few points. The report quotes the pair, not the friendlier one.
    """
    specifications = [
        (
            "frontier_p90_by_year",
            frontier["publication_year"].to_numpy(dtype=float),
            frontier["log10_compute_p90"].to_numpy(dtype=float),
            "Annual 90th percentile of log10 disclosed training compute.",
        ),
        (
            "frontier_max_by_year",
            frontier["publication_year"].to_numpy(dtype=float),
            frontier["log10_compute_max"].to_numpy(dtype=float),
            "Annual maximum of log10 disclosed training compute.",
        ),
        (
            "all_disclosed_models",
            window["publication_year"].to_numpy(dtype=float),
            window["log10_compute"].to_numpy(dtype=float),
            "Every model with disclosed compute, one observation per model.",
        ),
    ]

    frontier_flagged = window[window["frontier_flag"]]
    if frontier_flagged["publication_year"].nunique() >= 4:
        specifications.append(
            (
                "epoch_frontier_flagged_models",
                frontier_flagged["publication_year"].to_numpy(dtype=float),
                frontier_flagged["log10_compute"].to_numpy(dtype=float),
                "Models the source curator flags as frontier systems.",
            )
        )

    rows = []
    for name, x, y, description in specifications:
        fit = ols_hc3(x, y)
        rows.append(
            {
                "specification": name,
                "estimator": "ols_hc3",
                "n": fit.n,
                "log10_flop_per_year": fit.slope,
                "ci_low": fit.slope_low,
                "ci_high": fit.slope_high,
                "doubling_time_months": _doubling_months(fit.slope),
                "doubling_time_months_low": _doubling_months(fit.slope_high),
                "doubling_time_months_high": _doubling_months(fit.slope_low),
                "r_squared": fit.r_squared,
                "description": description,
            }
        )
        slope, low, high = theil_sen_slope(x, y)
        rows.append(
            {
                "specification": name,
                "estimator": "theil_sen",
                "n": int(np.isfinite(x).sum()),
                "log10_flop_per_year": slope,
                "ci_low": low,
                "ci_high": high,
                "doubling_time_months": _doubling_months(slope),
                "doubling_time_months_low": _doubling_months(high),
                "doubling_time_months_high": _doubling_months(low),
                "r_squared": float("nan"),
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def _current_year() -> int:
    return datetime.now(UTC).year


def _doubling_months(log10_per_year: float) -> float:
    """Convert a log10-per-year slope into a doubling time in months.

    A slope of ``s`` in log10 units per year multiplies compute by ``10**s`` per year, so the
    doubling time is ``log10(2) / s`` years. Non-positive slopes have no doubling time and return
    NaN rather than a negative number that would read as a plausible duration.
    """
    if not math.isfinite(log10_per_year) or log10_per_year <= 0:
        return float("nan")
    return float(math.log10(2.0) / log10_per_year * 12.0)


def _selection_sensitivity(window: pd.DataFrame) -> pd.DataFrame:
    """How much does the estimated growth rate move under plausible selection restrictions?

    Each restriction targets a different way disclosure could bias the slope. None of them proves
    the estimate unbiased; together they bound how fragile it is.
    """
    baseline = ols_hc3(
        window["publication_year"].to_numpy(dtype=float), window["log10_compute"].to_numpy(dtype=float)
    )

    restrictions: list[tuple[str, pd.DataFrame, str]] = [
        ("all_disclosed", window, "No restriction."),
        (
            "top_decile_within_year",
            window[
                window["log10_compute"]
                >= window.groupby("publication_year")["log10_compute"].transform(lambda s: s.quantile(0.9))
            ],
            "Only the largest 10% of disclosed models each year: tests whether the trend is driven "
            "by the growing tail of small models rather than by the frontier.",
        ),
        (
            "vendors_with_5plus_disclosures",
            window[window.groupby("vendor")["vendor"].transform("size") >= 5],
            "Only organisations that disclose repeatedly: removes one-off disclosures whose "
            "propensity may correlate with model size.",
        ),
        (
            "language_domain_only",
            window[window["domain"].astype(str).str.contains("Language", na=False)],
            "Language models only: removes composition change as domains enter the dataset.",
        ),
        (
            "excluding_open_weights",
            window[window["weights_class"] == "closed_weights"],
            "Closed-weight models only: the subset whose disclosure is most likely selective.",
        ),
        (
            "open_weights_only",
            window[window["weights_class"].isin(["open_weights", "open_weights_restricted"])],
            "Open-weight models only: the subset with the most complete disclosure.",
        ),
    ]

    rows = []
    for name, subset, rationale in restrictions:
        if subset["publication_year"].nunique() < 4:
            rows.append(
                {
                    "restriction": name,
                    "n": int(len(subset)),
                    "log10_flop_per_year": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "doubling_time_months": float("nan"),
                    "relative_change_vs_baseline": float("nan"),
                    "status": "insufficient_years",
                    "rationale": rationale,
                }
            )
            continue
        fit = ols_hc3(
            subset["publication_year"].to_numpy(dtype=float), subset["log10_compute"].to_numpy(dtype=float)
        )
        relative = (
            (fit.slope - baseline.slope) / baseline.slope
            if math.isfinite(baseline.slope) and baseline.slope != 0
            else float("nan")
        )
        rows.append(
            {
                "restriction": name,
                "n": fit.n,
                "log10_flop_per_year": fit.slope,
                "ci_low": fit.slope_low,
                "ci_high": fit.slope_high,
                "doubling_time_months": _doubling_months(fit.slope),
                "relative_change_vs_baseline": relative,
                "status": "estimated",
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def _backtests(frontier: pd.DataFrame) -> tuple[dict[int, BacktestResult], pd.DataFrame]:
    years = frontier["publication_year"].to_numpy(dtype=float)
    values = frontier["log10_compute_p90"].to_numpy(dtype=float)

    results: dict[int, BacktestResult] = {}
    rows = []
    for horizon in BACKTEST_HORIZONS:
        result = rolling_origin_backtest(years, values, horizon=horizon)
        results[horizon] = result
        rows.append(
            {
                "horizon_years": horizon,
                "folds": result.folds,
                "mae_log10": result.mae,
                "rmse_log10": result.rmse,
                "baseline_mae_log10": result.baseline_mae,
                "skill_ratio_vs_last_value": result.skill_ratio,
                "beats_baseline": result.beats_baseline,
                "error_quantile_low": result.error_low,
                "error_quantile_high": result.error_high,
                "baseline": "last observed frontier value carried forward",
                "note": (
                    "Errors are in log10 FLOP. A skill ratio below 1 means the fitted trend "
                    "forecasts better out of sample than assuming no change."
                ),
            }
        )
    return results, pd.DataFrame(rows)


def _forecast(
    frontier: pd.DataFrame, backtests: dict[int, BacktestResult], context: RunContext
) -> pd.DataFrame:
    """Emit a forecast only for horizons whose backtest beat the naive baseline.

    Prediction intervals are the fitted point shifted by the empirical backtest error quantiles,
    which makes the published width an out-of-sample measurement. Note the sign: a *positive*
    forecast error means the rule over-predicted, so the interval subtracts the upper error
    quantile from the point to obtain the lower bound.
    """
    usable = [horizon for horizon, result in backtests.items() if result.beats_baseline]
    if not usable:
        context.refuse(
            claim="Forecast of the disclosed training-compute frontier.",
            reason=(
                "At every backtested horizon a fitted linear trend forecast no better than carrying "
                "the last observed value forward, so the trend adds no out-of-sample information."
            ),
            unblocked_by="A longer or less noisy frontier series.",
            analysis="compute_scaling",
        )
        return pd.DataFrame()

    years = frontier["publication_year"].to_numpy(dtype=float)
    values = frontier["log10_compute_p90"].to_numpy(dtype=float)
    fit = ols_hc3(years, values)
    last_year = int(years.max())

    rows = []
    for horizon in sorted(usable):
        result = backtests[horizon]
        target_year = last_year + horizon
        point = fit.intercept + fit.slope * target_year
        rows.append(
            {
                "target_year": target_year,
                "horizon_years": horizon,
                # The frontier series stops at the censoring boundary, so the nearest horizons land
                # on years that have already happened but are not yet completely curated. Those are
                # backfill expectations for years the dataset cannot yet measure, which is a
                # different epistemic object from a forecast of the future, and is labelled as such
                # rather than being quietly presented as prediction.
                "target_kind": "backfill_of_provisional_year" if target_year <= _current_year() else "future",
                "log10_compute_p90_forecast": point,
                "pi_low": point - result.error_high,
                "pi_high": point - result.error_low,
                "flop_forecast": 10.0**point,
                "interval_source": "empirical rolling-origin backtest error quantiles",
                "backtest_folds": result.folds,
                "skill_ratio_vs_last_value": result.skill_ratio,
                "interpretation": (
                    "Expectation for the 90th percentile of *disclosed* training compute. Because "
                    "disclosure is incomplete and voluntary, read it as a lower bound on the true "
                    "frontier."
                ),
            }
        )

    refused = [horizon for horizon in BACKTEST_HORIZONS if horizon not in usable]
    if refused:
        context.refuse(
            claim=f"Compute-frontier forecast at horizon(s) {refused} years.",
            reason="The fitted trend did not beat a last-value baseline out of sample at these horizons.",
            unblocked_by="A frontier series long enough for the trend to demonstrate skill at these horizons.",
            analysis="compute_scaling",
        )
    return pd.DataFrame(rows)
