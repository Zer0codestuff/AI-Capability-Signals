"""Estimators and uncertainty quantification.

Every published interval in this project comes from one of the functions below, and every
function here is covered by a test that checks it recovers a known answer on synthetic data
(see ``tests/test_stats.py``). That pairing is the point: the previous version's headline
"uncertainty" came from procedures that had never been checked against a case with a known
answer, and would not have survived one.

Conventions
-----------
* ``*_ci`` functions return ``(point, low, high)`` triples at :data:`aicap.config.CONFIDENCE_LEVEL`.
* Bootstrap routines resample the *unit of analysis* (a model, an occupation, a benchmark), never
  a row of a table that may contain repeated snapshots of the same unit.
* Functions return NaN triples rather than raising when the input is too small to support an
  estimate. Callers are expected to record a refusal in that case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .config import BOOTSTRAP_DRAWS, CONFIDENCE_LEVEL, SEED

NAN3 = (float("nan"), float("nan"), float("nan"))


def _z(confidence: float) -> float:
    return float(sps.norm.ppf(0.5 + confidence / 2.0))


# --------------------------------------------------------------------------------------------
# Proportions
# --------------------------------------------------------------------------------------------


def wilson_interval(
    successes: int, total: int, confidence: float = CONFIDENCE_LEVEL
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the Wald interval because disclosure rates are frequently near 0 or 1 with
    small denominators, exactly where Wald intervals leave the unit interval and undercover.
    """
    if total <= 0:
        return NAN3
    z = _z(confidence)
    phat = successes / total
    denominator = 1.0 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    return phat, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = SEED,
    min_n: int = 8,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a statistic of one sample.

    ``min_n`` guards the small-sample regime where a percentile bootstrap interval is badly
    calibrated. Below it the function returns NaNs so the caller records a refusal instead of
    publishing an interval that does not cover.
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size < min_n:
        return NAN3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, array.size, size=(draws, array.size))
    replicates = np.asarray([statistic(array[row]) for row in idx], dtype=float)
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return NAN3
    alpha = (1.0 - confidence) / 2.0
    return (
        float(statistic(array)),
        float(np.quantile(replicates, alpha)),
        float(np.quantile(replicates, 1.0 - alpha)),
    )


def paired_bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = SEED,
    min_n: int = 8,
) -> tuple[float, float, float]:
    """Percentile bootstrap for a statistic of paired observations.

    Resamples *pairs*, which is the correct unit when the statistic is a correlation or a
    difference measured on the same units. Used for rank-agreement intervals, where the unit is
    a model observed on two benchmarks.
    """
    clean = [
        (float(a), float(b))
        for a, b in pairs
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]
    if len(clean) < min_n:
        return NAN3
    xs = np.asarray([p[0] for p in clean])
    ys = np.asarray([p[1] for p in clean])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xs.size, size=(draws, xs.size))
    replicates = []
    for row in idx:
        value = statistic(xs[row], ys[row])
        if np.isfinite(value):
            replicates.append(value)
    if not replicates:
        return NAN3
    alpha = (1.0 - confidence) / 2.0
    return (
        float(statistic(xs, ys)),
        float(np.quantile(replicates, alpha)),
        float(np.quantile(replicates, 1.0 - alpha)),
    )


# --------------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFit:
    """An OLS fit with heteroskedasticity-consistent (HC3) inference."""

    slope: float
    intercept: float
    slope_se: float
    slope_low: float
    slope_high: float
    r_squared: float
    n: int

    @property
    def valid(self) -> bool:
        return self.n >= 3 and math.isfinite(self.slope_se)


def ols_hc3(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> LinearFit:
    """Simple linear regression with HC3 robust standard errors.

    HC3 rather than classical standard errors because the residual variance of disclosed
    training compute grows over time (more labs, wider spread), and classical errors would
    overstate precision. HC3 is the recommended small-sample variant of the sandwich estimator.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = xa.size
    if n < 3 or np.allclose(xa, xa[0]):
        return LinearFit(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n)

    design = np.column_stack([np.ones(n), xa])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ ya
    residuals = ya - design @ beta

    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    # HC3 inflates each squared residual by 1/(1-h_i)^2, which downweights high-leverage points
    # instead of letting a single early observation set the slope.
    weights = (residuals / np.clip(1.0 - leverage, 1e-8, None)) ** 2
    meat = design.T @ (design * weights[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    slope_se = float(math.sqrt(max(covariance[1, 1], 0.0)))

    total_ss = float(((ya - ya.mean()) ** 2).sum())
    resid_ss = float((residuals**2).sum())
    r_squared = 1.0 - resid_ss / total_ss if total_ss > 0 else float("nan")

    # t rather than normal quantiles: these fits routinely have tens, not thousands, of points.
    crit = float(sps.t.ppf(0.5 + confidence / 2.0, df=max(n - 2, 1)))
    return LinearFit(
        slope=float(beta[1]),
        intercept=float(beta[0]),
        slope_se=slope_se,
        slope_low=float(beta[1] - crit * slope_se),
        slope_high=float(beta[1] + crit * slope_se),
        r_squared=r_squared,
        n=n,
    )


def theil_sen_slope(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """Theil-Sen median-of-pairwise-slopes estimator with its distribution-free interval.

    Reported alongside OLS as a robustness check. When the two disagree materially, the OLS
    slope is being driven by outliers and the analysis says so rather than picking the
    friendlier number.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3 or np.allclose(xa, xa[0]):
        return NAN3
    result = sps.theilslopes(ya, xa, alpha=confidence)
    return float(result[0]), float(result[2]), float(result[3])


# --------------------------------------------------------------------------------------------
# Rank agreement
# --------------------------------------------------------------------------------------------


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b, which is the tie-corrected variant. Benchmark scores tie often."""
    if x.size < 3:
        return float("nan")
    tau = sps.kendalltau(x, y, variant="b", nan_policy="omit").statistic
    return float(tau)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    return float(sps.spearmanr(x, y, nan_policy="omit").statistic)


def top_k_overlap(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Share of the top ``k`` by ``x`` that is also in the top ``k`` by ``y``.

    A rank correlation can look healthy while the *leaderboard head* disagrees completely, and
    the head is what readers use. This reports the head agreement directly.
    """
    k = int(min(k, x.size, y.size))
    if k <= 0:
        return float("nan")
    top_x = set(np.argsort(-x, kind="stable")[:k].tolist())
    top_y = set(np.argsort(-y, kind="stable")[:k].tolist())
    return len(top_x & top_y) / k


# --------------------------------------------------------------------------------------------
# Combining repeated estimates
# --------------------------------------------------------------------------------------------


def inverse_variance_combine(
    estimates: Sequence[float], variances: Sequence[float]
) -> tuple[float, float]:
    """Combine repeated estimates of one quantity, returning ``(estimate, standard_error)``.

    Uses inverse-variance weighting for the point estimate and adds the between-replicate spread
    to the standard error, in the spirit of a random-effects meta-analysis. The second term
    matters here: upstream leaderboard duplicates disagree by more than their individual
    variances imply, and ignoring that would understate uncertainty.
    """
    est = np.asarray(estimates, dtype=float)
    var = np.asarray(variances, dtype=float)
    mask = np.isfinite(est)
    est = est[mask]
    var = var[mask] if var.size == mask.size else np.full(est.size, np.nan)
    if est.size == 0:
        return float("nan"), float("nan")
    if est.size == 1:
        single_var = float(var[0]) if var.size and np.isfinite(var[0]) else float("nan")
        return float(est[0]), math.sqrt(single_var) if np.isfinite(single_var) else float("nan")

    usable = np.isfinite(var) & (var > 0)
    if usable.all():
        weights = 1.0 / var
        point = float((weights * est).sum() / weights.sum())
        within_se_sq = float(1.0 / weights.sum())
    else:
        point = float(np.median(est))
        within_se_sq = float(np.nanmean(var)) / est.size if np.isfinite(var).any() else 0.0

    between_var = float(np.var(est, ddof=1))
    return point, math.sqrt(within_se_sq + between_var / est.size)


# --------------------------------------------------------------------------------------------
# Time-to-event with right censoring
# --------------------------------------------------------------------------------------------


def kaplan_meier_median(durations: Sequence[float], observed: Sequence[bool]) -> tuple[float, float]:
    """Kaplan-Meier median duration and the largest duration reached, given right censoring.

    Returns ``(median, max_duration)``. The median is NaN when the survival curve never falls to
    0.5, which happens when more than half the observations are still censored — in that case the
    honest answer is "longer than the follow-up", not a number.

    Why this rather than the mean of the completed cases: the observations that have *not* completed
    are systematically the hard ones, so averaging only completed cases understates the duration.
    That is the same survivorship error as reading a price history off a catalogue of surviving
    models, and it is worth avoiding in both places.
    """
    times = np.asarray(durations, dtype=float)
    events = np.asarray(observed, dtype=bool)
    mask = np.isfinite(times)
    times, events = times[mask], events[mask]
    if times.size == 0:
        return float("nan"), float("nan")

    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    at_risk = times.size
    survival = 1.0
    median = float("nan")
    for time in np.unique(times):
        at_this_time = times == time
        deaths = int(events[at_this_time].sum())
        censored = int((~events[at_this_time]).sum())
        if deaths and at_risk > 0:
            survival *= 1.0 - deaths / at_risk
            if survival <= 0.5 and not math.isfinite(median):
                median = float(time)
        at_risk -= deaths + censored
    return median, float(times.max())


# --------------------------------------------------------------------------------------------
# Multiple comparisons and permutation
# --------------------------------------------------------------------------------------------


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Returns ``(rejected, q_values)`` aligned with the input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward so q-values are non-decreasing in p.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q_out = np.empty(n, dtype=float)
    q_out[order] = q
    return q_out <= alpha, q_out


def permutation_p_value(
    observed: float,
    resample: Callable[[np.random.Generator], float],
    draws: int = 10_000,
    seed: int = SEED,
) -> float:
    """One-sided permutation p-value with the standard ``(b + 1) / (m + 1)`` correction.

    The correction keeps the p-value strictly positive, so it can never be reported as exactly
    zero from a finite number of draws.
    """
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        if resample(rng) >= observed:
            exceed += 1
    return (exceed + 1) / (draws + 1)


# --------------------------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Rolling-origin evaluation of a forecasting rule against a naive baseline."""

    horizon: int
    folds: int
    mae: float
    rmse: float
    baseline_mae: float
    #: MAE of the rule divided by MAE of the naive baseline. Below 1 means the rule helps.
    skill_ratio: float
    #: Empirical quantiles of signed error, used to build honest prediction intervals.
    error_low: float
    error_high: float

    @property
    def beats_baseline(self) -> bool:
        return math.isfinite(self.skill_ratio) and self.skill_ratio < 1.0


def rolling_origin_backtest(
    x: Sequence[float],
    y: Sequence[float],
    horizon: int,
    min_train: int = 6,
    confidence: float = CONFIDENCE_LEVEL,
) -> BacktestResult:
    """Backtest a linear-trend forecast at a fixed horizon, against a last-value baseline.

    For each origin ``t`` with at least ``min_train`` prior points, fit the trend on
    ``x[:t] -> y[:t]`` and predict ``y`` at ``x[t + horizon - 1]``. The baseline is the last
    observed value, which is the honest thing to beat: if a fitted trend cannot beat "assume no
    change", the trend carries no forecasting information and no forecast should be published.

    The returned error quantiles are what the forecast module uses for prediction intervals, so
    the published interval width is measured out-of-sample rather than assumed.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    order = np.argsort(xa, kind="stable")
    xa, ya = xa[order], ya[order]

    errors: list[float] = []
    baseline_errors: list[float] = []
    for cut in range(min_train, xa.size - horizon + 1):
        fit = ols_hc3(xa[:cut], ya[:cut])
        if not math.isfinite(fit.slope):
            continue
        target_index = cut + horizon - 1
        prediction = fit.intercept + fit.slope * xa[target_index]
        actual = ya[target_index]
        errors.append(float(prediction - actual))
        baseline_errors.append(float(ya[cut - 1] - actual))

    if not errors:
        nan = float("nan")
        return BacktestResult(horizon, 0, nan, nan, nan, nan, nan, nan)

    err = np.asarray(errors)
    base = np.asarray(baseline_errors)
    mae = float(np.mean(np.abs(err)))
    baseline_mae = float(np.mean(np.abs(base)))
    alpha = (1.0 - confidence) / 2.0
    return BacktestResult(
        horizon=horizon,
        folds=err.size,
        mae=mae,
        rmse=float(math.sqrt(float(np.mean(err**2)))),
        baseline_mae=baseline_mae,
        skill_ratio=mae / baseline_mae if baseline_mae > 0 else float("nan"),
        error_low=float(np.quantile(err, alpha)),
        error_high=float(np.quantile(err, 1.0 - alpha)),
    )
