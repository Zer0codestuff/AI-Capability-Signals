"""Small, deterministic calculations. Scenarios are not probability forecasts."""

import math
import statistics
from datetime import date

from pipeline.policy import METR_RELIABLE_MINUTES


def fit_log_trend(points: list[tuple[float, float]]) -> dict:
    """Fit log2(minutes) against elapsed days, with a robust slope alongside."""
    if len(points) < 6 or any(y <= 0 or not math.isfinite(y) for _, y in points):
        raise ValueError("A trend needs six positive, finite observations.")
    xs = [x for x, _ in points]
    ys = [math.log2(y) for _, y in points]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    ssx = sum((x - mean_x) ** 2 for x in xs)
    if ssx == 0:
        raise ValueError("A trend needs distinct dates.")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys, strict=True)]
    robust_slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[i] != xs[j]
    ]
    robust = statistics.median(robust_slopes)
    if slope <= 0 or robust <= 0:
        raise ValueError("The selected data does not support an increasing trend.")
    return {
        "slope_per_day": slope,
        "intercept": intercept,
        "doubling_days": 1 / slope,
        "robust_doubling_days": 1 / robust,
        "n": len(points),
        "rmse_log2": math.sqrt(statistics.mean(r * r for r in residuals)),
    }


def backtest(points: list[tuple[float, float]]) -> dict:
    """Leave each later date out. Equal-date models never leak into training."""
    errors, baseline_errors = [], []
    for target_date in sorted({x for x, _ in points}):
        prior = [(x, y) for x, y in points if x < target_date]
        if len(prior) < 6:
            continue
        try:
            fit = fit_log_trend(prior)
        except ValueError:
            continue
        last_date = max(x for x, _ in prior)
        baseline = max(y for x, y in prior if x == last_date)
        for x, y in points:
            if x == target_date:
                errors.append(abs(fit["intercept"] + fit["slope_per_day"] * x - math.log2(y)))
                baseline_errors.append(abs(math.log2(baseline) - math.log2(y)))
    if not errors:
        return {"n": 0, "mae_log2": None, "baseline_mae_log2": None, "skill_ratio": None}
    mae, baseline_mae = statistics.mean(errors), statistics.mean(baseline_errors)
    return {
        "n": len(errors),
        "mae_log2": mae,
        "baseline_mae_log2": baseline_mae,
        "skill_ratio": mae / baseline_mae if baseline_mae > 0 else None,
        "scope": "Retrospective next-release test on today's revised benchmark snapshot.",
    }


def scenario(minutes: float, days: float, doubling_days: float, pace: float = 1) -> float:
    # Browser counterpart: src/lib/math.ts project(), tested with the same known values.
    if minutes <= 0 or days < 0 or doubling_days <= 0 or not 0 <= pace <= 1.5:
        raise ValueError("Invalid scenario inputs.")
    return minutes * 2 ** (days * pace / doubling_days)


def build_trends(models: list[dict]) -> dict:
    result = {}
    for reliability in ("p50", "p80"):
        # Use one highest point per release date, excluding measurements above METR's
        # stated reliable range. This is a defined selection, not a universal AI law.
        by_date: dict[str, dict] = {}
        for model in models:
            value = model[reliability]["estimate"]
            if value > METR_RELIABLE_MINUTES:
                continue
            key = model["release_date"]
            if key not in by_date or value > by_date[key][reliability]["estimate"]:
                by_date[key] = model
        selected = sorted(by_date.values(), key=lambda m: m["release_date"])
        origin = date.fromisoformat(selected[0]["release_date"])
        points = [
            ((date.fromisoformat(m["release_date"]) - origin).days, m[reliability]["estimate"])
            for m in selected
        ]
        fit = fit_log_trend(points)
        anchor = max(selected, key=lambda m: (m[reliability]["estimate"], m["release_date"]))
        result[reliability] = {
            **fit,
            "origin": origin.isoformat(),
            "start": selected[0]["release_date"],
            "end": selected[-1]["release_date"],
            "anchor_id": anchor["id"],
            "anchor_date": anchor["release_date"],
            "anchor_minutes": anchor[reliability]["estimate"],
            "model_ids": [m["id"] for m in selected],
            "backtest": backtest(points),
            "method": "OLS of log2(minutes) on days; highest eligible point per release date.",
        }
    return result
