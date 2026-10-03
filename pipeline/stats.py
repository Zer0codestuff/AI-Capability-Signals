"""Small, dependency-free statistics shared by every chapter.

One procedure is applied to every metric so the charts can be compared:

1. Fit a straight line by ordinary least squares. For quantities that grow by
   multiplication (compute, cost, task length, price) the line is fitted to log10 of
   the value, so a straight line means steady exponential change.
2. Ask whether the pace changed: find the split date that lets two separate lines
   fit best and compare the two slopes. The split is chosen after looking at the data,
   so the bootstrap repeats the search for the split on every resample. A change counts
   only if 95% of the resamples agree on its direction.
3. Project from the recent part when the pace clearly changed, otherwise from the whole
   window. The 80% band comes from resampling the fitted points and their residuals.
4. Backtest the whole procedure on the past: cut the data at earlier dates, project a
   year ahead and compare with what was actually released.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta

ORIGIN = 2020.0
LOG2 = math.log10(2)
MIN_PART_POINTS = 5
# A two-line fit is considered only if each part is long enough to be a pace rather than
# a burst: the recent part is what a projection would be based on, so it needs more.
MIN_PART_SHARE = 0.25
MIN_EARLY_YEARS = 1.0
MIN_RECENT_YEARS = 2.0


def year_fraction(day: date) -> float:
    start = date(day.year, 1, 1)
    return day.year + (day - start).days / (date(day.year + 1, 1, 1) - start).days


def from_year_fraction(value: float) -> date:
    year = math.floor(value)
    start = date(year, 1, 1)
    days = (date(year + 1, 1, 1) - start).days
    return start + timedelta(days=round((value - year) * days))


def quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        raise ValueError("quantile of empty list")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class Fit:
    slope: float
    intercept: float
    r2: float
    residuals: tuple[float, ...]

    def at(self, x: float) -> float:
        return self.intercept + self.slope * x


def ols(xs: list[float], ys: list[float]) -> Fit | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = tuple(y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True))
    total = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in residuals) / total if total > 0 else 1.0
    return Fit(slope, intercept, r2, residuals)


def split_bounds(xs: list[float]) -> tuple[float, float]:
    """Earliest end of the first part and latest start of the recent part."""
    span = xs[-1] - xs[0]
    return (
        xs[0] + max(MIN_EARLY_YEARS, MIN_PART_SHARE * span),
        xs[-1] - max(MIN_RECENT_YEARS, MIN_PART_SHARE * span),
    )


def best_split(
    xs: list[float], ys: list[float], bounds: tuple[float, float]
) -> tuple[int, float, float] | None:
    """Best place to break sorted points into two separately fitted lines.

    Returns the index that starts the second part and the two slopes. Prefix sums make
    every candidate O(1), which keeps the bootstrap of the search cheap.
    """
    n = len(xs)
    if n < 2 * MIN_PART_POINTS:
        return None
    sums = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for x, y in zip(xs, ys, strict=True):
        sx, sy, sxx, sxy, syy = sums[-1]
        sums.append((sx + x, sy + y, sxx + x * x, sxy + x * y, syy + y * y))

    def part(start: int, end: int) -> tuple[float, float] | None:
        count = end - start
        sx, sy, sxx, sxy, syy = (b - a for a, b in zip(sums[start], sums[end], strict=True))
        var_x = sxx - sx * sx / count
        if var_x <= 1e-9:
            return None
        cov = sxy - sx * sy / count
        return max(0.0, syy - sy * sy / count - cov * cov / var_x), cov / var_x

    best: tuple[float, int, float, float] | None = None
    for index in range(MIN_PART_POINTS, n - MIN_PART_POINTS + 1):
        if xs[index - 1] < bounds[0] or xs[index] > bounds[1]:
            continue
        first, second = part(0, index), part(index, n)
        if first is None or second is None:
            continue
        error = first[0] + second[0]
        if best is None or error < best[0]:
            best = (error, index, first[1], second[1])
    return best[1:] if best else None


class Line:
    """An OLS line with bootstrap replicates, on x = years since ORIGIN."""

    def __init__(self, xs: list[float], ys: list[float], *, seed: int, reps: int):
        fit = ols(xs, ys)
        if fit is None:
            raise ValueError("not enough distinct points to fit a line")
        self.fit = fit
        self.n = len(xs)
        rng = random.Random(seed)
        self.replicates: list[tuple[float, float, float]] = []
        for _ in range(reps):
            picks = [rng.randrange(self.n) for _ in range(self.n)]
            replicate = ols([xs[i] for i in picks], [ys[i] for i in picks])
            if replicate is not None:
                self.replicates.append((replicate.slope, replicate.intercept, rng.choice(fit.residuals)))
        self.slopes = sorted(slope for slope, _, _ in self.replicates)

    def slope_interval(self, low: float = 0.05, high: float = 0.95) -> tuple[float, float]:
        return quantile(self.slopes, low), quantile(self.slopes, high)

    def predict(self, x: float, low: float = 0.1, high: float = 0.9) -> tuple[float, float, float]:
        """Central estimate and prediction interval for one new observation at x."""
        draws = sorted(a + b * x + noise for b, a, noise in self.replicates)
        return self.fit.at(x), quantile(draws, low), quantile(draws, high)

    def crossing(self, level: float) -> tuple[float, float, float] | None:
        """Year (as a fraction) at which the line reaches `level`, with an 80% interval."""
        if self.fit.slope == 0:
            return None
        direction = 1 if self.fit.slope > 0 else -1
        years = sorted((level - a) / b for b, a, _ in self.replicates if b * direction > 1e-9)
        if len(years) < len(self.replicates) * 0.9:
            return None
        mid = (level - self.fit.intercept) / self.fit.slope
        return mid, quantile(years, 0.1), quantile(years, 0.9)


@dataclass
class Shape:
    verdict: str  # "speeding_up", "slowing_down", "steady" or "unknown"
    split: float | None = None
    early: Line | None = None
    late: Line | None = None
    difference: tuple[float, float, float] | None = None


class Trend:
    """The full procedure for one series of (date, value) points."""

    def __init__(
        self,
        points: list[tuple[date, float]],
        *,
        log: bool,
        seed: int = 7,
        reps: int = 2000,
    ):
        if len(points) < 3:
            raise ValueError("a trend needs at least three points")
        self.log = log
        self.points = sorted(points)
        self.xs = [year_fraction(day) - ORIGIN for day, _ in self.points]
        self.ys = [math.log10(value) if log else value for _, value in self.points]
        self.full = Line(self.xs, self.ys, seed=seed, reps=reps)
        self.shape = self._shape(seed, reps)
        changed = self.shape.verdict in ("speeding_up", "slowing_down")
        self.basis = "recent" if changed and self.shape.late else "full"
        self.line = self.shape.late if self.basis == "recent" else self.full

    def _shape(self, seed: int, reps: int) -> Shape:
        bounds = split_bounds(self.xs)
        found = best_split(self.xs, self.ys, bounds)
        if found is None:
            return Shape("unknown")
        index = found[0]
        try:
            first = Line(self.xs[:index], self.ys[:index], seed=seed + 1, reps=reps)
            second = Line(self.xs[index:], self.ys[index:], seed=seed + 2, reps=reps)
        except ValueError:
            return Shape("unknown")
        # Positive pace means "more of what the whole window is doing".
        direction = 1 if self.full.fit.slope >= 0 else -1
        rng = random.Random(seed + 3)
        count = len(self.xs)
        differences = []
        for _ in range(reps):
            picks = sorted(rng.randrange(count) for _ in range(count))
            again = best_split([self.xs[i] for i in picks], [self.ys[i] for i in picks], bounds)
            # A resample with no admissible split shows no detectable change.
            differences.append(direction * (again[2] - again[1]) if again else 0.0)
        differences.sort()
        low, high = quantile(differences, 0.025), quantile(differences, 0.975)
        centre = direction * (second.fit.slope - first.fit.slope)
        verdict = "speeding_up" if low > 0 else "slowing_down" if high < 0 else "steady"
        split = (self.xs[index - 1] + self.xs[index]) / 2
        return Shape(verdict, split, first, second, (centre, low, high))

    def predict(self, day: date) -> tuple[float, float, float]:
        """Prediction in the original units."""
        mid, low, high = self.line.predict(year_fraction(day) - ORIGIN)
        return (10**mid, 10**low, 10**high) if self.log else (mid, low, high)

    def crossing(self, level: float) -> tuple[date, date, date] | None:
        result = self.line.crossing(math.log10(level) if self.log else level)
        if result is None:
            return None
        if any(not 1990 < value + ORIGIN < 2200 for value in result):
            return None
        return tuple(from_year_fraction(value + ORIGIN) for value in result)  # type: ignore[return-value]


def rate(slope: float, log: bool) -> float:
    """Yearly change: a multiplication factor for log series, units per year otherwise."""
    return 10**slope if log else slope


def doubling_months(slope: float) -> float | None:
    """Months to double (positive) or halve (negative) for a log10 slope per year."""
    return 12 * LOG2 / slope if abs(slope) > 1e-9 else None


def running_records(
    points: list[tuple[date, float, object]], *, lowest: bool = False
) -> list[tuple[date, float, object]]:
    """Points that set a new record at the time they appeared."""
    best: float | None = None
    records = []
    for point in sorted(points, key=lambda p: (p[0], -p[1] if not lowest else p[1])):
        value = point[1]
        if best is None or (value < best if lowest else value > best):
            best = value
            records.append(point)
    return records


def top_at_release(
    points: list[tuple[date, float, object]], *, k: int = 10
) -> list[tuple[date, float, object]]:
    """Points that ranked in the top k of everything released up to their own date."""
    seen: list[float] = []
    selected = []
    for point in sorted(points, key=lambda p: p[0]):
        value = point[1]
        if len(seen) < k or value > sorted(seen)[-k]:
            selected.append(point)
        seen.append(value)
    return selected


def backtest(
    points: list[tuple[date, float]],
    *,
    log: bool,
    seed: int = 11,
    reps: int = 300,
    min_train: int = 6,
    step: float = 0.5,
    lead: tuple[float, float] = (0.5, 1.5),
) -> dict | None:
    """Rolling-origin check of the whole procedure about one year ahead.

    For each past cutoff, fit on points released up to the cutoff and compare the
    projection with the points released 6 to 18 months later. The naive rival assumes
    no further change: it repeats the median of the last three fitted points.
    """
    ordered = sorted(points)
    if len(ordered) < min_train + 2:
        return None
    xs = [year_fraction(day) for day, _ in ordered]
    errors: list[float] = []
    naive_errors: list[float] = []
    inside = 0
    cutoffs = 0
    cutoff = xs[min_train - 1]
    while cutoff + lead[0] < xs[-1]:
        train = [p for p, x in zip(ordered, xs, strict=True) if x <= cutoff]
        test = [p for p, x in zip(ordered, xs, strict=True) if cutoff + lead[0] < x <= cutoff + lead[1]]
        if len(train) >= min_train and test:
            try:
                trend = Trend(train, log=log, seed=seed + cutoffs, reps=reps)
            except ValueError:
                cutoff += step
                continue
            recent = sorted(value for _, value in train[-3:])
            flat = recent[len(recent) // 2]
            cutoffs += 1
            for day, actual in test:
                mid, low, high = trend.predict(day)
                if log:
                    errors.append(math.log10(mid / actual))
                    naive_errors.append(math.log10(flat / actual))
                else:
                    errors.append(mid - actual)
                    naive_errors.append(flat - actual)
                inside += low <= actual <= high
        cutoff += step
    if len(errors) < 5:
        return None
    absolute = sorted(abs(e) for e in errors)
    naive_absolute = sorted(abs(e) for e in naive_errors)
    return {
        "cutoffs": cutoffs,
        "tests": len(errors),
        "typical_error": quantile(absolute, 0.5),
        "bias": quantile(sorted(errors), 0.5),
        "naive_error": quantile(naive_absolute, 0.5),
        "coverage": inside / len(errors),
        "log": log,
    }
