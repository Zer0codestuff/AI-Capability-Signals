"""Do model releases cluster on calendar features, and does that survive multiple testing?

Question
--------
Across notable model releases, do weekdays, months or quarters appear more often than a year-
preserving random-date null would produce, and how many of those clusters survive a Benjamini-
Hochberg correction across the family of tests?

Why this exists
---------------
The previous version computed moon phases, Mercury-retrograde windows and geocentric planetary
zodiac signs for model releases, ran per-feature permutation tests with *no* multiple-comparison
correction, and rendered a zodiac heatmap. The stated intent was a demonstration of spurious
pattern-finding. The execution undermined it: testing many calendar features without correction is
exactly the error the demonstration was supposedly about.

This module keeps the lesson and drops the ephemeris. It tests a small, pre-registered family of
calendar features (weekday, month, quarter), applies Benjamini-Hochberg across that family, and
reports how many nominally significant results survive. The answer, if the null is roughly right,
should be close to zero — and that is the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED
from ..provenance import RunContext
from ..sources.epoch import censoring_boundary
from ..stats import benjamini_hochberg, permutation_p_value

FIRST_YEAR = 2018
PERMUTATION_DRAWS = 5_000

#: Pre-registered feature family. Nothing else is tested, so the size of the family — and therefore
#: the Benjamini-Hochberg correction — is known before looking at the data.
FEATURES: tuple[tuple[str, int], ...] = (
    ("weekday", 7),
    ("month", 12),
    ("quarter", 4),
)


def build(epoch: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    censored_from = censoring_boundary(epoch)
    dated = epoch.dropna(subset=["publication_date"]).copy()
    dated = dated[dated["date_precision"] == "day"]
    dated["year"] = dated["publication_date"].dt.year
    dated = dated[(dated["year"] >= FIRST_YEAR) & (dated["year"] < censored_from)]

    if len(dated) < 50:
        context.refuse(
            claim="Calendar clustering of model release dates.",
            reason="Fewer than 50 day-precise publication dates survive the censoring cut.",
            unblocked_by="More day-precise publication dates in the model dataset.",
            analysis="calendar_control",
        )
        return {}

    dated["weekday"] = dated["publication_date"].dt.day_name()
    dated["month"] = dated["publication_date"].dt.month_name()
    dated["quarter"] = "Q" + dated["publication_date"].dt.quarter.astype(str)

    year_counts = dated["year"].value_counts().to_dict()
    tests = _feature_tests(dated, year_counts)
    return {
        "calendar_release_tests": tests,
        "calendar_release_counts": _feature_counts(dated),
    }


def _feature_counts(dated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, _categories in FEATURES:
        counts = dated[feature].value_counts()
        for label, count in counts.items():
            rows.append(
                {
                    "feature": feature,
                    "bucket": label,
                    "releases": int(count),
                    "share": float(count / len(dated)),
                }
            )
    return pd.DataFrame(rows)


def _feature_tests(dated: pd.DataFrame, year_counts: dict[int, int]) -> pd.DataFrame:
    """Test each feature against a year-preserving random-date null.

    The null keeps the year distribution of releases fixed and redraws the day of year uniformly.
    That absorbs the fact that more models are released in recent years, so a slow-changing feature
    (quarter, month) is not credited with a cluster that is really just "recent years dominate".
    """
    rows = []
    for feature, categories in FEATURES:
        counts = dated[feature].value_counts()
        top_bucket = str(counts.index[0])
        top_count = int(counts.iloc[0])

        def resample(rng: np.random.Generator, year_counts=year_counts, feature=feature) -> float:
            sampled = []
            for year, count in year_counts.items():
                start = pd.Timestamp(year=int(year), month=1, day=1)
                end = pd.Timestamp(year=int(year), month=12, day=31)
                span = (end - start).days + 1
                offsets = rng.integers(0, span, size=int(count))
                sampled.extend(start + pd.to_timedelta(offsets, unit="D"))
            series = pd.Series(sampled)
            if feature == "weekday":
                buckets = series.dt.day_name()
            elif feature == "month":
                buckets = series.dt.month_name()
            else:
                buckets = "Q" + series.dt.quarter.astype(str)
            return float(buckets.value_counts().max())

        p_value = permutation_p_value(
            float(top_count),
            resample,
            draws=PERMUTATION_DRAWS,
            seed=SEED + categories,
        )
        rows.append(
            {
                "feature": feature,
                "top_bucket": top_bucket,
                "top_count": top_count,
                "releases": int(len(dated)),
                "share": top_count / len(dated),
                "null_categories": categories,
                "null_model": "year-preserving uniform random dates",
                "permutation_p_value": p_value,
                "permutation_draws": PERMUTATION_DRAWS,
            }
        )

    frame = pd.DataFrame(rows)
    rejected, q_values = benjamini_hochberg(frame["permutation_p_value"].to_numpy())
    frame["q_value_bh"] = q_values
    frame["significant_after_bh"] = rejected
    frame["family_size"] = int(len(frame))
    frame["surviving_after_correction"] = int(rejected.sum())
    surviving = int(rejected.sum())
    if surviving == 0:
        interpretation = (
            "No calendar feature survives Benjamini-Hochberg correction. Apparent clusters that "
            "look impressive before correction are the spurious-pattern lesson this module exists "
            "to make visible."
        )
    else:
        interpretation = (
            f"{surviving} of {len(frame)} calendar features survive Benjamini-Hochberg correction "
            "against a year-preserving random-date null. That is consistent with real release "
            "scheduling (weekdays, conference seasons) rather than a multiple-testing artifact. "
            "It is not evidence of anything beyond vendor calendars."
        )
    frame["interpretation"] = interpretation
    return frame.sort_values("permutation_p_value").reset_index(drop=True)
