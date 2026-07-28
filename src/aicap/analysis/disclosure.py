"""How much do developers disclose about the models they release?

Question
--------
For each pre-registered disclosure field, what share of notable models report it, and does that
share differ between open-weight and closed-weight releases?

Why this question is answerable when most "capability" questions are not
-----------------------------------------------------------------------
Disclosure is a property of the *record*, not a latent trait needing a measurement model. Either
the field is populated or it is not. The denominator is the curated model list, the numerator is a
count, and the uncertainty is binomial. There is no scale to invent and no unit conversion to get
wrong, which is why this is the project's most solidly grounded result.

What would falsify the headline
-------------------------------
The claim is that open-weight releases disclose more. It is falsified if the difference in
disclosure rate is not distinguishable from zero once multiple comparisons across fields are
accounted for, or if it reverses on the frontier-flagged subset.

Known confound, stated rather than adjusted away
------------------------------------------------
Open-weight releases skew academic and closed releases skew commercial, so the comparison is
partly publication culture rather than a policy choice about openness. This is reported as a
limitation; the data contain no instrument that would separate the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BOOTSTRAP_DRAWS, SEED
from ..provenance import RunContext
from ..sources.epoch import DISCLOSURE_FIELDS, censoring_boundary
from ..stats import benjamini_hochberg, wilson_interval
from ..taxonomy import WEIGHTS_CLOSED, WEIGHTS_OPEN, WEIGHTS_RESTRICTED

#: Earliest year included. Chosen so that every year has enough models for a usable binomial
#: interval, and because pre-2018 records are curated retrospectively under different norms.
FIRST_YEAR = 2018

#: Fields excluded from the open-versus-closed comparison because the comparison would be circular.
#: ``accessibility_raw`` is the field the weights class is *derived from*, so within any group that
#: has a known weights class its disclosure rate is 1.0 by construction. It stays in the by-year and
#: by-class tables, where it measures something real (the share of all models with an accessibility
#: statement at all), and is dropped only from the test that its presence would trivially pass.
CIRCULAR_FIELDS = frozenset({"accessibility_raw"})


def build(epoch: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    censored_from = censoring_boundary(epoch)
    dated = epoch.dropna(subset=["publication_year"]).copy()
    dated["publication_year"] = dated["publication_year"].astype(int)
    window = dated[(dated["publication_year"] >= FIRST_YEAR) & (dated["publication_year"] < censored_from)]

    if window.empty:
        context.refuse(
            claim="Disclosure rates by year.",
            reason="No publication years survive the censoring cut.",
            unblocked_by="A less censored snapshot of the model dataset.",
            analysis="disclosure",
        )
        return {}

    return {
        "disclosure_by_year": _by_year(window),
        "disclosure_by_weights_class": _by_weights_class(window),
        "disclosure_open_vs_closed": _open_vs_closed(window, context),
        "disclosure_by_vendor": _by_vendor(window),
    }


def _by_year(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in window.groupby("publication_year"):
        for field_name, label in DISCLOSURE_FIELDS:
            successes = int(group[f"discloses_{field_name}"].sum())
            total = int(len(group))
            rate, low, high = wilson_interval(successes, total)
            rows.append(
                {
                    "publication_year": int(year),
                    "field": field_name,
                    "field_label": label,
                    "models": total,
                    "models_disclosing": successes,
                    "disclosure_rate": rate,
                    "ci_low": low,
                    "ci_high": high,
                    "estimator": "wilson_score_interval",
                }
            )
    return pd.DataFrame(rows).sort_values(["field", "publication_year"]).reset_index(drop=True)


def _by_weights_class(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weights_class, group in window.groupby("weights_class"):
        for field_name, label in DISCLOSURE_FIELDS:
            successes = int(group[f"discloses_{field_name}"].sum())
            total = int(len(group))
            rate, low, high = wilson_interval(successes, total)
            rows.append(
                {
                    "weights_class": weights_class,
                    "field": field_name,
                    "field_label": label,
                    "models": total,
                    "models_disclosing": successes,
                    "disclosure_rate": rate,
                    "ci_low": low,
                    "ci_high": high,
                    "estimator": "wilson_score_interval",
                }
            )
    return pd.DataFrame(rows).sort_values(["field", "weights_class"]).reset_index(drop=True)


def _open_vs_closed(window: pd.DataFrame, context: RunContext) -> pd.DataFrame:
    """Test the disclosure-rate difference between open- and closed-weight releases.

    The difference is tested per field with a two-sample bootstrap on the difference of
    proportions, then Benjamini-Hochberg is applied across the fields. Correcting across the family
    of fields matters: five tests at nominal 5% give a better-than-one-in-five chance of at least
    one spurious "significant" difference, and the previous version's appendix made exactly that
    mistake while claiming to demonstrate it.
    """
    open_group = window[window["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED])]
    closed_group = window[window["weights_class"] == WEIGHTS_CLOSED]

    if len(open_group) < 30 or len(closed_group) < 30:
        context.refuse(
            claim="Open-weight releases disclose more than closed-weight releases.",
            reason=(
                f"Only {len(open_group)} open-weight and {len(closed_group)} closed-weight models "
                "fall in the analysis window; the difference cannot be estimated usefully."
            ),
            unblocked_by="More classified models in the window.",
            analysis="disclosure",
        )
        return pd.DataFrame()

    rng = np.random.default_rng(SEED)
    rows = []
    tested_fields = [
        (field_name, label) for field_name, label in DISCLOSURE_FIELDS if field_name not in CIRCULAR_FIELDS
    ]
    for field_name, label in tested_fields:
        open_flags = open_group[f"discloses_{field_name}"].to_numpy(dtype=float)
        closed_flags = closed_group[f"discloses_{field_name}"].to_numpy(dtype=float)
        observed = float(open_flags.mean() - closed_flags.mean())

        open_draws = rng.integers(0, open_flags.size, size=(BOOTSTRAP_DRAWS, open_flags.size))
        closed_draws = rng.integers(0, closed_flags.size, size=(BOOTSTRAP_DRAWS, closed_flags.size))
        differences = open_flags[open_draws].mean(axis=1) - closed_flags[closed_draws].mean(axis=1)

        # Two-sided bootstrap p-value: the share of resampled differences on the opposite side of
        # zero from the observed difference, doubled and capped at 1.
        if observed >= 0:
            tail = float((differences <= 0).mean())
        else:
            tail = float((differences >= 0).mean())
        p_value = min(1.0, 2 * tail + 1.0 / BOOTSTRAP_DRAWS)

        rows.append(
            {
                "field": field_name,
                "field_label": label,
                "open_weight_models": int(open_flags.size),
                "closed_weight_models": int(closed_flags.size),
                "open_disclosure_rate": float(open_flags.mean()),
                "closed_disclosure_rate": float(closed_flags.mean()),
                "difference": observed,
                "ci_low": float(np.quantile(differences, 0.025)),
                "ci_high": float(np.quantile(differences, 0.975)),
                "p_value": p_value,
                "estimator": "two_sample_bootstrap_difference_of_proportions",
            }
        )

    frame = pd.DataFrame(rows)
    rejected, q_values = benjamini_hochberg(frame["p_value"].to_numpy())
    frame["q_value_bh"] = q_values
    frame["significant_after_bh"] = rejected
    frame["multiple_comparison_correction"] = f"benjamini_hochberg over {len(frame)} tested fields"
    frame["confound"] = (
        "Open-weight releases skew academic and closed releases skew commercial, so part of this "
        "difference is publication culture rather than a policy choice about openness. The data "
        "contain no instrument that separates the two."
    )
    return frame.sort_values("difference", ascending=False).reset_index(drop=True)


def _by_vendor(window: pd.DataFrame, min_models: int = 8) -> pd.DataFrame:
    """Per-vendor disclosure completeness.

    ``completeness`` is the share of the pre-registered fields a vendor populates, averaged over its
    models. It is a proportion with a stated denominator, not a latent index: no scaling, no
    weights, and each field counts once. Vendors below ``min_models`` are excluded because a
    proportion over a handful of models has an interval too wide to rank on.
    """
    flag_columns = [f"discloses_{field_name}" for field_name, _ in DISCLOSURE_FIELDS]
    rows = []
    for vendor, group in window.groupby("vendor"):
        if len(group) < min_models:
            continue
        # Denominator: models x fields. Each observation is one field of one model.
        successes = int(group[flag_columns].to_numpy(dtype=bool).sum())
        total = int(len(group) * len(flag_columns))
        rate, low, high = wilson_interval(successes, total)
        rows.append(
            {
                "vendor": vendor,
                "models": int(len(group)),
                "fields_per_model": len(flag_columns),
                "fields_disclosed": successes,
                "disclosure_completeness": rate,
                "ci_low": low,
                "ci_high": high,
                "open_weight_share": float(
                    group["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED]).mean()
                ),
                "estimator": "wilson_score_interval",
                "interval_note": (
                    "Fields within a model are not independent, so this interval understates "
                    "uncertainty. It is reported to show precision order-of-magnitude, and vendors "
                    "whose intervals overlap should not be read as ranked."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("disclosure_completeness", ascending=False).reset_index(drop=True)
