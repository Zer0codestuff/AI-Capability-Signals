"""Data-quality diagnostics, published as findings rather than footnotes.

Question: which properties of the ingested data constrain what can be claimed from it?

This module runs first and its output is meant to be read first. Every check here corresponds to a
way the previous version of this project produced a wrong number, so the table doubles as a
regression guard against those specific mistakes: if a check's severity rises, an analysis
downstream is at risk and the report says which one.

Severity vocabulary:

``blocking``
    A claim that would otherwise be published is withheld. Paired with a refusal.
``constraining``
    A claim is still publishable but its scope is narrowed (a shorter window, a subset).
``informational``
    Worth knowing, no effect on published claims.
"""

from __future__ import annotations

import pandas as pd

from ..provenance import RunContext
from ..sources import epoch as epoch_source
from ..sources import lmarena as lmarena_source
from ..sources import swebench as swebench_source
from ..taxonomy import EPOCH_ACCESSIBILITY_MAP, WEIGHTS_UNKNOWN

#: Days after which a dated leaderboard is treated as unable to support present-tense claims.
#: Set to one quarter: long enough to tolerate normal publication gaps, short enough that a
#: frozen leaderboard cannot be quoted as current.
STALENESS_LIMIT_DAYS = 90


def build(
    epoch: pd.DataFrame,
    openrouter: pd.DataFrame,
    lmarena: pd.DataFrame,
    swebench: pd.DataFrame,
    aei: pd.DataFrame,
    context: RunContext,
) -> dict[str, pd.DataFrame]:
    findings: list[dict[str, object]] = []

    findings.extend(_epoch_checks(epoch, context))
    findings.extend(_openrouter_checks(openrouter, context))
    findings.extend(_lmarena_checks(lmarena, context))
    findings.extend(_swebench_checks(swebench, context))
    findings.extend(_aei_checks(aei))

    frame = pd.DataFrame(findings)
    severity_order = {"blocking": 0, "constraining": 1, "informational": 2}
    frame["severity_rank"] = frame["severity"].map(severity_order)
    frame = frame.sort_values(["severity_rank", "source_id", "check"]).drop(columns="severity_rank")

    return {
        "data_quality_findings": frame.reset_index(drop=True),
        "lmarena_duplicate_report": lmarena_source.duplicate_report(lmarena),
    }


def _finding(
    source_id: str,
    check: str,
    severity: str,
    metric: str,
    value: object,
    interpretation: str,
    affects: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "check": check,
        "severity": severity,
        "metric": metric,
        "value": value,
        "interpretation": interpretation,
        "affects_analysis": affects,
    }


def _epoch_checks(epoch: pd.DataFrame, context: RunContext) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    censored_from = epoch_source.censoring_boundary(epoch)
    dated = epoch.dropna(subset=["publication_year"])
    counts = dated.groupby("publication_year").size()
    latest_year = int(counts.index.max())

    out.append(
        _finding(
            "epoch_models",
            "curation_right_censoring",
            "blocking",
            "first_provisional_publication_year",
            censored_from,
            (
                f"Model counts fall monotonically from {censored_from} to {latest_year}, which is "
                "consistent with curation lag and also with a genuine slowdown. The two cannot be "
                "distinguished from this dataset, so release-count trends are not published for "
                "these years."
            ),
            "release_counts",
        )
    )
    context.refuse(
        claim="Number of notable AI models released per year, including the most recent years.",
        reason=(
            f"Epoch AI model counts decline monotonically from {censored_from} onward. Curation lag "
            "and a real slowdown produce identical shapes in this dataset."
        ),
        unblocked_by=(
            "A dataset revision that marks completeness per year, or waiting until the affected "
            "years stop growing between snapshots."
        ),
        analysis="data_quality",
    )

    for field_name, label in epoch_source.DISCLOSURE_FIELDS:
        rate = float(epoch[f"discloses_{field_name}"].mean())
        out.append(
            _finding(
                "epoch_models",
                "field_disclosure_rate",
                "constraining" if rate < 0.5 else "informational",
                f"share_disclosing::{field_name}",
                round(rate, 4),
                (
                    f"{label} is disclosed for {rate:.1%} of models. Any statistic conditioned on "
                    "this field describes disclosing models, which are not a random sample."
                ),
                "compute_scaling, disclosure",
            )
        )

    unknown_share = float((epoch["weights_class"] == WEIGHTS_UNKNOWN).mean())
    out.append(
        _finding(
            "epoch_models",
            "unmapped_accessibility_values",
            "constraining" if unknown_share > 0.1 else "informational",
            "share_weights_class_unknown",
            round(unknown_share, 4),
            (
                f"{unknown_share:.1%} of models have no usable accessibility class, either because "
                "the field is empty or because the accessibility statement and the open-weights "
                "flag disagree. These rows are excluded from open-versus-closed comparisons rather "
                "than assigned to the closed side."
            ),
            "disclosure, open_weights_lag",
        )
    )

    observed_values = {
        str(value).strip().lower() for value in epoch["accessibility_raw"].dropna().unique()
    }
    unmapped = sorted(observed_values - set(EPOCH_ACCESSIBILITY_MAP))
    if unmapped:
        out.append(
            _finding(
                "epoch_models",
                "accessibility_vocabulary_drift",
                "constraining",
                "unmapped_values",
                "; ".join(unmapped[:6]),
                (
                    "The source uses accessibility values this project does not map. They fall "
                    "through to 'unknown'. Extend EPOCH_ACCESSIBILITY_MAP to recover them."
                ),
                "disclosure",
            )
        )
    return out


def _openrouter_checks(openrouter: pd.DataFrame, context: RunContext) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tiered = int(openrouter["has_price_tiers"].sum())
    out.append(
        _finding(
            "openrouter_models",
            "prompt_length_price_tiers",
            "constraining",
            "models_with_price_tiers",
            tiered,
            (
                f"{tiered} of {len(openrouter)} listed models charge more above a prompt-length "
                "threshold. Reading only the base rate understates long-context cost, so price "
                "tables report the base rate and the rate at "
                f"{openrouter.attrs.get('long_prompt_tokens', 64000):,} prompt tokens separately."
            ),
            "price_structure",
        )
    )

    quality_coverage = float(openrouter["aa_intelligence_index"].notna().mean())
    out.append(
        _finding(
            "openrouter_models",
            "benchmark_index_coverage",
            "constraining",
            "share_with_intelligence_index",
            round(quality_coverage, 4),
            (
                f"A vendor-published quality index exists for {quality_coverage:.1%} of listed "
                "models. Price-versus-quality statements describe that subset, which skews toward "
                "prominent models."
            ),
            "price_structure, benchmark_agreement",
        )
    )

    context.refuse(
        claim="How listed prices for a given capability level changed over time.",
        reason=(
            "The catalogue is a cross-section of currently listed models. Withdrawn models are "
            "absent and no historical price field is published, so any 'price over time' series "
            "built from it would be a survivorship-biased cohort comparison."
        ),
        unblocked_by=(
            "Repeated snapshots of this catalogue captured over time, or a source that publishes "
            "dated historical prices."
        ),
        analysis="data_quality",
    )

    missing_price = int(openrouter["output_usd_per_1m"].isna().sum())
    if missing_price:
        out.append(
            _finding(
                "openrouter_models",
                "missing_output_price",
                "informational",
                "models_without_output_price",
                missing_price,
                (
                    "Some listings expose no usable completion price, typically non-text models. "
                    "They are excluded from price statistics."
                ),
                "price_structure",
            )
        )

    expiring = int(openrouter["expiration_date"].notna().sum())
    out.append(
        _finding(
            "openrouter_models",
            "announced_retirements",
            "informational",
            "models_with_expiration_date",
            expiring,
            (
                f"{expiring} listings carry an expiration date. This is direct evidence of the "
                "catalogue turnover that makes the cross-section unusable as a price history."
            ),
            "price_structure",
        )
    )
    return out


def _lmarena_checks(lmarena: pd.DataFrame, context: RunContext) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    duplicated_rows = int((lmarena["replicate_count"] > 1).sum())
    share = duplicated_rows / len(lmarena) if len(lmarena) else 0.0
    out.append(
        _finding(
            "lmarena_leaderboard",
            "duplicate_leaderboard_rows",
            "constraining",
            "share_of_rows_from_duplicate_keys",
            round(share, 4),
            (
                f"{duplicated_rows:,} rows ({share:.2%}) share a model, category and publication "
                "date with at least one other row carrying a different rating. They are combined by "
                "inverse-variance weighting with the between-replicate spread added to the standard "
                "error. Taking a maximum instead would bias affected models upward by tens of "
                "rating points."
            ),
            "open_weights_lag, benchmark_agreement",
        )
    )

    regimes = lmarena["methodology_regime"].value_counts()
    out.append(
        _finding(
            "lmarena_leaderboard",
            "rating_methodology_breaks",
            "constraining",
            "distinct_methodology_regimes",
            int(len(regimes)),
            (
                "The rating system changed on 2024-01-09 (Elo to Bradley-Terry), 2025-05-16 (style "
                "control by default) and 2025-07-23 (frequency re-weighting). Trends are estimated "
                "within a single regime; a slope spanning a break would partly measure the "
                "methodology change."
            ),
            "open_weights_lag",
        )
    )

    unknown_licence = float((lmarena["weights_class"] == WEIGHTS_UNKNOWN).mean())
    out.append(
        _finding(
            "lmarena_leaderboard",
            "unclassifiable_licence",
            "informational" if unknown_licence < 0.15 else "constraining",
            "share_licence_unknown",
            round(unknown_licence, 4),
            (
                f"{unknown_licence:.1%} of arena rows carry a licence string this project cannot "
                "classify as open or proprietary. They are excluded from the open-versus-closed "
                "frontier rather than assumed closed."
            ),
            "open_weights_lag",
        )
    )

    context.refuse(
        claim="A single number for how far behind open-weight models are, across all capabilities.",
        reason=(
            "The gap is measured per arena and per category and varies widely between them. One "
            "pooled number would hide that variation and depend on which categories were included."
        ),
        unblocked_by="Nothing in this data. The quantity is category-specific by construction.",
        analysis="data_quality",
    )
    return out


def _swebench_checks(swebench: pd.DataFrame, context: RunContext) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    stale = swebench_source.staleness_days(swebench, context.reference_date)
    severity = "blocking" if stale > STALENESS_LIMIT_DAYS else "informational"
    latest = pd.to_datetime(swebench["submission_date"], errors="coerce").max()
    out.append(
        _finding(
            "swebench_verified",
            "leaderboard_staleness",
            severity,
            "days_since_latest_submission",
            stale,
            (
                f"The most recent public submission is dated {latest.date()}, {stale} days before "
                "the reference date. The leaderboard supports historical statements only; it cannot "
                "describe current agent capability."
            ),
            "swebench_progress",
        )
    )
    if severity == "blocking":
        context.refuse(
            claim="Current state-of-the-art coding-agent capability on SWE-bench Verified.",
            reason=(
                f"The public submission directory has received no entry for {stale} days, so its "
                "top score is a lower bound from an earlier period, not a current measurement."
            ),
            unblocked_by="New submissions to the public SWE-bench Verified experiments directory.",
            analysis="data_quality",
        )

    unresolved_attribution = int(swebench["uses_open_weights_model"].isna().sum())
    out.append(
        _finding(
            "swebench_verified",
            "scaffold_model_entanglement",
            "constraining",
            "submissions_without_weight_attribution",
            unresolved_attribution,
            (
                "A submission measures an agent scaffold together with a model. Directory names are "
                "author-chosen, so the underlying model cannot always be identified. Scores are "
                "attributed to systems, never to models alone."
            ),
            "swebench_progress",
        )
    )
    return out


def _aei_checks(aei: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    periods = sorted(pd.to_datetime(aei["period_start"]).dt.date.unique())
    out.append(
        _finding(
            "anthropic_economic_index",
            "panel_length",
            "blocking",
            "monthly_periods_available",
            len(periods),
            (
                f"The release covers {len(periods)} monthly period(s) "
                f"({periods[0]} to {periods[-1]}). Cross-sectional comparisons are supported; "
                "trends in usage composition are not."
            ),
            "usage_composition",
        )
    )
    out.append(
        _finding(
            "anthropic_economic_index",
            "single_vendor_coverage",
            "constraining",
            "vendors_covered",
            1,
            (
                "Usage composition is measured for one vendor's products. It describes who uses "
                "that product and how, and carries no information about the workforce or about "
                "employment outcomes."
            ),
            "usage_composition",
        )
    )
    return out
