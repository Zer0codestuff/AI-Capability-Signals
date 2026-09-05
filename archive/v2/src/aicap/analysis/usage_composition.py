"""What does observed Claude usage look like across occupations?

Question
--------
On each measured surface (consumer product and first-party API), which occupations account for the
most usage, how much of that usage is classified as automation versus augmentation, and how much do
the two surfaces disagree about the same occupations?

What this module refuses
------------------------
Anything about employment, wages, job loss or "replacement". The Anthropic Economic Index measures
the composition of one vendor's observed usage. Usage share is not employment impact, and an
occupation with high usage may be growing. The previous version constructed "substitution pressure"
and "replacement feasibility" indices from keyword hits on O*NET text and ranked real occupations
by how replaceable they were. That claim had the weakest support of anything in the previous
project and is not revived here.

The surface contrast is the finding
-----------------------------------
The same occupations show systematically different automation shares on the consumer product and on
the first-party API. Averaging them into one "share of work automated" number would invent a
consensus that does not exist. The contrast is reported; no average is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BOOTSTRAP_DRAWS, SEED
from ..provenance import RunContext
from ..stats import bootstrap_ci, paired_bootstrap_ci

#: Minimum usage share for an occupation to appear in "top occupations" tables. Below this the
#: occupation is in the long tail where sampling noise dominates the published percentages.
MIN_USAGE_SHARE_PCT = 0.05


def build(aei: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    periods = sorted(pd.to_datetime(aei["period_start"]).dropna().unique())
    if len(periods) < 1:
        context.refuse(
            claim="Composition of observed AI usage by occupation.",
            reason="No dated periods in the Economic Index release.",
            unblocked_by="A release that covers at least one monthly period.",
            analysis="usage_composition",
        )
        return {}

    # Use the most recent month for cross-sectional claims. Two months is not a panel long enough
    # for a trend, and that refusal is recorded explicitly.
    latest = aei[aei["period_start"] == periods[-1]].copy()
    detailed = latest[latest["hierarchy_level"] == 0].copy()

    if len(periods) < 3:
        context.refuse(
            claim="Trend in the automation share of observed AI usage.",
            reason=(
                f"The release covers {len(periods)} monthly period(s). A trend needs a longer "
                "panel; with this little history any slope is indistinguishable from noise."
            ),
            unblocked_by="A longer run of monthly releases from the same surface.",
            analysis="usage_composition",
        )

    tables: dict[str, pd.DataFrame] = {
        "usage_surface_summary": _surface_summary(detailed),
        "usage_top_occupations": _top_occupations(detailed),
        "usage_surface_contrast": _surface_contrast(detailed, context),
        "usage_major_group_composition": _major_group_composition(detailed),
    }
    return tables


def _surface_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    """Occupation-level medians of automation, augmentation and autonomy, per surface."""
    rows = []
    for surface, group in detailed.groupby("surface"):
        for metric, column in (
            ("automation_share_pct", "automation_share_pct"),
            ("augmentation_share_pct", "augmentation_share_pct"),
            ("ai_autonomy_mean", "ai_autonomy_mean"),
        ):
            values = group[column].dropna().to_numpy(dtype=float)
            point, low, high = bootstrap_ci(values, statistic=np.median, min_n=8)
            rows.append(
                {
                    "surface": surface,
                    "period_start": group["period_start"].iloc[0],
                    "occupations": int(len(group)),
                    "metric": metric,
                    "median": point,
                    "ci_low": low,
                    "ci_high": high,
                    "estimator": "percentile_bootstrap_of_median",
                    "scope": (
                        "Median across occupations with published values. Occupations are not "
                        "employment-weighted: this describes the composition of the published "
                        "occupation set, not of the workforce."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _top_occupations(detailed: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    rows = []
    for surface, group in detailed.groupby("surface"):
        ranked = group[group["usage_share_pct"] >= MIN_USAGE_SHARE_PCT].sort_values(
            "usage_share_pct", ascending=False
        )
        for _, row in ranked.head(top_n).iterrows():
            rows.append(
                {
                    "surface": surface,
                    "period_start": row["period_start"],
                    "soc_code": row["soc_code"],
                    "occupation": row["occupation"],
                    "soc_major_group": row["soc_major_group"],
                    "usage_share_pct": float(row["usage_share_pct"]),
                    "automation_share_pct": float(row["automation_share_pct"])
                    if pd.notna(row["automation_share_pct"])
                    else float("nan"),
                    "augmentation_share_pct": float(row["augmentation_share_pct"])
                    if pd.notna(row["augmentation_share_pct"])
                    else float("nan"),
                    "ai_autonomy_mean": float(row["ai_autonomy_mean"])
                    if pd.notna(row["ai_autonomy_mean"])
                    else float("nan"),
                    "caveat": (
                        "Usage share is the share of this vendor's observed conversations, not a "
                        "share of employment or of work performed."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _surface_contrast(detailed: pd.DataFrame, context: RunContext) -> pd.DataFrame:
    """Paired comparison of the same occupations across the two surfaces.

    Occupations that appear on both surfaces are the unit of analysis. The difference in automation
    share is bootstrapped over occupations, so the interval answers: how much would the contrast
    move if a different set of occupations were published.
    """
    surfaces = sorted(detailed["surface"].unique())
    if len(surfaces) < 2:
        context.refuse(
            claim="Whether consumer and API surfaces agree about occupation automation shares.",
            reason="Only one surface is present in this release.",
            unblocked_by="A release that covers both the consumer product and the first-party API.",
            analysis="usage_composition",
        )
        return pd.DataFrame()

    left, right = surfaces[0], surfaces[1]
    left_frame = detailed[detailed["surface"] == left].set_index("soc_code")
    right_frame = detailed[detailed["surface"] == right].set_index("soc_code")
    common = left_frame.index.intersection(right_frame.index)
    if len(common) < 20:
        context.refuse(
            claim="Surface contrast in occupation automation shares.",
            reason=f"Only {len(common)} occupations appear on both surfaces with published values.",
            unblocked_by="Wider occupation coverage on both surfaces.",
            analysis="usage_composition",
        )
        return pd.DataFrame()

    left_auto = left_frame.loc[common, "automation_share_pct"].to_numpy(dtype=float)
    right_auto = right_frame.loc[common, "automation_share_pct"].to_numpy(dtype=float)
    mask = np.isfinite(left_auto) & np.isfinite(right_auto)
    left_auto, right_auto = left_auto[mask], right_auto[mask]

    def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a - b))

    point, low, high = paired_bootstrap_ci(
        list(zip(left_auto, right_auto, strict=True)),
        statistic=mean_difference,
        draws=min(BOOTSTRAP_DRAWS, 5000),
        seed=SEED,
        min_n=20,
    )

    # Also report Spearman agreement so a large mean difference can be read against whether the
    # *ranking* of occupations still agrees across surfaces.
    from ..stats import spearman_rho

    return pd.DataFrame(
        [
            {
                "surface_a": left,
                "surface_b": right,
                "occupations_in_common": int(mask.sum()),
                "mean_automation_share_a": float(np.mean(left_auto)),
                "mean_automation_share_b": float(np.mean(right_auto)),
                "mean_difference_a_minus_b": point,
                "ci_low": low,
                "ci_high": high,
                "spearman_rho_of_automation_shares": spearman_rho(left_auto, right_auto),
                "estimator": "paired_bootstrap_of_mean_difference_over_occupations",
                "interpretation": (
                    "Mean difference in automation share for the same occupations across the two "
                    "surfaces. A difference whose interval excludes zero means the surfaces disagree "
                    "about how automated usage is, even for the same jobs. Averaging them into one "
                    "number would invent a consensus."
                ),
            }
        ]
    )


def _major_group_composition(detailed: pd.DataFrame) -> pd.DataFrame:
    """Usage share rolled up to SOC major groups, per surface."""
    rows = []
    for (surface, major), group in detailed.groupby(["surface", "soc_major_group"]):
        rows.append(
            {
                "surface": surface,
                "soc_major_group": major,
                "occupations": int(len(group)),
                "usage_share_pct": float(group["usage_share_pct"].sum()),
                "median_automation_share_pct": float(group["automation_share_pct"].median()),
                "median_ai_autonomy_mean": float(group["ai_autonomy_mean"].median()),
                "top_occupation": str(
                    group.loc[group["usage_share_pct"].idxmax(), "occupation"]
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["surface", "usage_share_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )
