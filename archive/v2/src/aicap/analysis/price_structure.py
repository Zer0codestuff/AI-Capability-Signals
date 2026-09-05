"""What does capability cost, and how much does price vary at a fixed capability?

Question
--------
In the current catalogue, what is the cheapest listed way to obtain a given measured quality level,
how wide is the price spread among models of comparable quality, and does published-weight
availability predict a lower price once quality is held fixed?

Why the join problem disappears here
------------------------------------
Quality and price come from the *same catalogue record*: OpenRouter publishes Artificial Analysis
intelligence, coding and agentic indices alongside each listing's prices. There is no cross-source
name matching, so the failure mode that dominated the previous version — attaching a flagship's
benchmark score to a cheaper sibling — cannot occur in this module by construction.

What this module refuses
------------------------
Everything temporal. A catalogue of currently listed models is a cross-section: withdrawn models are
absent and no historical prices are published. Grouping today's prices by release year and reading a
slope off it, as the previous version did, measures survivorship, not price change. The refusal is
recorded rather than worked around.

Quality caveat that limits every number below
---------------------------------------------
The quality index is a vendor-published composite. This project does not endorse its construction;
it uses it because it is the only quality measure that arrives already keyed to a price. Its
agreement with independent benchmarks is measured separately in
:mod:`aicap.analysis.benchmark_agreement`, and the price conclusions inherit whatever validity that
measurement supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BOOTSTRAP_DRAWS, SEED
from ..provenance import RunContext
from ..sources.openrouter import LONG_PROMPT_TOKENS
from ..stats import bootstrap_ci, ols_hc3
from ..taxonomy import WEIGHTS_OPEN, WEIGHTS_RESTRICTED

#: Reference workload for the blended price, in tokens. A 3:1 input-to-output ratio matches typical
#: assistant traffic more closely than either price alone, and both component prices are also
#: published so a reader can reweight.
BLEND_INPUT_TOKENS = 3_000
BLEND_OUTPUT_TOKENS = 1_000

#: Quality thresholds for the "cheapest way to reach this level" table, on the vendor index scale.
QUALITY_THRESHOLDS = (20, 30, 40, 50, 55)

#: Width of the quality band used when measuring price spread at comparable quality.
QUALITY_BAND_WIDTH = 5.0


def build(openrouter: pd.DataFrame, context: RunContext) -> dict[str, pd.DataFrame]:
    priced = openrouter[
        openrouter["output_usd_per_1m"].notna()
        & openrouter["input_usd_per_1m"].notna()
        & ~openrouter["is_free_tier"]
    ].copy()

    priced["blended_usd_per_call"] = (
        priced["input_usd_per_1m"] * BLEND_INPUT_TOKENS
        + priced["output_usd_per_1m"] * BLEND_OUTPUT_TOKENS
    ) / 1_000_000.0
    priced["blended_usd_per_call_long_prompt"] = (
        priced["input_usd_per_1m_long_prompt"] * LONG_PROMPT_TOKENS
        + priced["output_usd_per_1m_long_prompt"] * BLEND_OUTPUT_TOKENS
    ) / 1_000_000.0
    priced["weights_available"] = priced["weights_class"].isin([WEIGHTS_OPEN, WEIGHTS_RESTRICTED])

    with_quality = priced[priced["aa_intelligence_index"].notna()].copy()

    tables: dict[str, pd.DataFrame] = {
        "price_distribution_by_weights": _price_distribution(priced),
        "price_tier_impact": _tier_impact(priced, context),
    }

    if len(with_quality) < 30:
        context.refuse(
            claim="Price of a given measured quality level.",
            reason=(
                f"Only {len(with_quality)} listed models carry a published quality index, which is "
                "too few to characterise the price-quality relationship."
            ),
            unblocked_by="Wider benchmark-index coverage in the catalogue.",
            analysis="price_structure",
        )
        return tables

    tables["price_quality_frontier"] = _pareto_frontier(with_quality)
    tables["cheapest_at_quality_threshold"] = _cheapest_at_threshold(with_quality)
    tables["price_spread_at_matched_quality"] = _spread_at_matched_quality(with_quality)
    tables["price_quality_regression"] = _price_regression(with_quality)
    return tables


def _price_distribution(priced: pd.DataFrame) -> pd.DataFrame:
    """Listed-price distribution by weight availability, with a bootstrap interval on the median.

    Reported as a *catalogue* description. It is not a like-for-like comparison, because the two
    groups differ in capability; the capability-controlled comparison is :func:`_price_regression`.
    """
    rows = []
    for label, group in (
        ("weights_published", priced[priced["weights_available"]]),
        ("weights_not_published", priced[~priced["weights_available"]]),
        ("all_listed", priced),
    ):
        for metric, column in (
            ("input_usd_per_1m", "input_usd_per_1m"),
            ("output_usd_per_1m", "output_usd_per_1m"),
            ("blended_usd_per_call", "blended_usd_per_call"),
        ):
            values = group[column].dropna().to_numpy(dtype=float)
            point, low, high = bootstrap_ci(values, statistic=np.median)
            rows.append(
                {
                    "group": label,
                    "metric": metric,
                    "models": int(values.size),
                    "median": point,
                    "ci_low": low,
                    "ci_high": high,
                    "p10": float(np.quantile(values, 0.10)) if values.size else float("nan"),
                    "p90": float(np.quantile(values, 0.90)) if values.size else float("nan"),
                    "estimator": "percentile_bootstrap_of_median",
                    "scope": "cross-section of currently listed models; not a price history",
                }
            )
    return pd.DataFrame(rows)


def _tier_impact(priced: pd.DataFrame, context: RunContext) -> pd.DataFrame:
    """How much prompt-length price tiers raise cost for the models that have them.

    The previous version read only the base rate. For tiered models the base rate understates the
    price of long-context work, which is exactly the workload their large context windows are sold
    for.
    """
    tiered = priced[priced["has_price_tiers"]].copy()
    if tiered.empty:
        context.refuse(
            claim="Effect of prompt-length price tiers.",
            reason="No listed model in this snapshot publishes prompt-length price overrides.",
            unblocked_by="A snapshot in which tiered pricing is present.",
            analysis="price_structure",
        )
        return pd.DataFrame()

    tiered["input_multiplier"] = tiered["input_usd_per_1m_long_prompt"] / tiered["input_usd_per_1m"]
    # A model can publish tiers whose first threshold sits above the evaluation length, in which case
    # the base rate still applies and the multiplier is exactly 1. Pooling those with genuinely
    # escalating models would drag the median to 1 and hide the effect, so the two groups are
    # separated and both counts are published.
    binding = tiered[np.isfinite(tiered["input_multiplier"]) & (tiered["input_multiplier"] > 1.0)]
    ratios = binding["input_multiplier"].to_numpy(dtype=float)
    point, low, high = bootstrap_ci(ratios, statistic=np.median, min_n=5)
    return pd.DataFrame(
        [
            {
                "models_listed": int(len(priced)),
                "models_with_tiers": int(len(tiered)),
                "share_with_tiers": float(len(tiered) / len(priced)),
                "evaluation_prompt_tokens": LONG_PROMPT_TOKENS,
                "models_with_tier_binding_at_evaluation_length": int(len(binding)),
                "median_input_price_multiplier_when_binding": point,
                "ci_low": low,
                "ci_high": high,
                "max_input_price_multiplier": float(np.nanmax(ratios)) if ratios.size else float("nan"),
                "estimator": "percentile_bootstrap_of_median",
                "interpretation": (
                    "Among listings whose price tier actually applies at "
                    f"{LONG_PROMPT_TOKENS:,} prompt tokens, the input price relative to the base "
                    "rate. Reading only the base rate understates long-context cost by this factor "
                    "for those models."
                ),
            }
        ]
    )


def _pareto_frontier(with_quality: pd.DataFrame) -> pd.DataFrame:
    """Identify the price-quality Pareto frontier.

    A listing is on the frontier when no other listing is both cheaper and at least as good. This is
    a partial order read directly off two measured columns — no weights, no scaling, no composite —
    which is why it survives when a weighted "price-performance index" would not.
    """
    frame = with_quality[
        [
            "model_id",
            "display_name",
            "vendor",
            "family",
            "weights_class",
            "weights_available",
            "aa_intelligence_index",
            "aa_coding_index",
            "aa_agentic_index",
            "input_usd_per_1m",
            "output_usd_per_1m",
            "blended_usd_per_call",
            "context_length",
            "created_date",
        ]
    ].copy()

    prices = frame["blended_usd_per_call"].to_numpy(dtype=float)
    quality = frame["aa_intelligence_index"].to_numpy(dtype=float)
    dominated = np.zeros(len(frame), dtype=bool)
    for i in range(len(frame)):
        # Dominated if some other listing is cheaper-or-equal and better-or-equal, strictly better on
        # at least one axis.
        cheaper_or_equal = prices <= prices[i]
        better_or_equal = quality >= quality[i]
        strictly_better = (prices < prices[i]) | (quality > quality[i])
        dominated[i] = bool(np.any(cheaper_or_equal & better_or_equal & strictly_better))

    frame["is_pareto_efficient"] = ~dominated
    frame["usd_per_index_point"] = frame["blended_usd_per_call"] / frame["aa_intelligence_index"]
    frame["workload_note"] = (
        f"blended cost of one call at {BLEND_INPUT_TOKENS:,} input and {BLEND_OUTPUT_TOKENS:,} "
        "output tokens, at list prices"
    )
    return frame.sort_values(
        ["is_pareto_efficient", "aa_intelligence_index"], ascending=[False, False]
    ).reset_index(drop=True)


def _cheapest_at_threshold(with_quality: pd.DataFrame) -> pd.DataFrame:
    """Cheapest listing at or above each quality threshold.

    This is the most directly useful and least assumption-laden statistic in the module: a minimum
    over a filtered set, with the winning model named so a reader can check it.
    """
    rows = []
    for threshold in QUALITY_THRESHOLDS:
        eligible = with_quality[with_quality["aa_intelligence_index"] >= threshold]
        if eligible.empty:
            rows.append(
                {
                    "quality_threshold": threshold,
                    "eligible_models": 0,
                    "cheapest_model_id": None,
                    "cheapest_blended_usd_per_call": float("nan"),
                    "status": "no_listed_model_reaches_threshold",
                }
            )
            continue
        winner = eligible.loc[eligible["blended_usd_per_call"].idxmin()]
        best_quality = eligible.loc[eligible["aa_intelligence_index"].idxmax()]
        rows.append(
            {
                "quality_threshold": threshold,
                "eligible_models": int(len(eligible)),
                "cheapest_model_id": str(winner["model_id"]),
                "cheapest_model_vendor": str(winner["vendor"]),
                "cheapest_model_weights": str(winner["weights_class"]),
                "cheapest_model_quality": float(winner["aa_intelligence_index"]),
                "cheapest_blended_usd_per_call": float(winner["blended_usd_per_call"]),
                "premium_vs_cheapest": float(
                    best_quality["blended_usd_per_call"] / winner["blended_usd_per_call"]
                ),
                "highest_quality_model_id": str(best_quality["model_id"]),
                "status": "estimated",
            }
        )
    return pd.DataFrame(rows)


def _spread_at_matched_quality(with_quality: pd.DataFrame, min_models: int = 5) -> pd.DataFrame:
    """Price spread within narrow quality bands.

    If quality determined price, the spread inside a band would be small. A large spread means the
    market is not pricing measured capability alone, which is a substantive finding and one that a
    single "price-performance index" would have averaged into invisibility.
    """
    frame = with_quality.copy()
    lowest = float(np.floor(frame["aa_intelligence_index"].min() / QUALITY_BAND_WIDTH) * QUALITY_BAND_WIDTH)
    frame["band_low"] = (
        np.floor((frame["aa_intelligence_index"] - lowest) / QUALITY_BAND_WIDTH) * QUALITY_BAND_WIDTH + lowest
    )

    rows = []
    for band_low, group in frame.groupby("band_low"):
        if len(group) < min_models:
            continue
        prices = group["blended_usd_per_call"].to_numpy(dtype=float)
        ratio = float(np.nanmax(prices) / np.nanmin(prices)) if np.nanmin(prices) > 0 else float("nan")
        point, low, high = bootstrap_ci(
            prices,
            statistic=lambda sample: float(np.nanmax(sample) / np.nanmin(sample))
            if np.nanmin(sample) > 0
            else float("nan"),
            min_n=min_models,
        )
        rows.append(
            {
                "band_low": float(band_low),
                "quality_band": f"{band_low:.0f}-{band_low + QUALITY_BAND_WIDTH:.0f}",
                "models": int(len(group)),
                "quality_min": float(group["aa_intelligence_index"].min()),
                "quality_max": float(group["aa_intelligence_index"].max()),
                "cheapest_usd_per_call": float(np.nanmin(prices)),
                "dearest_usd_per_call": float(np.nanmax(prices)),
                "max_min_price_ratio": ratio,
                "ratio_ci_low": low,
                "ratio_ci_high": high,
                "cheapest_model_id": str(group.loc[group["blended_usd_per_call"].idxmin(), "model_id"]),
                "dearest_model_id": str(group.loc[group["blended_usd_per_call"].idxmax(), "model_id"]),
                "estimator": "percentile_bootstrap_of_max_min_ratio",
                "caveat": (
                    "A max/min ratio is a function of extremes, so its bootstrap interval is wide "
                    "and asymmetric. Read the ratio as an order of magnitude, not a precise figure."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("band_low").reset_index(drop=True)


def _price_regression(with_quality: pd.DataFrame) -> pd.DataFrame:
    """Does weight availability predict a lower price once measured quality is held fixed?

    Specification: ``log10(blended price) ~ quality + weights_available``, fitted by OLS with HC3
    standard errors. The weights coefficient is reported as a multiplicative price factor.

    This is an association in a catalogue, not a causal estimate. Published-weight models are served
    by competing hosts while closed models are served by their developer, so the coefficient mixes
    the effect of openness with the effect of host competition. That is stated rather than adjusted,
    because nothing in this data separates them.
    """
    frame = with_quality[with_quality["blended_usd_per_call"] > 0].copy()
    log_price = np.log10(frame["blended_usd_per_call"].to_numpy(dtype=float))
    quality = frame["aa_intelligence_index"].to_numpy(dtype=float)
    open_flag = frame["weights_available"].to_numpy(dtype=float)

    design = np.column_stack([np.ones(len(frame)), quality, open_flag])
    coefficients, ses = _multiple_ols_hc3(design, log_price)

    quality_only = ols_hc3(quality, log_price)
    rng = np.random.default_rng(SEED)
    # Bootstrap the open-weights coefficient over models, so the interval reflects resampling the
    # unit of analysis rather than only the sandwich approximation.
    boot = []
    for _ in range(min(BOOTSTRAP_DRAWS // 5, 2000)):
        idx = rng.integers(0, len(frame), size=len(frame))
        try:
            coef, _ = _multiple_ols_hc3(design[idx], log_price[idx])
        except np.linalg.LinAlgError:
            continue
        boot.append(coef[2])
    boot_array = np.asarray(boot, dtype=float)

    rows = [
        {
            "term": "quality_index",
            "coefficient_log10_price": coefficients[1],
            "se_hc3": ses[1],
            "ci_low": coefficients[1] - 1.96 * ses[1],
            "ci_high": coefficients[1] + 1.96 * ses[1],
            "price_factor": float(10.0 ** coefficients[1]),
            "interpretation": (
                "Multiplicative change in list price per one point of the vendor quality index, "
                "holding weight availability fixed."
            ),
        },
        {
            "term": "weights_published",
            "coefficient_log10_price": coefficients[2],
            "se_hc3": ses[2],
            "ci_low": float(np.quantile(boot_array, 0.025)) if boot_array.size else float("nan"),
            "ci_high": float(np.quantile(boot_array, 0.975)) if boot_array.size else float("nan"),
            "price_factor": float(10.0 ** coefficients[2]),
            "interpretation": (
                "Multiplicative change in list price for listings whose weights are published, at "
                "equal measured quality. Interval is a model-level bootstrap."
            ),
        },
    ]
    frame_out = pd.DataFrame(rows)
    frame_out["n"] = int(len(frame))
    frame_out["r_squared_quality_only"] = quality_only.r_squared
    frame_out["specification"] = "log10(blended_usd_per_call) ~ quality_index + weights_published"
    frame_out["estimator"] = "ols_hc3_plus_model_level_bootstrap"
    frame_out["causal_status"] = (
        "association in a catalogue. Published-weight models are served by competing hosts while "
        "closed models are served by their developer, so this mixes openness with host competition."
    )
    return frame_out


def _multiple_ols_hc3(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multiple regression with HC3 standard errors, returning ``(coefficients, ses)``."""
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residuals = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    weights = (residuals / np.clip(1.0 - leverage, 1e-8, None)) ** 2
    covariance = xtx_inv @ (design.T @ (design * weights[:, None])) @ xtx_inv
    return beta, np.sqrt(np.clip(np.diag(covariance), 0.0, None))
