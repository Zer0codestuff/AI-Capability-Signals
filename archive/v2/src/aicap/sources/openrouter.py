"""OpenRouter public model catalogue.

This is the project's price source, and it carries two things the previous version missed.

**Prompt-length price tiers.** ``pricing.overrides`` raises the per-token rate above a prompt-length
threshold. Reading only the base rate understates the cost of long-context work for every model
that has tiers. Both the base rate and the effective rate at a stated prompt length are computed
here, and analyses declare which they use.

**Vendor-published benchmark indices.** The catalogue includes Artificial Analysis intelligence,
coding and agentic indices and a Design Arena score *inside the same record as the price*. Quality
and price therefore come from one row, which removes the cross-source name join that produced the
previous version's worst errors.

The catalogue is a cross-section of currently listed models. It cannot support a price history, and
:func:`load` records no date other than each model's ``created`` timestamp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..netcache import Fetcher
from ..schema import Contract
from ..taxonomy import classify_family, classify_vendor, openrouter_weights_class

URL = "https://openrouter.ai/api/v1/models"

#: Prompt length at which the "effective" price is evaluated, in tokens. Chosen to sit above the
#: common 32k first tier so that tiered pricing is actually exercised, and stated in the report
#: wherever an effective price appears.
LONG_PROMPT_TOKENS = 64_000

#: Benchmark indices published inside the catalogue, with the scale each uses. Kept as separate
#: columns and never combined: their comparability is measured in
#: :mod:`aicap.analysis.benchmark_agreement`, not assumed.
BENCHMARK_COLUMNS = {
    "benchmarks.artificial_analysis.intelligence_index": "aa_intelligence_index",
    "benchmarks.artificial_analysis.coding_index": "aa_coding_index",
    "benchmarks.artificial_analysis.agentic_index": "aa_agentic_index",
    "benchmarks.design_arena": "design_arena_score",
}

CONTRACT = Contract(
    name="openrouter_models",
    required=(
        "model_id",
        "display_name",
        "vendor",
        "family",
        "created_date",
        "context_length",
        "input_usd_per_1m",
        "output_usd_per_1m",
        "input_usd_per_1m_long_prompt",
        "output_usd_per_1m_long_prompt",
        "has_price_tiers",
        "weights_class",
        "hugging_face_id",
        "aa_intelligence_index",
        "aa_coding_index",
        "modality",
    ),
    numeric=("context_length", "input_usd_per_1m", "output_usd_per_1m", "aa_intelligence_index"),
    non_null=("model_id", "display_name"),
    min_rows=50,
    unique_key=("model_id",),
    notes="One row per currently listed model.",
)


def load(fetcher: Fetcher) -> pd.DataFrame:
    artifact = fetcher.fetch("openrouter_models", URL, "openrouter/models.json")
    payload = artifact.read_json()
    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError("openrouter_models: response has no non-empty 'data' array.")

    flat = pd.json_normalize(records)
    for required in ("id", "name", "pricing.prompt", "pricing.completion", "context_length"):
        if required not in flat.columns:
            raise ValueError(
                f"openrouter_models: upstream payload lacks {required!r}. Present: {sorted(flat.columns)}"
            )

    frame = pd.DataFrame(
        {
            "model_id": flat["id"].astype(str),
            "canonical_slug": flat.get("canonical_slug", pd.Series(index=flat.index, dtype=object)),
            "display_name": flat["name"].astype(str),
            "hugging_face_id": flat.get("hugging_face_id", pd.Series(index=flat.index, dtype=object)),
            "context_length": pd.to_numeric(flat["context_length"], errors="coerce"),
            "max_output_tokens": pd.to_numeric(
                flat.get("top_provider.max_completion_tokens", pd.Series(index=flat.index)), errors="coerce"
            ),
            "modality": flat.get("architecture.modality", pd.Series(index=flat.index, dtype=object)),
            "knowledge_cutoff": flat.get("knowledge_cutoff", pd.Series(index=flat.index, dtype=object)),
            "expiration_date": flat.get("expiration_date", pd.Series(index=flat.index, dtype=object)),
            "input_usd_per_1m": _per_million(flat["pricing.prompt"]),
            "output_usd_per_1m": _per_million(flat["pricing.completion"]),
            "cache_read_usd_per_1m": _per_million(
                flat.get("pricing.input_cache_read", pd.Series(index=flat.index))
            ),
        }
    )

    frame["created_date"] = pd.to_datetime(
        pd.to_numeric(flat.get("created", pd.Series(index=flat.index)), errors="coerce"),
        unit="s",
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    overrides = flat["pricing.overrides"] if "pricing.overrides" in flat.columns else None
    tiered = [
        _effective_tier_price(overrides.iloc[i] if overrides is not None else None)
        for i in range(len(frame))
    ]
    frame["has_price_tiers"] = [bool(t["has_tiers"]) for t in tiered]
    frame["input_usd_per_1m_long_prompt"] = [
        _resolve_tier(base, tier["prompt"]) for base, tier in zip(frame["input_usd_per_1m"], tiered, strict=True)
    ]
    frame["output_usd_per_1m_long_prompt"] = [
        _resolve_tier(base, tier["completion"])
        for base, tier in zip(frame["output_usd_per_1m"], tiered, strict=True)
    ]

    for source_column, target in BENCHMARK_COLUMNS.items():
        frame[target] = pd.to_numeric(
            flat.get(source_column, pd.Series(index=flat.index, dtype=float)), errors="coerce"
        )

    frame["vendor"] = [
        classify_vendor(name, model_id.split("/")[0] if "/" in model_id else "")
        for name, model_id in zip(frame["display_name"], frame["model_id"], strict=True)
    ]
    frame["family"] = [
        classify_family(name, model_id) for name, model_id in zip(frame["display_name"], frame["model_id"], strict=True)
    ]
    frame["weights_class"] = [openrouter_weights_class(hf) for hf in frame["hugging_face_id"]]
    frame["is_free_tier"] = frame["model_id"].str.endswith(":free")
    frame["source_id"] = "openrouter_models"
    frame["source_url"] = "https://openrouter.ai/" + frame["model_id"]

    return CONTRACT.validate(frame.reset_index(drop=True))


def _per_million(series: pd.Series) -> pd.Series:
    """Convert OpenRouter per-token string prices to USD per million tokens.

    Negative sentinel values (used upstream for "not applicable") become NaN rather than negative
    prices, which would otherwise sit at the bottom of any "cheapest model" ranking.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric >= 0)
    return numeric * 1_000_000.0


def _effective_tier_price(overrides: Any) -> dict[str, Any]:
    """Extract the price tier that applies at :data:`LONG_PROMPT_TOKENS`.

    ``overrides`` is a list of ``{min_prompt_tokens, prompt, completion, ...}`` objects. The
    applicable tier is the one with the largest ``min_prompt_tokens`` not exceeding the evaluation
    length.
    """
    result: dict[str, Any] = {"has_tiers": False, "prompt": np.nan, "completion": np.nan}
    if not isinstance(overrides, list) or not overrides:
        return result
    result["has_tiers"] = True
    applicable: dict[str, Any] | None = None
    threshold = -1.0
    for tier in overrides:
        if not isinstance(tier, dict):
            continue
        minimum = pd.to_numeric(tier.get("min_prompt_tokens"), errors="coerce")
        if pd.isna(minimum) or minimum > LONG_PROMPT_TOKENS:
            continue
        if float(minimum) > threshold:
            threshold = float(minimum)
            applicable = tier
    if applicable is None:
        return result
    prompt = pd.to_numeric(applicable.get("prompt"), errors="coerce")
    completion = pd.to_numeric(applicable.get("completion"), errors="coerce")
    result["prompt"] = float(prompt) * 1e6 if pd.notna(prompt) and prompt >= 0 else np.nan
    result["completion"] = float(completion) * 1e6 if pd.notna(completion) and completion >= 0 else np.nan
    return result


def _resolve_tier(base: float, tiered: float) -> float:
    """The long-prompt price is the tier price when present, otherwise the flat base price."""
    if pd.notna(tiered):
        return float(tiered)
    return float(base) if pd.notna(base) else np.nan
