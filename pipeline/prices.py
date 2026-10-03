"""Attach a usage price to models of the Epoch Capabilities Index.

No single public dataset has launch prices for every model, so four sources are
combined in a fixed order of trust. Each attached price keeps its provenance:

1. ``observed``: a price Epoch AI recorded at the time (2021 to early 2025).
2. ``list_initial``: the vendor list price before a later documented price change.
3. ``list_current``: today's first-party list price (models.dev, then llm-prices.com),
   used as the launch price.
4. ``list_current`` via OpenRouter, only for closed models, where OpenRouter passes
   the vendor list price through.

Today's third-party hosting prices of open-weight models are deliberately not used for
the historical series: assigning a 2026 hosting price to a 2024 release would make the
past look cheaper than it was.

Prices are USD per million tokens, blended 3:1 between input and output tokens.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

ALIASES = json.loads((Path(__file__).parent / "aliases.json").read_text())

# Words that describe packaging rather than a different model.
NOISE = {"preview", "latest", "instruct", "chat", "beta", "it", "experimental", "exp"}

# models.dev provider ids that are the developer's own API, per ECI organisation.
FIRST_PARTY = {
    "OpenAI": ["openai"],
    "Anthropic": ["anthropic"],
    "Google DeepMind": ["google"],
    "DeepSeek": ["deepseek"],
    "Mistral AI": ["mistral"],
    "xAI": ["xai"],
    "Alibaba": ["alibaba"],
    "Moonshot": ["moonshotai"],
    "Z.ai (Zhipu AI)": ["zai", "zhipuai"],
    "MiniMax": ["minimax"],
    "Cohere": ["cohere"],
}

OPENROUTER_PREFIX = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google DeepMind",
    "x-ai": "xAI",
    "deepseek": "DeepSeek",
    "mistralai": "Mistral AI",
    "qwen": "Alibaba",
    "moonshotai": "Moonshot",
    "z-ai": "Z.ai (Zhipu AI)",
    "minimax": "MiniMax",
    "meta-llama": "Meta AI",
    "amazon": "Amazon",
    "cohere": "Cohere",
}


@dataclass(frozen=True)
class Price:
    usd: float
    source: str
    kind: str


def blended(price_in: float, price_out: float) -> float:
    return (3 * price_in + price_out) / 4


def normalise(name: str, *, keep_variant: bool = True) -> str:
    text = name.lower()
    if not keep_variant:
        text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    words = [word for word in text.split() if word not in NOISE]
    return " ".join(words)


def keys(name: str) -> list[str]:
    """Lookup keys from most to least specific."""
    full = normalise(name)
    base = normalise(name, keep_variant=False)
    return [full] if full == base else [full, base]


def _observed(epoch_files: list[bytes]) -> dict[str, tuple[str, float]]:
    """Earliest Epoch observation per source model name."""
    earliest: dict[str, tuple[str, float]] = {}
    for body in epoch_files:
        for row in csv.DictReader(io.StringIO(body.decode("utf-8-sig"))):
            name = row["Model Name"].strip()
            try:
                usd = float(row["USD per 1M Tokens"])
            except ValueError:
                continue
            when = row["Release Date"].strip()
            if usd > 0 and (name not in earliest or when < earliest[name][0]):
                earliest[name] = (when, usd)
    return earliest


def _vendor_list(history: dict) -> dict[str, tuple[float, str]]:
    """First known vendor list price per model.

    A record without a start date is the price the model launched with. If it also has
    an end date, the vendor changed the price later, so the launch price is documented.
    """
    result: dict[str, tuple[float, str]] = {}
    for record in history.get("prices", []):
        if record.get("from_date") is not None:
            continue
        if not record.get("input") or not record.get("output"):
            continue
        usd = blended(float(record["input"]), float(record["output"]))
        kind = "list_initial" if record.get("to_date") else "list_current"
        for key in keys(record["name"]) + keys(record["id"]):
            result.setdefault(key, (usd, kind))
    return result


def _first_party(catalogue: dict) -> dict[str, dict[str, float]]:
    """Current first-party list prices per ECI organisation, keyed by normalised name."""
    result: dict[str, dict[str, float]] = {}
    for organisation, providers in FIRST_PARTY.items():
        table = result.setdefault(organisation, {})
        for provider in providers:
            for model_id, model in (catalogue.get(provider, {}).get("models") or {}).items():
                cost = model.get("cost") or {}
                if not cost.get("input") or not cost.get("output"):
                    continue
                usd = blended(float(cost["input"]), float(cost["output"]))
                for key in keys(model.get("name") or model_id) + keys(model_id):
                    table.setdefault(key, usd)
    return result


def _openrouter(catalogue: dict) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model in catalogue.get("data", []):
        model_id = model.get("id", "")
        if ":" in model_id or "/" not in model_id:
            continue
        organisation = OPENROUTER_PREFIX.get(model_id.split("/", 1)[0])
        pricing = model.get("pricing") or {}
        try:
            price_in = float(pricing.get("prompt", 0)) * 1e6
            price_out = float(pricing.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            continue
        if organisation is None or price_in <= 0 or price_out <= 0:
            continue
        table = result.setdefault(organisation, {})
        display = model.get("name", "").split(": ", 1)[-1]
        for key in keys(display) + keys(model_id.split("/", 1)[1]):
            table.setdefault(key, blended(price_in, price_out))
    return result


def organisation_of(raw: str) -> str:
    first = raw.split(",")[0].strip()
    return {"Google": "Google DeepMind", "Microsoft Research": "Microsoft"}.get(first, first)


def attach(
    models: list[dict],
    *,
    epoch_files: list[bytes],
    history: dict,
    models_dev: dict,
    openrouter: dict,
) -> dict[str, Price]:
    """Return a price for every ECI model that can be matched, keyed by model name."""
    observed_raw = _observed(epoch_files)
    names = {model["name"] for model in models}
    observed: dict[str, float] = {}
    index = {key: model["name"] for model in models for key in keys(model["name"])[:1]}
    for source_name, (_, usd) in sorted(observed_raw.items(), key=lambda item: item[1][0]):
        target = ALIASES.get(source_name)
        if target is None:
            target = index.get(normalise(source_name))
        if target in names:
            observed.setdefault(target, usd)

    vendor = _vendor_list(history)
    first_party = _first_party(models_dev)
    routed = _openrouter(openrouter)

    result: dict[str, Price] = {}
    for model in models:
        name = model["name"]
        organisation = model["org"]
        if name in observed:
            result[name] = Price(observed[name], "epoch_prices", "observed")
            continue
        candidates = keys(name)
        listed = next((vendor[key] for key in candidates if key in vendor), None)
        if listed is not None and listed[1] == "list_initial":
            result[name] = Price(listed[0], "llm_prices", "list_initial")
            continue
        table = first_party.get(organisation, {})
        match = next((table[key] for key in candidates if key in table), None)
        if match is not None:
            result[name] = Price(match, "models_dev", "list_current")
            continue
        if listed is not None:
            result[name] = Price(listed[0], "llm_prices", "list_current")
            continue
        if model["access"] == "closed":
            table = routed.get(organisation, {})
            match = next((table[key] for key in candidates if key in table), None)
            if match is not None:
                result[name] = Price(match, "openrouter", "list_current")
    return result
