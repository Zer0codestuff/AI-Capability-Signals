from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from frontier_ai.pipeline import clean_text

MATCH_CONFIDENCE_ORDER = {
    "exact": 0,
    "normalized_exact": 1,
    "alias_match": 2,
    "family_only": 3,
    "unmatched": 4,
}

NOISE_TOKENS = {
    "api",
    "beta",
    "chat",
    "experimental",
    "fast",
    "free",
    "instruct",
    "latest",
    "preview",
    "thinking",
    "turbo",
}
# Tier tokens below are deliberately NOT noise: "mini"/"pro"/"lite" name
# distinct products with different prices and benchmark results, so stripping
# them made o3 match o3-mini or GPT-4o match GPT-4o-mini at high confidence.

PROVIDER_PREFIXES = {
    "ai21",
    "alibaba",
    "amazon",
    "anthropic",
    "cohere",
    "deepseek",
    "deepseek-ai",
    "google",
    "meta",
    "meta-llama",
    "microsoft",
    "minimax",
    "mistral",
    "mistralai",
    "moonshot",
    "moonshotai",
    "nvidia",
    "openai",
    "perplexity",
    "qwen",
    "x-ai",
}

FAMILY_ALIASES = {
    "GPT": ["gpt", "chatgpt", "o1", "o3", "o4", "o5"],
    # Tier words ("opus"/"sonnet"/"haiku") must not act as standalone aliases:
    # as substrings they matched every Claude generation to each other.
    "Claude": ["claude"],
    "Gemini": ["gemini", "palm"],
    "Gemma": ["gemma"],
    "Qwen": ["qwen", "qwq", "tongyi"],
    "Llama": ["llama", "meta-llama"],
    "Mistral": ["mistral", "mixtral", "codestral", "ministral", "pixtral", "devstral"],
    "DeepSeek": ["deepseek"],
    "Grok": ["grok", "xai", "x-ai"],
    "Phi": ["phi"],
    "Command": ["command", "command-r"],
}

_VERSION_PATTERNS = {
    "GPT": [r"gpt[-_ ]?([0-9]+(?:\.[0-9]+)?)", r"(o[0-9])"],
    "Claude": [r"claude[-_ ]?(opus|sonnet|haiku)?[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Gemini": [r"gemini[-_ ]?([0-9]+(?:\.[0-9]+)?)?[-_ ]?(pro|flash)?"],
    "Gemma": [r"gemma[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Qwen": [r"qwen[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Llama": [r"llama[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Mistral": [r"(mistral|mixtral|codestral|ministral|pixtral|devstral)[-_ ]?([a-z0-9.]+)?"],
    "DeepSeek": [r"deepseek[-_ ]?([a-z0-9.]+)?"],
    "Grok": [r"grok[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Phi": [r"phi[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
    "Command": [r"command[-_ ]?r?[-_ ]?([a-z0-9.]+)?"],
}


@dataclass(frozen=True)
class ModelMatch:
    confidence: str
    benchmark_model_name: str
    benchmark_family: str
    match_key: str
    benchmark_record: dict[str, Any] | None

    @property
    def direct_model_match(self) -> bool:
        # Only full-identity matches count as direct model evidence.  Coarse
        # alias overlap ("gpt4", "claudesonnet") can bridge generations or
        # variants, so alias_match stays visible in audits but never direct.
        return self.confidence in {"exact", "normalized_exact"}


def _token_text(value: str) -> str:
    tokens = [tok for tok in re.split(r"[^a-z0-9.]+", value.lower()) if tok and tok not in NOISE_TOKENS]
    return " ".join(tokens)


@lru_cache(maxsize=65_536)
def normalize_model_name(value: Any) -> str:
    text = clean_text(value).lower()
    if ":" in text:
        text = text.split(":", 1)[1]
    if "/" in text:
        prefix, rest = text.split("/", 1)
        if prefix in PROVIDER_PREFIXES:
            text = rest
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:20\d{2})[-_ ]?(?:0[1-9]|1[0-2])[-_ ]?(?:0[1-9]|[12]\d|3[01])\b", " ", text)
    text = re.sub(r"\b20\d{6}\b", " ", text)
    text = re.sub(r"\b20\d{2}\b", " ", text)
    tokens = re.split(r"[^a-z0-9.]+", text)
    tokens = [token for token in tokens if token and token not in NOISE_TOKENS]
    collapsed = "".join(tokens)
    collapsed = collapsed.replace(".", "")
    return collapsed


@lru_cache(maxsize=65_536)
def full_name_aliases(name: Any, model_id: Any = "", family: str = "") -> frozenset[str]:
    """Aliases that represent the COMPLETE model identity.

    Includes the normalized full names plus a version-pattern expansion only
    when the pattern consumes the entire token string (e.g. "GPT-4 latest"
    expands to gpt4, but "GPT-4o mini" never collapses to gpt4).
    """
    family_alias = normalize_model_name(family)
    aliases: set[str] = set()
    for part in [clean_text(name), clean_text(model_id)]:
        if not part:
            continue
        norm = normalize_model_name(part)
        if norm and norm != family_alias:
            aliases.add(norm)
        text = _token_text(part)
        for pattern in _VERSION_PATTERNS.get(family, []):
            match = re.fullmatch(pattern, text)
            if not match:
                continue
            parts = [group for group in match.groups() if group]
            if not parts:
                continue
            alias = normalize_model_name(f"{family} {' '.join(parts)}")
            if alias and alias != family_alias:
                aliases.add(alias)
    return frozenset(aliases)


@lru_cache(maxsize=65_536)
def normalized_aliases(name: Any, model_id: Any = "", family: str = "") -> frozenset[str]:
    """Coarse aliases for ranking fallbacks only; never direct evidence."""
    raw_parts = [clean_text(name), clean_text(model_id)]
    aliases = set(full_name_aliases(name, model_id, family))
    text = " ".join(raw_parts).lower()
    family_tokens = FAMILY_ALIASES.get(family, [])
    for token in family_tokens:
        if token in text:
            aliases.add(normalize_model_name(token))
    aliases.update(_version_aliases(text, family))
    return frozenset(alias for alias in aliases if alias)


@lru_cache(maxsize=65_536)
def _version_aliases(text: str, family: str) -> frozenset[str]:
    aliases: set[str] = set()
    for pattern in _VERSION_PATTERNS.get(family, []):
        for match in re.finditer(pattern, text):
            parts = [part for part in match.groups() if part]
            if family and parts:
                aliases.add(normalize_model_name(f"{family} {' '.join(parts)}"))
            if family:
                aliases.add(normalize_model_name(family))
    return frozenset(aliases)


def _best_hit(hits: list[tuple[int, dict[str, Any], str]]) -> tuple[int, dict[str, Any], str]:
    """Pick the strongest overlapping candidate instead of the first listed.

    Ties keep the earliest candidate, so results stay deterministic.
    """

    def strength(item: tuple[int, dict[str, Any], str]) -> float:
        try:
            return -float(item[1].get("sort_score") or 0)
        except (TypeError, ValueError):
            return 0.0

    return min(hits, key=strength)


class PreparedModelMatcher:
    """Precompute candidate aliases once for repeated catalog matching."""

    def __init__(self, candidates: Iterable[dict[str, Any]]):
        self.candidates = list(candidates)
        self.records = []
        self.exact: dict[str, int] = {}
        self.by_family: dict[str, list[int]] = {}
        for index, candidate in enumerate(self.candidates):
            name = clean_text(candidate.get("model_name"))
            family = clean_text(candidate.get("family"))
            family_alias = normalize_model_name(family)
            full_aliases = full_name_aliases(name, candidate.get("model_id"), family)
            coarse = normalized_aliases(name, candidate.get("model_id"), family)
            substantive = frozenset(alias for alias in coarse if alias != family_alias)
            self.records.append((candidate, substantive, full_aliases))
            self.exact.setdefault(name.lower(), index)
            self.by_family.setdefault(family, []).append(index)

    def match(self, query_name: Any, query_id: Any, query_family: str) -> ModelMatch:
        raw_queries = {clean_text(query_name).lower(), clean_text(query_id).lower()}
        for raw in raw_queries:
            if raw in self.exact:
                candidate = self.candidates[self.exact[raw]]
                return _match("exact", candidate, raw)

        family_alias = normalize_model_name(query_family)
        query_full = {alias for alias in full_name_aliases(query_name, query_id, query_family) if alias != family_alias}
        hits = []
        for index, (candidate, _, full_aliases) in enumerate(self.records):
            overlap = query_full.intersection(full_aliases)
            if overlap:
                hits.append((index, candidate, sorted(overlap)[0]))
        if hits:
            _, candidate, key = _best_hit(hits)
            return _match("normalized_exact", candidate, key)

        query_coarse = {
            alias
            for alias in normalized_aliases(query_name, query_id, query_family)
            if alias != family_alias
        }
        family_hits = []
        for index in self.by_family.get(query_family, []):
            candidate, aliases, _ = self.records[index]
            overlap = query_coarse.intersection(aliases)
            if overlap:
                family_hits.append((index, candidate, sorted(overlap)[0]))
        if family_hits:
            _, candidate, key = _best_hit(family_hits)
            return _match("alias_match", candidate, key)

        family_indexes = self.by_family.get(query_family, [])
        if family_indexes:
            candidate = max(
                (self.candidates[index] for index in family_indexes),
                key=lambda row: float(row.get("sort_score") or 0),
            )
            return _match("family_only", candidate, query_family)
        return ModelMatch("unmatched", "", "", "", None)


def find_best_model_match(
    query_name: Any,
    query_id: Any,
    query_family: str,
    candidates: Iterable[dict[str, Any]],
) -> ModelMatch:
    candidate_list = list(candidates)
    raw_queries = {clean_text(query_name).lower(), clean_text(query_id).lower()}
    family_only_alias = normalize_model_name(query_family)

    for candidate in candidate_list:
        candidate_name = clean_text(candidate.get("model_name"))
        if candidate_name.lower() in raw_queries:
            return _match("exact", candidate, candidate_name.lower())

    query_full = {alias for alias in full_name_aliases(query_name, query_id, query_family) if alias != family_only_alias}
    hits = []
    for index, candidate in enumerate(candidate_list):
        candidate_full = full_name_aliases(
            candidate.get("model_name"),
            candidate.get("model_id"),
            clean_text(candidate.get("family")),
        )
        overlap = query_full.intersection(candidate_full)
        if overlap:
            hits.append((index, candidate, sorted(overlap)[0]))
    if hits:
        _, candidate, key = _best_hit(hits)
        return _match("normalized_exact", candidate, key)

    query_coarse = {alias for alias in normalized_aliases(query_name, query_id, query_family) if alias != family_only_alias}
    family_candidates = [candidate for candidate in candidate_list if clean_text(candidate.get("family")) == query_family]
    family_hits = []
    for index, candidate in enumerate(family_candidates):
        candidate_norms = normalized_aliases(
            candidate.get("model_name"),
            candidate.get("model_id"),
            clean_text(candidate.get("family")),
        )
        candidate_family_alias = normalize_model_name(clean_text(candidate.get("family")))
        substantive_candidate_norms = {alias for alias in candidate_norms if alias != candidate_family_alias}
        overlap = query_coarse.intersection(substantive_candidate_norms)
        if overlap:
            family_hits.append((index, candidate, sorted(overlap)[0]))
    if family_hits:
        _, candidate, key = _best_hit(family_hits)
        return _match("alias_match", candidate, key)

    if family_candidates:
        candidate = sorted(family_candidates, key=lambda row: float(row.get("sort_score") or 0), reverse=True)[0]
        return _match("family_only", candidate, query_family)

    return ModelMatch(
        confidence="unmatched",
        benchmark_model_name="",
        benchmark_family="",
        match_key="",
        benchmark_record=None,
    )


def _match(confidence: str, candidate: dict[str, Any], match_key: str) -> ModelMatch:
    return ModelMatch(
        confidence=confidence,
        benchmark_model_name=clean_text(candidate.get("model_name")),
        benchmark_family=clean_text(candidate.get("family")),
        match_key=match_key,
        benchmark_record=candidate,
    )
