from __future__ import annotations

import re
from dataclasses import dataclass
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
    "lite",
    "mini",
    "preview",
    "pro",
    "thinking",
    "turbo",
}

PROVIDER_PREFIXES = {
    "ai21",
    "alibaba",
    "anthropic",
    "deepseek",
    "google",
    "meta",
    "meta-llama",
    "microsoft",
    "mistral",
    "openai",
    "qwen",
    "x-ai",
}

FAMILY_ALIASES = {
    "GPT": ["gpt", "chatgpt", "o1", "o3", "o4", "o5"],
    "Claude": ["claude", "opus", "sonnet", "haiku"],
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


@dataclass(frozen=True)
class ModelMatch:
    confidence: str
    benchmark_model_name: str
    benchmark_family: str
    match_key: str
    benchmark_record: dict[str, Any] | None

    @property
    def direct_model_match(self) -> bool:
        return self.confidence in {"exact", "normalized_exact", "alias_match"}


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


def normalized_aliases(name: Any, model_id: Any = "", family: str = "") -> set[str]:
    raw_parts = [clean_text(name), clean_text(model_id)]
    aliases = {normalize_model_name(part) for part in raw_parts if clean_text(part)}
    text = " ".join(raw_parts).lower()
    family_tokens = FAMILY_ALIASES.get(family, [])
    for token in family_tokens:
        if token in text:
            aliases.add(normalize_model_name(token))
    aliases.update(_version_aliases(text, family))
    return {alias for alias in aliases if alias}


def _version_aliases(text: str, family: str) -> set[str]:
    aliases: set[str] = set()
    patterns = {
        "GPT": [r"\bgpt[-_ ]?([0-9]+(?:\.[0-9]+)?)", r"\b(o[0-9])\b"],
        "Claude": [r"\bclaude[-_ ]?(opus|sonnet|haiku)?[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Gemini": [r"\bgemini[-_ ]?([0-9]+(?:\.[0-9]+)?)?[-_ ]?(pro|flash)?"],
        "Gemma": [r"\bgemma[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Qwen": [r"\bqwen[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Llama": [r"\bllama[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Mistral": [r"\b(mistral|mixtral|codestral|ministral|pixtral|devstral)[-_ ]?([a-z0-9.]+)?"],
        "DeepSeek": [r"\bdeepseek[-_ ]?([a-z0-9.]+)?"],
        "Grok": [r"\bgrok[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Phi": [r"\bphi[-_ ]?([0-9]+(?:\.[0-9]+)?)?"],
        "Command": [r"\bcommand[-_ ]?r?[-_ ]?([a-z0-9.]+)?"],
    }
    for pattern in patterns.get(family, []):
        for match in re.finditer(pattern, text):
            parts = [part for part in match.groups() if part]
            if family and parts:
                aliases.add(normalize_model_name(f"{family} {' '.join(parts)}"))
            if family:
                aliases.add(normalize_model_name(family))
    return aliases


def find_best_model_match(
    query_name: Any,
    query_id: Any,
    query_family: str,
    candidates: Iterable[dict[str, Any]],
) -> ModelMatch:
    candidate_list = list(candidates)
    raw_queries = {clean_text(query_name).lower(), clean_text(query_id).lower()}
    query_norms = normalized_aliases(query_name, query_id, query_family)
    family_only_alias = normalize_model_name(query_family)
    substantive_query_norms = {alias for alias in query_norms if alias != family_only_alias}
    family_candidates = [candidate for candidate in candidate_list if clean_text(candidate.get("family")) == query_family]

    for candidate in candidate_list:
        candidate_name = clean_text(candidate.get("model_name"))
        if candidate_name.lower() in raw_queries:
            return _match("exact", candidate, candidate_name.lower())

    for candidate in candidate_list:
        candidate_norms = normalized_aliases(candidate.get("model_name"), candidate.get("model_id"), clean_text(candidate.get("family")))
        candidate_family_alias = normalize_model_name(clean_text(candidate.get("family")))
        substantive_candidate_norms = {alias for alias in candidate_norms if alias != candidate_family_alias}
        overlap = substantive_query_norms.intersection(substantive_candidate_norms)
        if overlap:
            return _match("normalized_exact", candidate, sorted(overlap)[0])

    for candidate in family_candidates:
        candidate_norms = normalized_aliases(candidate.get("model_name"), candidate.get("model_id"), clean_text(candidate.get("family")))
        candidate_family_alias = normalize_model_name(clean_text(candidate.get("family")))
        substantive_candidate_norms = {alias for alias in candidate_norms if alias != candidate_family_alias}
        overlap = substantive_query_norms.intersection(substantive_candidate_norms)
        if overlap:
            return _match("alias_match", candidate, sorted(overlap)[0])

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
