"""Cross-source model identity resolution.

Why this module is small and strict
-----------------------------------
The previous version matched models across sources by normalising names, accepted the first
alias overlap in iteration order, and — when nothing matched — fell back to *the highest-scoring
model in the same family*. That fallback silently attached a flagship's benchmark score to cheap
small models, and it produced more "matches" (813) than exact ones (504).

The rebuilt matcher differs in three ways:

1. **Version tokens are preserved.** ``gpt-5.4`` and ``gpt-5.5`` must never collide, so
   normalisation strips punctuation and marketing noise but keeps every digit.
2. **There is no family fallback.** ``unmatched`` is a valid, published outcome.
3. **Tiers are ranked and consumers choose a floor.** Price-versus-quality joins accept only
   :data:`TIER_EXACT` and :data:`TIER_ALIAS`; configuration variants are recorded but excluded,
   because a reasoning-effort variant is a different system under test.

The matcher's precision is measured, not asserted: :func:`evaluate` scores it against the
labelled pairs in :data:`GOLD_PAIRS` and the result is published as a diagnostic table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Identical canonical keys.
TIER_EXACT = "exact"
#: Linked through a curated alias in :data:`ALIASES`.
TIER_ALIAS = "alias"
#: Same base model, different serving configuration (reasoning effort, latency tier, quantisation).
#: Recorded for auditing, excluded from published model-level claims.
TIER_CONFIGURATION_VARIANT = "configuration_variant"
#: No acceptable correspondence found.
TIER_UNMATCHED = "unmatched"

#: Tiers acceptable when a claim is about a specific model.
STRICT_TIERS = frozenset({TIER_EXACT, TIER_ALIAS})

TIER_RANK = {
    TIER_EXACT: 0,
    TIER_ALIAS: 1,
    TIER_CONFIGURATION_VARIANT: 2,
    TIER_UNMATCHED: 3,
}

#: Vendor prefixes stripped from catalogue slugs before comparison.
_VENDOR_PREFIXES = re.compile(
    r"^(openai|anthropic|google|meta|meta-llama|x-ai|xai|mistralai|mistral|qwen|alibaba|deepseek|"
    r"moonshotai|moonshot|z-ai|zai-org|microsoft|cohere|coherelabs|amazon|nvidia|ai21|baidu|"
    r"bytedance|tencent|allenai|perplexity|reka|inflection|stabilityai|thinkingmachines|"
    r"nousresearch|nous|01-ai|minimax|liquid|upstage|ibm-granite|ibm|databricks|snowflake|"
    r"eleutherai|tiiuae|lg-ai|lgai|naver|sakanaai|arcee-ai|nex|aion-labs|inclusionai)[/:]"
)

#: Tokens that carry no identity information. Dropping them lets ``GPT-5.5 (preview)`` meet
#: ``gpt-5.5-preview``. Reasoning-effort and latency markers are *not* here: they are handled as
#: configuration variants, because they change the system under test.
_NOISE_TOKENS = frozenset(
    {
        "ai",
        "beta",
        "chat",
        "experimental",
        "instruct",
        "it",
        "latest",
        "model",
        "preview",
        "release",
        "stable",
        "text",
        "v1",
        "version",
    }
)

#: Suffixes that denote a *serving configuration* of one underlying model: reasoning effort,
#: latency tier, quantisation, context length.
#:
#: Size and tier words (``mini``, ``nano``, ``flash``, ``lite``, ``pro``, ``max``, ``medium``) are
#: deliberately absent. ``gemini-3.1-flash-lite`` and ``gemini-3.1-pro`` are different models with
#: different weights, not two settings of one model, so those words are part of a model's identity.
#: Treating them as configuration would have re-created the previous version's central error of
#: attaching a flagship's score to a cheap sibling. Ambiguous words are resolved toward identity,
#: which loses matches rather than inventing them.
_CONFIGURATION_SUFFIXES = (
    "high",
    "low",
    "minimal",
    "fast",
    "thinking",
    "reasoning",
    "nothinking",
    "extended",
    "turbo",
    "online",
    "free",
    "exp",
    "fp8",
    "bf16",
    "int4",
    "awq",
    "gptq",
    "8k",
    "16k",
    "32k",
    "64k",
    "128k",
    "200k",
)

#: Hand-curated equivalences that normalisation cannot derive. Keys and values are canonical keys.
#: Every entry needs a reason, because an undocumented alias is indistinguishable from a bug.
ALIASES: dict[str, str] = {
    # Arena publishes date-stamped build names for some models; the catalogue lists the family slug.
    "chatgpt4olatest": "gpt4o",
    "gpt4turbo20240409": "gpt4turbo",
    # Vendor rebrands where both names refer to one released model.
    "gpt4omini": "gpt4omini",
}


@dataclass(frozen=True)
class Match:
    """Outcome of matching one query model against a candidate pool."""

    tier: str
    candidate_id: str | None
    candidate_name: str | None
    #: The canonical key that produced the match, for auditing.
    matched_on: str | None

    @property
    def is_strict(self) -> bool:
        return self.tier in STRICT_TIERS


def canonical_key(name: object) -> str:
    """Reduce a model name to a comparison key that preserves version identity.

    Steps: lowercase, strip a vendor prefix, split on non-alphanumerics, drop marketing noise,
    then rejoin. Digits are always kept, and a token containing digits is never treated as noise,
    which is what prevents ``gpt-5.4`` and ``gpt-5.5`` from colliding.
    """
    text = str(name or "").strip().lower()
    if not text:
        return ""
    text = text.replace(" ", "-")
    text = _VENDOR_PREFIXES.sub("", text)
    # A leading "vendor:" display prefix, as in "OpenAI: GPT-5.5".
    text = re.sub(r"^[a-z0-9-]+:\s*", "", text)
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    kept = [token for token in tokens if any(char.isdigit() for char in token) or token not in _NOISE_TOKENS]
    return "".join(kept)


def split_configuration(key: str) -> tuple[str, tuple[str, ...]]:
    """Split a canonical key into its base and any trailing configuration markers.

    Only *trailing* markers are stripped, and never all of them: a key that is entirely
    configuration tokens (``"mini"``) keeps its last token, so distinct products are not collapsed
    into an empty base.
    """
    if not key:
        return "", ()
    tokens = _tokenize(key)
    suffixes: list[str] = []
    while len(tokens) > 1 and tokens[-1] in _CONFIGURATION_SUFFIXES:
        suffixes.insert(0, tokens.pop())
    return "".join(tokens), tuple(suffixes)


def _tokenize(key: str) -> list[str]:
    """Recover token boundaries from a joined canonical key.

    ``canonical_key`` joins tokens without separators, so configuration suffixes are detected by
    matching known markers against the end of the string, longest first.
    """
    remaining = key
    tail: list[str] = []
    changed = True
    while changed:
        changed = False
        for suffix in sorted(_CONFIGURATION_SUFFIXES, key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                tail.insert(0, suffix)
                remaining = remaining[: -len(suffix)]
                changed = True
                break
    return [remaining, *tail] if remaining else tail


@dataclass
class Candidate:
    """A model in the target pool."""

    candidate_id: str
    display_name: str
    key: str
    base: str
    suffixes: tuple[str, ...]

    @classmethod
    def create(cls, candidate_id: str, display_name: str) -> Candidate:
        key = canonical_key(display_name or candidate_id)
        base, suffixes = split_configuration(key)
        return cls(candidate_id, display_name, key, base, suffixes)


class Matcher:
    """Matches model names against a fixed candidate pool.

    The pool is indexed once, so matching a few thousand names is linear rather than quadratic.
    Ambiguity is resolved deterministically and conservatively: if two candidates tie at a tier,
    the match is rejected as ambiguous rather than resolved by iteration order.
    """

    def __init__(self, candidates: Iterable[tuple[str, str]]) -> None:
        self.candidates: list[Candidate] = [
            Candidate.create(candidate_id, display_name) for candidate_id, display_name in candidates
        ]
        self._by_key: dict[str, list[Candidate]] = {}
        self._by_base: dict[str, list[Candidate]] = {}
        for candidate in self.candidates:
            if candidate.key:
                self._by_key.setdefault(candidate.key, []).append(candidate)
            if candidate.base:
                self._by_base.setdefault(candidate.base, []).append(candidate)

    def match(self, name: object) -> Match:
        key = canonical_key(name)
        if not key:
            return Match(TIER_UNMATCHED, None, None, None)

        hit = self._unique(self._by_key.get(key, []))
        if hit is not None:
            return Match(TIER_EXACT, hit.candidate_id, hit.display_name, key)

        aliased = ALIASES.get(key)
        if aliased and aliased != key:
            hit = self._unique(self._by_key.get(aliased, []))
            if hit is not None:
                return Match(TIER_ALIAS, hit.candidate_id, hit.display_name, aliased)

        base, _ = split_configuration(key)
        if base and base != key:
            # The query carries configuration markers the candidate lacks (arena's "gpt-5.4-high"
            # against the catalogue's "gpt-5.4"): same weights, different serving configuration.
            hit = self._unique(self._by_key.get(base, []))
            if hit is not None:
                return Match(TIER_CONFIGURATION_VARIANT, hit.candidate_id, hit.display_name, base)
        if base:
            # The candidate carries markers the query lacks.
            hit = self._unique([c for c in self._by_base.get(base, []) if c.key != key])
            if hit is not None:
                return Match(TIER_CONFIGURATION_VARIANT, hit.candidate_id, hit.display_name, base)

        return Match(TIER_UNMATCHED, None, None, None)

    @staticmethod
    def _unique(hits: Sequence[Candidate]) -> Candidate | None:
        """Return the single candidate, or ``None`` when absent or ambiguous."""
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Deduplicate identical ids (a catalogue can list one model twice) before rejecting.
            unique_ids = {hit.candidate_id for hit in hits}
            if len(unique_ids) == 1:
                return hits[0]
        return None


#: Labelled pairs used to measure matcher precision and recall.
#:
#: ``expected_id`` is ``None`` where the correct answer is "no match" — these negative cases are
#: what catch a matcher that has become too permissive. Names are chosen to cover the traps that
#: broke the previous version: adjacent versions, reasoning-effort suffixes, date stamps, and
#: two vendors using the same product word.
GOLD_PAIRS: tuple[tuple[str, str | None], ...] = (
    ("gpt-5.5", "openai/gpt-5.5"),
    ("GPT-5.5", "openai/gpt-5.5"),
    ("OpenAI: GPT-5.5", "openai/gpt-5.5"),
    ("gpt-5.5-preview", "openai/gpt-5.5"),
    ("gpt-5.4", "openai/gpt-5.4"),
    ("claude-opus-4.7", "anthropic/claude-opus-4.7"),
    ("Anthropic: Claude Opus 4.7", "anthropic/claude-opus-4.7"),
    ("claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
    ("gemini-3-pro", "google/gemini-3-pro"),
    ("qwen3-235b-a22b-instruct", "qwen/qwen3-235b-a22b"),
    # Negative cases: no candidate exists, so the only correct answer is unmatched.
    ("gpt-5.9", None),
    ("claude-opus-9", None),
    ("some-unreleased-internal-model", None),
    # Negative cases that a size-collapsing matcher would get wrong. A sibling of a different
    # size is not the model that was asked for, and answering with it is the failure this
    # matcher exists to prevent.
    ("gpt-5.4-nano", None),
    ("gemini-3-flash", None),
)

#: Candidate pool for :data:`GOLD_PAIRS`. Contains near-miss distractors on purpose.
GOLD_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("openai/gpt-5.5", "OpenAI: GPT-5.5"),
    ("openai/gpt-5.5-pro", "OpenAI: GPT-5.5 Pro"),
    ("openai/gpt-5.4", "OpenAI: GPT-5.4"),
    ("openai/gpt-5.4-mini", "OpenAI: GPT-5.4 Mini"),
    ("anthropic/claude-opus-4.7", "Anthropic: Claude Opus 4.7"),
    ("anthropic/claude-opus-4.7-fast", "Anthropic: Claude Opus 4.7 (Fast)"),
    ("anthropic/claude-opus-4.6", "Anthropic: Claude Opus 4.6"),
    ("anthropic/claude-sonnet-4.5", "Anthropic: Claude Sonnet 4.5"),
    ("google/gemini-3-pro", "Google: Gemini 3 Pro"),
    ("qwen/qwen3-235b-a22b", "Qwen: Qwen3 235B A22B"),
)


@dataclass(frozen=True)
class MatcherEvaluation:
    """Measured matcher quality on labelled pairs."""

    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    correct_rejections: int

    @property
    def precision(self) -> float:
        attempted = self.true_positives + self.false_positives
        return self.true_positives / attempted if attempted else float("nan")

    @property
    def recall(self) -> float:
        return self.true_positives / self.positives if self.positives else float("nan")

    @property
    def specificity(self) -> float:
        return self.correct_rejections / self.negatives if self.negatives else float("nan")


def evaluate(
    pairs: Sequence[tuple[str, str | None]] = GOLD_PAIRS,
    candidates: Sequence[tuple[str, str]] = GOLD_CANDIDATES,
    accepted_tiers: frozenset[str] = STRICT_TIERS,
) -> MatcherEvaluation:
    """Score the matcher against labelled pairs at a given tier floor."""
    matcher = Matcher(candidates)
    positives = sum(1 for _, expected in pairs if expected is not None)
    negatives = len(pairs) - positives
    tp = fp = fn = rejections = 0
    for query, expected in pairs:
        result = matcher.match(query)
        accepted = result.tier in accepted_tiers
        predicted = result.candidate_id if accepted else None
        if expected is None:
            if predicted is None:
                rejections += 1
            else:
                fp += 1
        elif predicted == expected:
            tp += 1
        elif predicted is None:
            fn += 1
        else:
            fp += 1
    return MatcherEvaluation(positives, negatives, tp, fp, fn, rejections)
