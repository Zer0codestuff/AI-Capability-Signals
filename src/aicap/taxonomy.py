"""Vendor, family and weight-availability classification.

Principle: prefer an authoritative source field over an inferred one, and when inference is
unavoidable, make it a table of explicit rules with a tested outcome and an ``unknown`` escape
hatch.

The previous version inferred weight availability from substring hints on model names
(``OPEN_WEIGHT_HINTS``), inventing a ``likely_open_weight`` class, even though Epoch AI publishes
``Model accessibility`` and ``Open model weights?`` directly and OpenRouter publishes
``hugging_face_id``. Those fields are used here instead, and the inference path exists only to
group models into product families, which no source publishes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Weight-availability classes. ``unknown`` is a real answer and is never silently converted into
#: one of the others.
WEIGHTS_OPEN = "open_weights"
WEIGHTS_RESTRICTED = "open_weights_restricted"
WEIGHTS_CLOSED = "closed_weights"
WEIGHTS_UNRELEASED = "unreleased"
WEIGHTS_UNKNOWN = "unknown"

#: Maps Epoch AI's ``Model accessibility`` values onto this project's classes. Any value not in
#: this table becomes ``unknown`` and is counted in the data-quality report, so an upstream
#: vocabulary change is visible instead of being absorbed.
EPOCH_ACCESSIBILITY_MAP = {
    "open weights (unrestricted)": WEIGHTS_OPEN,
    "open weights (restricted use)": WEIGHTS_RESTRICTED,
    "open weights (non-commercial)": WEIGHTS_RESTRICTED,
    "api access": WEIGHTS_CLOSED,
    "hosted access (no api)": WEIGHTS_CLOSED,
    "unreleased": WEIGHTS_UNRELEASED,
}

#: Ordered vendor rules. First match wins; order matters where names overlap (a Microsoft-hosted
#: OpenAI model is attributed to OpenAI, because the interesting unit is who trained it).
VENDOR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("OpenAI", (r"\bopenai\b", r"\bgpt-?\d", r"\bo[1-4]\b", r"\bcodex\b", r"\bsora\b")),
    ("Anthropic", (r"\banthropic\b", r"\bclaude\b")),
    ("Google", (r"\bgoogle\b", r"\bdeepmind\b", r"\bgemini\b", r"\bgemma\b", r"\bpalm\b")),
    ("Meta", (r"\bmeta\b", r"\bfacebook\b", r"\bllama\b", r"\bmuse\b")),
    ("xAI", (r"\bx-?ai\b", r"\bgrok\b")),
    ("Alibaba", (r"\balibaba\b", r"\bqwen\b", r"\btongyi\b")),
    ("DeepSeek", (r"\bdeepseek\b",)),
    ("Mistral", (r"\bmistral\b", r"\bmixtral\b", r"\bmagistral\b", r"\bdevstral\b")),
    ("Moonshot AI", (r"\bmoonshot\b", r"\bkimi\b")),
    ("Z.ai", (r"\bz-?ai\b", r"\bzhipu\b", r"\bglm\b", r"\bchatglm\b")),
    ("Microsoft", (r"\bmicrosoft\b", r"\bphi-?\d", r"\bmai-\b")),
    ("Cohere", (r"\bcohere\b", r"\bcommand-?r?\b", r"\baya\b", r"\bnorth\b")),
    ("Amazon", (r"\bamazon\b", r"\bnova\b", r"\btitan\b")),
    ("NVIDIA", (r"\bnvidia\b", r"\bnemotron\b")),
    ("AI21", (r"\bai21\b", r"\bjamba\b", r"\bjurassic\b")),
    ("Baidu", (r"\bbaidu\b", r"\bernie\b")),
    ("ByteDance", (r"\bbytedance\b", r"\bdoubao\b", r"\bseed-?\b")),
    ("Tencent", (r"\btencent\b", r"\bhunyuan\b")),
    ("Allen Institute for AI", (r"\ballenai\b", r"\bai2\b", r"\bolmo\b", r"\btulu\b")),
    ("Perplexity", (r"\bperplexity\b", r"\bsonar\b")),
    ("Reka", (r"\breka\b",)),
    ("Inflection", (r"\binflection\b",)),
    ("Stability AI", (r"\bstability\b", r"\bstable[- ]?(diffusion|lm)\b")),
    ("Thinking Machines", (r"\bthinkingmachines\b", r"\bthinking machines\b", r"\binkling\b")),
    ("Nous Research", (r"\bnous\b", r"\bhermes\b")),
    ("01.AI", (r"\b01[.-]?ai\b", r"\byi-?\d")),
    ("MiniMax", (r"\bminimax\b", r"\babab\b")),
    ("Liquid AI", (r"\bliquid\b", r"\blfm\b")),
    ("Upstage", (r"\bupstage\b", r"\bsolar\b")),
    ("IBM", (r"\bibm\b", r"\bgranite\b")),
    ("Databricks", (r"\bdatabricks\b", r"\bdbrx\b")),
    ("Snowflake", (r"\bsnowflake\b", r"\barctic\b")),
    ("EleutherAI", (r"\beleuther\b", r"\bpythia\b", r"\bgpt-neo\b", r"\bgpt-j\b")),
    ("TII", (r"\btii\b", r"\bfalcon\b")),
    ("LG AI Research", (r"\bexaone\b", r"\blg ai\b")),
    ("Naver", (r"\bnaver\b", r"\bhyperclova\b")),
    ("Sakana AI", (r"\bsakana\b",)),
    ("Ai2 / Others", ()),
)

#: Product-family rules. A "family" is a marketing line, which no source publishes as a field,
#: so this is inference by construction. It is used only for grouping and never as evidence.
FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Claude", (r"\bclaude\b",)),
    ("Gemini", (r"\bgemini\b",)),
    ("Gemma", (r"\bgemma\b",)),
    ("GPT", (r"\bgpt-?\d", r"\bgpt-?oss\b", r"\bo[1-4]\b", r"\bcodex\b", r"\bchatgpt\b")),
    ("Llama", (r"\bllama\b",)),
    ("Muse", (r"\bmuse\b",)),
    ("DeepSeek", (r"\bdeepseek\b",)),
    ("Qwen", (r"\bqwen\b", r"\bqwq\b", r"\bqvq\b")),
    ("Mistral", (r"\bmistral\b", r"\bmixtral\b", r"\bmagistral\b", r"\bdevstral\b", r"\bcodestral\b")),
    ("Grok", (r"\bgrok\b",)),
    ("Kimi", (r"\bkimi\b",)),
    ("GLM", (r"\bglm\b", r"\bchatglm\b")),
    ("Phi", (r"\bphi-?\d",)),
    ("Command", (r"\bcommand\b", r"\bnorth\b")),
    ("Nova", (r"\bnova\b",)),
    ("Nemotron", (r"\bnemotron\b",)),
    ("OLMo", (r"\bolmo\b",)),
    ("Ernie", (r"\bernie\b",)),
    ("Hunyuan", (r"\bhunyuan\b",)),
    ("Doubao", (r"\bdoubao\b",)),
    ("MiniMax", (r"\bminimax\b", r"\babab\b")),
    ("Jamba", (r"\bjamba\b",)),
    ("Granite", (r"\bgranite\b",)),
    ("Falcon", (r"\bfalcon\b",)),
    ("Yi", (r"\byi-?\d",)),
    ("Sonar", (r"\bsonar\b",)),
    ("EXAONE", (r"\bexaone\b",)),
    ("LFM", (r"\blfm\b",)),
    ("Inkling", (r"\binkling\b",)),
    ("Hermes", (r"\bhermes\b",)),
)

_COMPILED_VENDORS = tuple(
    (vendor, tuple(re.compile(pattern) for pattern in patterns)) for vendor, patterns in VENDOR_RULES
)
_COMPILED_FAMILIES = tuple(
    (family, tuple(re.compile(pattern) for pattern in patterns)) for family, patterns in FAMILY_RULES
)


@dataclass(frozen=True)
class Classification:
    vendor: str
    family: str


def _haystack(*parts: object) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    # Treat separators as spaces so word-boundary patterns match inside slugs like
    # "meta-llama/Llama-3.1-70b".
    return " " + re.sub(r"[/_.:]+", " ", text) + " "


def classify_vendor(name: str, organization: str = "") -> str:
    """Return the training organisation, or ``"Unknown"``.

    ``organization`` is consulted first because it is a source field; the model name is a
    fallback for catalogues that only expose a slug.
    """
    for haystack in (_haystack(organization), _haystack(name, organization)):
        for vendor, patterns in _COMPILED_VENDORS:
            if patterns and any(pattern.search(haystack) for pattern in patterns):
                return vendor
    cleaned = str(organization or "").strip()
    if cleaned and "," not in cleaned:
        return cleaned
    return "Unknown"


def classify_family(name: str, organization: str = "") -> str:
    """Return a product-family label, or ``"Other"`` when no rule matches."""
    haystack = _haystack(name, organization)
    for family, patterns in _COMPILED_FAMILIES:
        if any(pattern.search(haystack) for pattern in patterns):
            return family
    return "Other"


def classify(name: str, organization: str = "") -> Classification:
    return Classification(classify_vendor(name, organization), classify_family(name, organization))


def epoch_weights_class(accessibility: object, open_weights_flag: object) -> str:
    """Map Epoch AI accessibility fields onto a weight-availability class.

    The two fields are cross-checked: ``Open model weights? = Yes`` with an accessibility value
    that maps to closed is a source inconsistency, and is reported as ``unknown`` rather than
    resolved silently in favour of either field.
    """
    raw = str(accessibility or "").strip().lower()
    mapped = EPOCH_ACCESSIBILITY_MAP.get(raw, WEIGHTS_UNKNOWN)
    flag = str(open_weights_flag or "").strip().lower()

    if mapped == WEIGHTS_UNKNOWN:
        if flag == "yes":
            return WEIGHTS_OPEN
        if flag == "no":
            return WEIGHTS_CLOSED
        return WEIGHTS_UNKNOWN

    open_by_flag = flag == "yes"
    open_by_class = mapped in {WEIGHTS_OPEN, WEIGHTS_RESTRICTED}
    if flag in {"yes", "no"} and open_by_flag != open_by_class:
        return WEIGHTS_UNKNOWN
    return mapped


#: String forms that pandas and JSON both produce for "no value". Checked explicitly because
#: ``str(float("nan"))`` is the truthy string ``"nan"``, and treating that as a real identifier
#: would have classified every closed model as open-weight.
_MISSING_STRINGS = frozenset({"", "nan", "none", "null", "na", "<na>", "-"})


def _present(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _MISSING_STRINGS else text


def openrouter_weights_class(hugging_face_id: object, licence_hint: object = "") -> str:
    """Classify a listed API model by whether the vendor points at published weights.

    ``hugging_face_id`` is a vendor-supplied pointer to a weights repository. Its presence is
    strong evidence that weights are published; its absence is weaker evidence of the converse,
    so the negative case is labelled ``closed_weights`` only because every model in this
    catalogue is, by construction, available via API.
    """
    hf = _present(hugging_face_id)
    if hf:
        licence = _present(licence_hint).lower()
        if any(token in licence for token in ("non-commercial", "research", "restricted")):
            return WEIGHTS_RESTRICTED
        return WEIGHTS_OPEN
    return WEIGHTS_CLOSED


def lmarena_weights_class(licence: object) -> str:
    """Classify an arena row from its published licence string.

    LMArena reports a licence per model. ``Proprietary`` means closed; a named open licence means
    published weights; anything unrecognised stays ``unknown`` instead of being guessed from the
    model name.
    """
    text = _present(licence).lower()
    if not text or text == "unknown":
        return WEIGHTS_UNKNOWN
    if "proprietary" in text:
        return WEIGHTS_CLOSED
    restricted_tokens = ("non-commercial", "noncommercial", "research", "nc", "cc-by-nc")
    open_tokens = (
        "apache",
        "mit",
        "bsd",
        "openrail",
        "gemma",
        "llama",
        "qwen",
        "deepseek",
        "mistral ai research",
        "cc-by",
        "gpl",
        "falcon",
        "tongyi",
        "modified mit",
        "open",
    )
    if any(token in text for token in restricted_tokens):
        return WEIGHTS_RESTRICTED
    if any(token in text for token in open_tokens):
        return WEIGHTS_OPEN
    return WEIGHTS_UNKNOWN


def is_weights_available(weights_class: str) -> bool | None:
    """Collapse to a binary for open-vs-closed comparisons, preserving ``None`` for unknown.

    Returning ``None`` rather than ``False`` for unknowns is what keeps unclassified models out
    of the open/closed gap estimates instead of silently padding the closed side.
    """
    if weights_class in {WEIGHTS_OPEN, WEIGHTS_RESTRICTED}:
        return True
    if weights_class == WEIGHTS_CLOSED:
        return False
    return None
