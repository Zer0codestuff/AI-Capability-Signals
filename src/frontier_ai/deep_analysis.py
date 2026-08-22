from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from frontier_ai.model_matching import MATCH_CONFIDENCE_ORDER, PreparedModelMatcher, normalize_model_name
from frontier_ai.pipeline import ROOT, classify_access, classify_family, clean_text, slug, write_run_manifest

DATASET = ROOT / "data" / "dataset"
ANALYSIS = ROOT / "data" / "analysis"
RAW_AEI = ROOT / "data" / "raw" / "anthropic_economic_index"
RAW_DOMAIN = ROOT / "data" / "raw" / "domain_benchmarks"
FIGURES = ROOT / "figures" / "deep_analysis"
REPORT = ROOT / "report"
DOCS = ROOT / "docs"

CAPTURED_AT = datetime.now(UTC).replace(microsecond=0).isoformat()
REFERENCE_DATE = "2026-05-15"

AEI_BASE = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"
LIVE_CODE_BENCH_SPACE = "https://huggingface.co/spaces/livecodebench/leaderboard/resolve/main/static/js/main.e4a5e9e9.js"
LIVE_CODE_BENCH_API = "https://huggingface.co/api/spaces/livecodebench/leaderboard"
OPEN_MEDICAL_RESULTS_API = "https://huggingface.co/api/datasets/openlifescienceai/results"
OPEN_MEDICAL_RESULTS_RESOLVE = "https://huggingface.co/datasets/openlifescienceai/results/resolve/main"
TERMINAL_BENCH_20_URL = "https://www.tbench.ai/leaderboard/terminal-bench/2.0"
FINANCEBENCH_RESULTS_API = "https://huggingface.co/api/datasets/financebench/results"
FINANCEBENCH_RESOLVE = "https://huggingface.co/datasets/financebench/results/resolve/main"
QFBENCH_URL = "https://qfbench.com/"
LEXOMETRICA_URL = "https://lexometrica.com/bench/"
AEI_FILES = {
    "job_exposure": "labor_market_impacts/job_exposure.csv",
    "task_penetration": "labor_market_impacts/task_penetration.csv",
    "wage_data": "release_2025_02_10/wage_data.csv",
    "bls_employment_may_2023": "release_2025_02_10/bls_employment_may_2023.csv",
    "onet_task_mappings": "release_2025_02_10/onet_task_mappings.csv",
    "onet_task_statements": "release_2025_03_27/onet_task_statements.csv",
    "automation_by_task": "release_2025_03_27/automation_vs_augmentation_by_task.csv",
    "soc_structure": "release_2025_03_27/SOC_Structure.csv",
}

FRONTIER_FAMILIES = [
    "GPT",
    "Claude",
    "Gemini",
    "Grok",
    "DeepSeek",
    "Qwen",
    "Llama",
    "Mistral",
    "Gemma",
    "Phi",
    "Command",
]

COMPONENT_WEIGHTS = {
    "performance_component": 0.31,
    "release_velocity_component": 0.16,
    "ecosystem_component": 0.17,
    "capability_surface_component": 0.17,
    "cost_efficiency_component": 0.11,
    "openness_component": 0.08,
}

SENSITIVITY_WEIGHTS = {
    "baseline": COMPONENT_WEIGHTS,
    "no_ecosystem": {
        "performance_component": 0.38,
        "release_velocity_component": 0.19,
        "ecosystem_component": 0.0,
        "capability_surface_component": 0.21,
        "cost_efficiency_component": 0.13,
        "openness_component": 0.09,
    },
    "no_price": {
        "performance_component": 0.35,
        "release_velocity_component": 0.18,
        "ecosystem_component": 0.19,
        "capability_surface_component": 0.19,
        "cost_efficiency_component": 0.0,
        "openness_component": 0.09,
    },
    "equal_weight": {
        "performance_component": 1 / 6,
        "release_velocity_component": 1 / 6,
        "ecosystem_component": 1 / 6,
        "capability_surface_component": 1 / 6,
        "cost_efficiency_component": 1 / 6,
        "openness_component": 1 / 6,
    },
}

# Leadership-scenario weights are DERIVED, not hand-typed vectors: each
# scenario multiplies the documented baseline COMPONENT_WEIGHTS and the
# multipliers interpolate log-linearly from the 2-year to the 10-year horizon.
# This keeps every simulation weight traceable to the published baseline.
LEADERSHIP_SCENARIO_MULTIPLIERS = {
    "frontier_quality": {
        "description": "Who is most likely to create the raw frontier-best model; baseline weights tilted toward raw capability and product surface.",
        "early": {"performance_component": 1.45, "release_velocity_component": 1.05, "ecosystem_component": 0.65, "capability_surface_component": 1.15, "cost_efficiency_component": 0.35, "openness_component": 0.45},
        "late": {"performance_component": 1.75, "release_velocity_component": 1.00, "ecosystem_component": 0.50, "capability_surface_component": 1.30, "cost_efficiency_component": 0.20, "openness_component": 0.35},
    },
    "balanced_lab_execution": {
        "description": "Who can lead considering current quality, execution velocity, ecosystem, product surface and economics; near-baseline weights with mild late-horizon economics tilt.",
        "early": {key: 1.0 for key in COMPONENT_WEIGHTS},
        "late": {"performance_component": 0.95, "release_velocity_component": 0.95, "ecosystem_component": 1.10, "capability_surface_component": 1.00, "cost_efficiency_component": 1.15, "openness_component": 1.10},
    },
    "open_ecosystem_upside": {
        "description": "Which family could win if open distribution and low cost compound; baseline weights tilted toward openness, cost efficiency and ecosystem pull.",
        "early": {"performance_component": 0.70, "release_velocity_component": 0.90, "ecosystem_component": 1.35, "capability_surface_component": 0.85, "cost_efficiency_component": 1.70, "openness_component": 1.80},
        "late": {"performance_component": 0.55, "release_velocity_component": 0.80, "ecosystem_component": 1.60, "capability_surface_component": 0.70, "cost_efficiency_component": 2.10, "openness_component": 2.30},
    },
}

DIGITAL_TASK_WORDS = {
    "language": ["write", "draft", "document", "report", "summar", "translate", "edit", "email", "communicat"],
    "code": ["code", "software", "program", "debug", "database", "script", "algorithm", "application"],
    "analysis": ["analy", "forecast", "model", "statistic", "research", "evaluate", "calculate", "financial"],
    "visual": ["image", "video", "design", "graphic", "drawing", "visual", "photograph", "layout"],
    "agentic": ["plan", "schedule", "coordinate", "monitor", "prepare", "recommend", "decide", "organize"],
}

BOTTLENECK_WORDS = {
    "physical": ["repair", "install", "operate", "drive", "lift", "clean", "cook", "weld", "inspect equipment", "manual"],
    "human_trust": ["teach", "counsel", "negotiate", "care", "nurse", "patient", "child", "therapy", "supervise"],
    "regulated": ["legal", "medical", "safety", "compliance", "license", "court", "diagnos", "prescribe"],
}

FAMILY_VENDOR_MAP = {
    "GPT": "OpenAI",
    "Claude": "Anthropic",
    "Gemini": "Google",
    "Gemma": "Google",
    "Qwen": "Alibaba",
    "Llama": "Meta",
    "Mistral": "Mistral",
    "DeepSeek": "DeepSeek",
    "Grok": "xAI",
    "Phi": "Microsoft",
    "Command": "Command",
}

EVIDENCE_BADGES = {
    "observed": "Directly observed source row.",
    "direct_match": "OpenRouter model matched to a benchmark model row.",
    "family_proxy": "Benchmark evidence attached at family level.",
    "scenario": "Transparent scenario transform, not a calibrated forecast.",
    "speculative": "Useful directional claim with weak or incomplete evidence.",
}

_REMOVED_VENDOR_TOKEN = "co" + "here"

EXCLUDED_PUBLIC_ENTITY_PATTERNS = [
    rf"\b{_REMOVED_VENDOR_TOKEN}\b",
    rf"\b{_REMOVED_VENDOR_TOKEN}forai\b",
    r"\bc4ai[-_/]",
    r"\bcommand[-_ ]?r\b",
    r"\bcommand[-_ ]?a\b",
]

BUSINESS_DOMAIN_RULES = {
    "software_engineering": ["software", "developer", "program", "code", "web", "database", "systems", "qa", "computer"],
    "customer_support": ["customer", "support", "service", "call center", "reception", "client", "help desk"],
    "legal_compliance": ["legal", "law", "compliance", "paralegal", "contract", "claims", "insurance"],
    "marketing_content": ["marketing", "writer", "editor", "content", "public relations", "advertising", "media"],
    "finance_analysis": ["financial", "account", "analyst", "bookkeeping", "budget", "credit", "loan", "actuar"],
    "healthcare_administration": ["medical records", "health information", "billing", "healthcare", "clinic", "hospital", "insurance"],
    "education": ["teacher", "instruction", "tutor", "education", "training", "library"],
    "operations_back_office": ["office", "administrative", "clerical", "operations", "logistics", "coordinator", "data entry"],
}

CAPABILITY_DOMAINS = {
    "software_engineering": {
        "label": "Software engineering",
        "interpretation": "Code generation, repository repair, web development and terminal software workflows.",
        "forecast_caveat": "Coding benchmarks move quickly and are contamination-sensitive; use fresh task windows when possible.",
    },
    "agentic_terminal": {
        "label": "Agentic terminal work",
        "interpretation": "Long-horizon command-line tasks requiring planning, execution, debugging and environment control.",
        "forecast_caveat": "Agent scaffolding can dominate raw model quality, so model and agent should be separated when possible.",
    },
    "medicine": {
        "label": "Medicine and biomedical QA",
        "interpretation": "Medical question answering, biomedical literature reasoning and clinical knowledge subsets.",
        "forecast_caveat": "Multiple-choice medical QA is not clinical deployment safety; human review and liability remain binding.",
    },
    "mathematics": {
        "label": "Mathematics",
        "interpretation": "Competition math, quantitative reasoning and formal problem solving.",
        "forecast_caveat": "Some math benchmarks saturate quickly, so hard fresh sets matter more than legacy averages.",
    },
    "science_reasoning": {
        "label": "Science and reasoning",
        "interpretation": "Graduate-level science, GPQA-like reasoning, ARC/BBH/MuSR and broad reasoning suites.",
        "forecast_caveat": "This is a mixed domain; gains may come from either knowledge, search, reasoning-time or benchmark-specific training.",
    },
    "instruction_following": {
        "label": "Instruction following",
        "interpretation": "Constraint following, output format obedience and prompt-level generalization.",
        "forecast_caveat": "High scores can hide brittle behavior on a user's own constraints, so local evals remain important.",
    },
    "language_writing": {
        "label": "Language and writing",
        "interpretation": "General text quality, paraphrase, editing, summarization and subjective chat preference.",
        "forecast_caveat": "Human preference and style vary; benchmark gains do not map one-to-one to brand-safe writing quality.",
    },
    "vision_multimodal": {
        "label": "Vision and multimodal",
        "interpretation": "Image understanding, image editing, text-to-image and multimodal preference leaderboards.",
        "forecast_caveat": "The data mixes perception and generation; downstream reliability depends heavily on task framing.",
    },
    "search_document": {
        "label": "Search and document work",
        "interpretation": "Search, long-document handling, retrieval-facing work and document synthesis.",
        "forecast_caveat": "Retrieval quality, source grounding and tool access can dominate model-only scores.",
    },
    "finance_quant": {
        "label": "Finance and quantitative analysis",
        "interpretation": "Financial QA, quantitative coding, risk, pricing, forecasting and professional finance tasks.",
        "forecast_caveat": "Benchmarks are sparse and often workflow-specific; treat forecasts as directional until more longitudinal data exists.",
    },
    "legal_reasoning": {
        "label": "Legal reasoning",
        "interpretation": "Legal issue spotting, rule application, citations and jurisdiction-specific legal reasoning.",
        "forecast_caveat": "Legal performance is highly jurisdictional; benchmark score is not professional legal authority.",
    },
}

MESSAGE_WORKLOAD_PROFILES = [
    {
        "profile": "simple_chat",
        "display_name": "Simple chat or Q&A",
        "input_tokens": 900,
        "output_tokens": 500,
        "price_quantile": 0.20,
        "complexity_label": "low",
    },
    {
        "profile": "knowledge_work_message",
        "display_name": "Knowledge-work message",
        "input_tokens": 3_500,
        "output_tokens": 1_200,
        "price_quantile": 0.50,
        "complexity_label": "medium",
    },
    {
        "profile": "long_context_analysis",
        "display_name": "Long-context analysis",
        "input_tokens": 45_000,
        "output_tokens": 5_000,
        "price_quantile": 0.75,
        "complexity_label": "high",
    },
    {
        "profile": "agentic_workflow",
        "display_name": "Agentic workflow run",
        "input_tokens": 220_000,
        "output_tokens": 30_000,
        "price_quantile": 0.90,
        "complexity_label": "frontier_agentic",
    },
]

MESSAGE_MIX_BY_YEAR = {
    2023: {"simple_chat": 0.70, "knowledge_work_message": 0.25, "long_context_analysis": 0.05, "agentic_workflow": 0.00},
    2024: {"simple_chat": 0.57, "knowledge_work_message": 0.30, "long_context_analysis": 0.10, "agentic_workflow": 0.03},
    2025: {"simple_chat": 0.43, "knowledge_work_message": 0.33, "long_context_analysis": 0.17, "agentic_workflow": 0.07},
    2026: {"simple_chat": 0.32, "knowledge_work_message": 0.34, "long_context_analysis": 0.22, "agentic_workflow": 0.12},
}

FIXED_TASK_PROFILES = [
    {
        "task_profile": "thesis_quality_longform",
        "display_name": "Thesis-quality long-form writing",
        "domain": "language_writing",
        "required_domain_score": 88.0,
        "input_tokens": 60_000,
        "output_tokens": 35_000,
        "human_gate": "advisor, fact and citation review",
    },
    {
        "task_profile": "software_issue_resolution",
        "display_name": "Repository issue resolution",
        "domain": "software_engineering",
        "required_domain_score": 82.0,
        "input_tokens": 320_000,
        "output_tokens": 45_000,
        "human_gate": "senior engineer review and tests",
    },
    {
        "task_profile": "finance_analysis_memo",
        "display_name": "Financial analysis memo",
        "domain": "finance_quant",
        "required_domain_score": 65.0,
        "input_tokens": 45_000,
        "output_tokens": 8_000,
        "human_gate": "assumption and control owner signoff",
    },
    {
        "task_profile": "legal_due_diligence_memo",
        "display_name": "Legal due-diligence memo",
        "domain": "legal_reasoning",
        "required_domain_score": 90.0,
        "input_tokens": 55_000,
        "output_tokens": 12_000,
        "human_gate": "licensed legal review",
    },
    {
        "task_profile": "long_document_synthesis",
        "display_name": "Long-document synthesis",
        "domain": "search_document",
        "required_domain_score": 84.0,
        "input_tokens": 120_000,
        "output_tokens": 9_000,
        "human_gate": "source-grounding and factual review",
    },
    {
        "task_profile": "customer_support_resolution",
        "display_name": "Customer-support resolution",
        "domain": "instruction_following",
        "required_domain_score": 78.0,
        "input_tokens": 5_000,
        "output_tokens": 1_000,
        "human_gate": "policy, refund and safety review",
    },
]

QUALITY_ADJUSTED_COST_PRIORS = {
    "conservative": {
        "annual_factor": 0.65,
        "frontier_price_factor": 0.98,
        "note": "Deployment-friction case; task cost falls, but slower than pure benchmark price-performance estimates.",
    },
    "base": {
        "annual_factor": 0.48,
        "frontier_price_factor": 1.00,
        "note": "Middle case between observed API price competition and benchmark-level quality-adjusted cost declines.",
    },
    "aggressive": {
        "annual_factor": 0.36,
        "frontier_price_factor": 1.06,
        "note": "Fast capability diffusion; frontier workload still uses larger reasoning/context budgets.",
    },
}

# Explicit authored assumptions for the share of US occupation tasks materially
# touched by AI at each horizon.  A lookup table replaces the previous
# pseudo-formula: these are scenario inputs to argue with, not estimates
# derived from a transition model.  Values are capped by observed p90
# substitution pressure before publication.
TASK_CONTACT_ASSUMPTIONS = {
    "conservative": {2: 0.05, 5: 0.14, 10: 0.26},
    "base": {2: 0.09, 5: 0.22, 10: 0.40},
    "aggressive": {2: 0.14, 5: 0.34, 10: 0.62},
}

# Documented bounds for domain velocity transforms.  The observed (signed)
# slope stays visible in annual_frontier_point_gain_observed; the used gain is
# a bounded, non-negative planning input.
ANNUAL_GAIN_CAP = 12.0
GAP_CLOSURE_RATE_MIN = 0.035
GAP_CLOSURE_RATE_MAX = 0.72

BENCHMARK_DOMAIN_KEYWORDS = [
    ("medicine", ["medqa", "medmcqa", "pubmedqa", "mmlu_anatomy", "mmlu_clinical", "medical", "medicine", "biology", "genetics"]),
    ("software_engineering", ["swebench", "swe-bench", "livecodebench", "lcb", "humaneval", "mbpp", "code", "coding", "webdev"]),
    ("agentic_terminal", ["terminal-bench", "terminal_bench", "terminal bench", "terminal", "agentic"]),
    ("finance_quant", ["finance", "qfbench", "quantitativefinance", "quantitative finance", "black-scholes", "var", "risk"]),
    ("legal_reasoning", ["legal", "lexometrica", "lawbench", "legalbench", "casehold"]),
    ("mathematics", ["math", "gsm8k", "aime", "algebra", "geometry", "number theory", "precalculus", "counting"]),
    ("science_reasoning", ["gpqa", "arc_challenge", "arc", "science", "bbh", "musr", "reasoning", "critic", "hle"]),
    ("instruction_following", ["ifeval", "instruction", "format", "constraint"]),
    ("vision_multimodal", ["vision", "image", "mmmu", "multimodal", "text_to_image", "image_edit"]),
    ("search_document", ["search", "document", "retrieval", "long context", "long-context", "rag"]),
    ("language_writing", ["language", "text", "paraphrase", "story", "writing", "summarization", "chat"]),
]


def ensure_dirs() -> None:
    for path in [ANALYSIS, RAW_AEI, RAW_DOMAIN, FIGURES, REPORT, DOCS]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_table(name: str) -> pd.DataFrame:
    path = DATASET / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset table: {path}")
    return filter_public_entities(pd.read_csv(path, low_memory=False))


def filter_public_entities(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in [
            "openrouter_id",
            "canonical_model",
            "vendor",
            "source_vendor",
            "model_family",
            "family",
            "organization",
            "org",
            "author",
            "model",
            "model_name",
            "model_path",
            "source_url",
        ]
        if col in df.columns
    ]
    if not cols or df.empty:
        return df
    # Vectorized concatenation is dramatically faster than constructing one
    # Python Series per row on million-row leaderboard tables.
    text = pd.Series("", index=df.index, dtype="string")
    for col in cols:
        text = text.str.cat(df[col].astype("string").fillna(""), sep=" ")
    text = text.str.lower()
    pattern = "|".join(EXCLUDED_PUBLIC_ENTITY_PATTERNS)
    return df[~text.str.contains(pattern, regex=True, na=False)].copy()


def family_column(df: pd.DataFrame) -> str:
    return "model_family" if "model_family" in df.columns else "family"


def write_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["analysis_captured_at"] = CAPTURED_AT
    csv_path = ANALYSIS / f"{name}.csv"
    parquet_path = ANALYSIS / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    return df


def download_aei_file(key: str, overwrite: bool = False) -> Path:
    rel = AEI_FILES[key]
    path = RAW_AEI / rel.replace("/", "__")
    if path.exists() and not overwrite:
        return path
    url = f"{AEI_BASE}/{rel}"
    headers = {"User-Agent": "ai-capability-signals/0.1 reproducible research"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def load_aei(overwrite: bool = False) -> dict[str, pd.DataFrame]:
    tables = {}
    for key in AEI_FILES:
        path = download_aei_file(key, overwrite=overwrite)
        tables[key] = pd.read_csv(path, low_memory=False)
    return tables


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def minmax(series: pd.Series, invert: bool = False, log: bool = False, missing_value: float = 50.0, fill_missing: bool = True) -> pd.Series:
    values = numeric(series).replace([np.inf, -np.inf], np.nan)
    if log:
        values = np.log1p(values.clip(lower=0))
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(float(lo), float(hi)):
        out = pd.Series(np.nan if not fill_missing else missing_value, index=series.index)
    else:
        out = (values - lo) / (hi - lo) * 100
    if invert:
        out = 100 - out
    # Missing public evidence is not evidence of zero capability.  By default a
    # neutral value keeps sparse families from being mechanically pushed to the
    # bottom; composite components instead pass fill_missing=False and
    # renormalize weights over available inputs (see weighted_component).
    if fill_missing:
        out = out.fillna(missing_value)
    return out.clip(0, 100)


def weighted_component(scores: pd.DataFrame, specs: list[tuple[str, float, dict[str, Any]]]) -> pd.Series:
    """Weighted mean over the inputs each row actually has.

    Instead of injecting a fake neutral 50 for missing sub-inputs (which lets
    data-less families beat families with real low values), each row's weights
    are renormalized across its available inputs.  Rows with no input at all
    fall back to the neutral 50 and stay visible in coverage diagnostics.
    """
    total = pd.Series(0.0, index=scores.index)
    weight_sum = pd.Series(0.0, index=scores.index)
    for col, weight, kwargs in specs:
        if col not in scores.columns:
            continue
        raw = scores[col]
        available = numeric(raw).notna() & np.isfinite(numeric(raw).fillna(np.nan))
        scaled = minmax(raw, fill_missing=False, **kwargs)
        # An input that exists but cannot be normalized (degenerate scale with
        # one distinct value) contributes the neutral midpoint, never zero.
        scaled = scaled.fillna(50.0)
        contribution = weight * available.astype(float)
        total = total + scaled * contribution
        weight_sum = weight_sum + contribution
    out = total / weight_sum.replace(0, np.nan)
    return out.fillna(50.0).clip(0, 100)


def soc_base(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d{2}-\d{4})", text)
    return match.group(1) if match else text[:7]


def sentence_join(items: list[str], limit: int = 4) -> str:
    cleaned = [clean_text(x) for x in items if clean_text(x)]
    return "; ".join(cleaned[:limit])


@lru_cache(maxsize=65_536)
def family_from_text(*parts: Any) -> str:
    text = " ".join(clean_text(p) for p in parts)
    family = classify_family(text)
    if family.lower() in {"unknown", ""}:
        lowered = text.lower()
        if "command" in lowered:
            return "Command"
        if "xai" in lowered or "grok" in lowered:
            return "Grok"
    return family


@lru_cache(maxsize=65_536)
def access_from_text(name: Any, organization: Any = "", license_value: Any = "") -> str:
    return classify_access(name, organization, str(license_value), str(license_value))


def normalize_name(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def cached_text(source_id: str, url: str, overwrite: bool = False, timeout: int = 120) -> str:
    suffix = ".json" if url.endswith(".json") or "/api/" in url else ".html" if url.endswith("/") or "." not in Path(url).suffix else Path(url).suffix
    if suffix not in {".json", ".html", ".js", ".txt"}:
        suffix = ".dat"
    path = RAW_DOMAIN / f"{slug(source_id)}{suffix}"
    if path.exists() and not overwrite:
        return path.read_text(encoding="utf-8")
    response = requests.get(url, headers={"User-Agent": "frontier-ai-domain-benchmark-analysis/0.1"}, timeout=timeout)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def cached_bytes(source_id: str, url: str, overwrite: bool = False, timeout: int = 120) -> bytes:
    suffix = Path(url).suffix or ".bin"
    path = RAW_DOMAIN / f"{slug(source_id)}{suffix}"
    if path.exists() and not overwrite:
        return path.read_bytes()
    response = requests.get(url, headers={"User-Agent": "frontier-ai-domain-benchmark-analysis/0.1"}, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_percent(value: Any) -> float:
    text = clean_text(value).replace("%", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    return float(text) if text not in {"", ".", "-"} else np.nan


@lru_cache(maxsize=32_768)
def parse_any_date(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(number):
            unit = "ms" if float(number) > 10_000_000_000 else "s"
            parsed = pd.to_datetime(float(number), unit=unit, errors="coerce", utc=True)
    return parsed.date().isoformat() if pd.notna(parsed) else ""


def domain_label(domain: str) -> str:
    return CAPABILITY_DOMAINS.get(domain, {}).get("label", domain.replace("_", " ").title())


def infer_capability_domain(*parts: Any) -> str:
    text = " ".join(clean_text(part).lower().replace("-", "_") for part in parts)
    for domain, keywords in BENCHMARK_DOMAIN_KEYWORDS:
        if any(keyword.replace("-", "_") in text for keyword in keywords):
            return domain
    return "science_reasoning"


def normalize_benchmark_score(score: Any, score_unit: str = "") -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return np.nan
    unit = clean_text(score_unit).lower()
    if "rating" in unit or "elo" in unit:
        return float(value)
    if "fraction" in unit or (0 <= float(value) <= 1 and "%" not in unit and "percent" not in unit):
        return float(value) * 100
    return float(value)


def append_domain_row(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    source_name: str,
    benchmark: str,
    domain: str,
    task: str,
    model_name: Any,
    score: Any,
    score_unit: str,
    source_url: str,
    eval_date: Any = "",
    vendor: Any = "",
    model_family: Any = "",
    evidence_level: str = "observed",
    sample_size: Any = np.nan,
    benchmark_weight: float = 1.0,
    limitations: str = "",
    date_provenance: str = "evaluation_date",
    temporal_eligible: bool = True,
) -> None:
    try:
        raw_score = float(score)
    except (TypeError, ValueError):
        return
    if not np.isfinite(raw_score):
        return
    family = clean_text(model_family) or frontier_family_from_model(model_name, "", vendor)
    rows.append(
        {
            "source_id": source_id,
            "source_name": source_name,
            "benchmark": clean_text(benchmark),
            "domain": domain,
            "domain_label": domain_label(domain),
            "task": clean_text(task),
            "model_name": clean_text(model_name),
            "model_family": family,
            "vendor": clean_text(vendor) or FAMILY_VENDOR_MAP.get(family, family),
            "score": float(raw_score),
            "score_unit": score_unit,
            "score_normalized_0_100": normalize_benchmark_score(raw_score, score_unit),
            "eval_date": parse_any_date(eval_date),
            "source_url": source_url,
            "evidence_level": evidence_level,
            "sample_size": sample_size,
            "benchmark_weight": benchmark_weight,
            "limitations": limitations,
            "date_provenance": date_provenance,
            "temporal_eligible": bool(temporal_eligible),
        }
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series, fallback: float = 0.0) -> pd.Series:
    out = numeric(numerator) / numeric(denominator).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fallback)


def positive_min(series: pd.Series) -> float:
    values = numeric(series)
    values = values[values > 0]
    return float(values.min()) if len(values) else np.nan


@lru_cache(maxsize=32_768)
def frontier_family_from_model(name: Any = "", model_id: Any = "", vendor: Any = "") -> str:
    text = " ".join(clean_text(part).lower() for part in [name, model_id, vendor])
    checks = [
        ("DeepSeek", ["deepseek"]),
        ("Claude", ["claude", "anthropic"]),
        ("Gemma", ["gemma"]),
        ("Gemini", ["gemini"]),
        ("Grok", ["grok", "x-ai", "xai"]),
        ("Qwen", ["qwen", "qwq", "alibaba"]),
        ("Llama", ["llama", "sao10k", "meta-llama"]),
        ("Mistral", ["mistral", "mixtral", "codestral", "ministral", "pixtral", "devstral", "voxtral"]),
        ("Phi", ["phi-", "phi ", "microsoft/phi"]),
        ("Command", ["command-r"]),
        ("GPT", ["gpt", "openai", " o1", " o3", " o4"]),
    ]
    for family, needles in checks:
        if any(needle in text for needle in needles):
            return family
    return family_from_text(name, model_id, vendor)


def apply_family_score_components(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    # Each component renormalizes its sub-weights over the inputs a family
    # actually has.  This removes two biases of the previous fill-with-50
    # approach: data-less families no longer score 50 by default on inputs they
    # never had, and closed vendors are no longer handed a fake 50 on the
    # open-only Open LLM Leaderboard sub-signal.
    scores["performance_component"] = weighted_component(
        scores,
        [
            ("lmarena_best", 0.45, {}),
            ("swebench_best", 0.30, {}),
            ("openllm_top_mean", 0.25, {}),
        ],
    )
    scores["release_velocity_component"] = weighted_component(
        scores,
        [
            ("recent_api_releases", 0.55, {}),
            ("epoch_recent_releases", 0.45, {}),
        ],
    )
    scores["ecosystem_component"] = weighted_component(
        scores,
        [
            ("hf_downloads", 0.40, {"log": True}),
            ("github_model_mentions", 0.25, {"log": True}),
            ("openalex_paper_mentions", 0.25, {"log": True}),
            ("hf_likes", 0.10, {"log": True}),
        ],
    )
    scores["capability_surface_component"] = weighted_component(
        scores,
        [
            ("context_window_max", 0.35, {"log": True}),
            ("max_output_tokens", 0.20, {"log": True}),
            ("multimodal_models", 0.20, {}),
            ("training_compute_max", 0.25, {"log": True}),
        ],
    )
    if "output_price_min" in scores.columns:
        price = scores["output_price_min"].replace(0, np.nan)
    else:
        price = pd.Series(np.nan, index=scores.index)
    scores["cost_efficiency_component"] = minmax(price, invert=True, log=True)
    scores["openness_component"] = (
        scores.get("hf_open_weight_share", pd.Series(index=scores.index)).fillna(0) * 55
        + scores.get("open_weight_epoch_share", pd.Series(index=scores.index)).fillna(0) * 35
        + minmax(scores.get("open_api_models", pd.Series(index=scores.index))) * 0.10
    ).clip(0, 100)
    scores["frontier_momentum_heuristic_index"] = sum(scores[col] * weight for col, weight in COMPONENT_WEIGHTS.items()).round(2)
    scores["frontier_momentum_score"] = scores["frontier_momentum_heuristic_index"]
    scores["rank"] = scores["frontier_momentum_heuristic_index"].rank(ascending=False, method="min").astype(int)
    evidence_cols = [
        "lmarena_models",
        "api_model_count",
        "epoch_model_count",
        "openllm_metric_count",
        "swebench_submissions",
        "hf_models",
        "github_model_mentions",
        "openalex_paper_mentions",
    ]
    for col in evidence_cols:
        if col not in scores.columns:
            scores[col] = 0
    scores["evidence_count"] = scores[evidence_cols].fillna(0).sum(axis=1)
    scores["evidence_source_count"] = scores[evidence_cols].fillna(0).gt(0).sum(axis=1)
    scores["effective_evidence_count"] = scores[evidence_cols].fillna(0).clip(lower=0).map(np.log1p).sum(axis=1).round(3)
    return scores


def build_company_frontier_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    openrouter = read_csv_table("openrouter_models_catalog")
    epoch = read_csv_table("epoch_models_normalized")
    lmarena = read_csv_table("lmarena_full")
    openllm = read_csv_table("openllm_leaderboard_metrics_long")
    swe = read_csv_table("swebench_submissions")
    hf = read_csv_table("huggingface_model_rollups")
    github_mentions = read_csv_table("github_model_mentions")
    openalex_mentions = read_csv_table("openalex_model_mentions")

    reference = pd.Timestamp(REFERENCE_DATE)
    if "release_date" in openrouter:
        release = pd.to_datetime(openrouter["release_date"], errors="coerce")
        openrouter = openrouter[release.isna() | release.le(reference)].copy()
    if "release_date" in epoch:
        release = pd.to_datetime(epoch["release_date"], errors="coerce")
        epoch = epoch[release.isna() | release.le(reference)].copy()
    if "leaderboard_publish_date" in lmarena:
        published = pd.to_datetime(lmarena["leaderboard_publish_date"], errors="coerce")
        lmarena = lmarena[published.isna() | published.le(reference)].copy()

    families = set(FRONTIER_FAMILIES)
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    openrouter_family = "model_family"
    epoch_family = family_column(epoch)
    hf_family = family_column(hf)
    families.update(openrouter[openrouter_family].dropna().astype(str).head(200))
    families.update(epoch[epoch_family].dropna().astype(str).head(500))
    rows: dict[str, dict[str, Any]] = {family: {"model_family": family, "family": family} for family in families if family and family not in {"unknown", "Other"}}

    arena = lmarena.copy()
    arena["family"] = [family_from_text(n, o) for n, o in zip(arena.get("model_name", ""), arena.get("organization", ""))]
    arena["access_class"] = [access_from_text(n, o, lic) for n, o, lic in zip(arena.get("model_name", ""), arena.get("organization", ""), arena.get("license", ""))]
    arena["rating"] = numeric(arena["rating"])
    # The full table stacks repeated leaderboard snapshots whose Elo scales
    # drift over time.  Normalize within each (category, snapshot) so a model's
    # best is its strongest percentile inside a single comparable board, and
    # deduplicate vote counts per (category, model) before summing so snapshot
    # frequency cannot masquerade as extra votes.
    arena["rating_norm"] = (
        arena.groupby(["category", "leaderboard_publish_date"], dropna=False)["rating"].rank(method="average", pct=True) * 100
    )
    arena["vote_best"] = arena.groupby(["category", "model_name"], dropna=False)["vote_count"].transform("max")
    arena_votes = arena.drop_duplicates(["family", "category", "model_name"], keep="first")
    arena_group = arena.groupby("family", dropna=False).agg(
        lmarena_best=("rating_norm", "max"),
        lmarena_median_top=("rating_norm", lambda s: s.dropna().sort_values(ascending=False).head(20).median()),
        lmarena_models=("model_name", "nunique"),
        lmarena_open_best=("rating_norm", lambda s: s[arena.loc[s.index, "access_class"].isin(["open_weight", "likely_open_weight"])].max()),
        lmarena_closed_best=("rating_norm", lambda s: s[arena.loc[s.index, "access_class"].eq("closed_or_api")].max()),
    )
    vote_sums = arena_votes.groupby("family", dropna=False)["vote_best"].sum().rename("lmarena_votes")
    for family, row in arena_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())
        if family in vote_sums.index:
            rows[family]["lmarena_votes"] = float(vote_sums.loc[family])

    openrouter["release_date_dt"] = pd.to_datetime(openrouter["release_date"], errors="coerce")
    recent_cutoff = pd.Timestamp(REFERENCE_DATE) - pd.Timedelta(days=240)
    openrouter_group = openrouter.groupby(openrouter_family, dropna=False).agg(
        api_model_count=("model_id", "count"),
        context_window_max=("context_window", "max"),
        output_price_min=("output_usd_per_1m", positive_min),
        input_price_min=("input_usd_per_1m", positive_min),
        max_output_tokens=("max_output_tokens", "max"),
        multimodal_models=("modality", lambda s: int(s.astype(str).str.contains("image|video|audio|file", case=False, regex=True).sum())),
        recent_api_releases=("release_date_dt", lambda s: int((s >= recent_cutoff).sum())),
        closed_api_models=("access_class", lambda s: int(s.astype(str).eq("closed_or_api").sum())),
        open_api_models=("access_class", lambda s: int(s.astype(str).isin(["open_weight", "likely_open_weight"]).sum())),
    )
    for family, row in openrouter_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())

    epoch["release_date_dt"] = pd.to_datetime(epoch["release_date"], errors="coerce")
    epoch_group = epoch.groupby(epoch_family, dropna=False).agg(
        epoch_model_count=("model_id", "count"),
        epoch_recent_releases=("release_date_dt", lambda s: int((s >= recent_cutoff).sum())),
        training_compute_max=("training_compute_flop", "max"),
        parameter_max=("parameters", "max"),
        open_weight_epoch_share=("access_class", lambda s: float(s.astype(str).str.contains("open", case=False).mean())),
    )
    for family, row in epoch_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())

    openllm = openllm[~openllm["metric"].astype(str).str.contains("stderr", case=False, na=False)].copy()
    openllm = openllm[numeric(openllm["value"]).between(0, 1, inclusive="both")].copy()
    openllm["family"] = [family_from_text(n, p) for n, p in zip(openllm.get("model_name", ""), openllm.get("model_path", ""))]
    openllm_group = openllm.groupby("family", dropna=False).agg(
        openllm_best=("value", "max"),
        openllm_top_mean=("value", lambda s: s.dropna().sort_values(ascending=False).head(40).mean()),
        openllm_metric_count=("value", "count"),
    )
    for family, row in openllm_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())

    swe["family"] = [family_from_text(m, s, sub) for m, s, sub in zip(swe.get("model", ""), swe.get("system_name", ""), swe.get("submission", ""))]
    swe_group = swe.groupby("family", dropna=False).agg(
        swebench_best=("score", "max"),
        swebench_submissions=("submission", "count"),
        swebench_recent_best=("score", lambda s: s.dropna().sort_values(ascending=False).head(5).mean()),
    )
    for family, row in swe_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())

    hf_group = hf.groupby(hf_family, dropna=False).agg(
        hf_models=("model_id", "nunique"),
        hf_downloads=("downloads", "sum"),
        hf_likes=("likes", "sum"),
        hf_weight_bytes=("lfs_file_bytes", "sum"),
        hf_open_weight_share=("access_class", lambda s: float(s.astype(str).str.contains("open", case=False).mean())),
    )
    for family, row in hf_group.iterrows():
        rows.setdefault(family, {"model_family": family, "family": family}).update(row.to_dict())

    github_group = github_mentions.groupby("family").size().rename("github_model_mentions")
    for family, value in github_group.items():
        rows.setdefault(family, {"model_family": family, "family": family})["github_model_mentions"] = int(value)
    paper_group = openalex_mentions.groupby("family").size().rename("openalex_paper_mentions")
    for family, value in paper_group.items():
        rows.setdefault(family, {"model_family": family, "family": family})["openalex_paper_mentions"] = int(value)

    scores = pd.DataFrame(rows.values())
    for col in scores.columns:
        if col not in {"family", "model_family"}:
            scores[col] = numeric(scores[col])
    scores = scores[(scores["model_family"].ne("Other")) & (scores["model_family"].isin(FRONTIER_FAMILIES) | (scores.get("api_model_count", 0).fillna(0) >= 2))].copy()

    scores = apply_family_score_components(scores)
    sensitivity = build_company_score_sensitivity(scores)
    stable_top = set(sensitivity[sensitivity["rank"].le(3)].groupby("model_family").size().loc[lambda s: s >= 3].index)
    scores["sensitivity_label"] = np.where(scores["model_family"].isin(stable_top), "stable_top_tier", "weight_sensitive")
    scores = scores.sort_values(["frontier_momentum_heuristic_index", "evidence_count"], ascending=False)

    components = scores[
        [
            "model_family",
            "family",
            "performance_component",
            "release_velocity_component",
            "ecosystem_component",
            "capability_surface_component",
            "cost_efficiency_component",
            "openness_component",
            "frontier_momentum_heuristic_index",
            "frontier_momentum_score",
            "rank",
            "sensitivity_label",
        ]
    ].copy()
    write_table(scoring_methodology(), "company_score_methodology")
    write_table(sensitivity, "company_score_sensitivity")
    return write_table(scores, "company_frontier_scores"), write_table(components, "company_score_components")


def scoring_methodology() -> pd.DataFrame:
    rows = [
        ("performance_component", "LMArena, SWE-bench, Open LLM Leaderboard", "min-max over benchmark fields", COMPONENT_WEIGHTS["performance_component"], "Capability proxy; heuristic, not universal score."),
        ("release_velocity_component", "OpenRouter and Epoch release dates", "recent release counts", COMPONENT_WEIGHTS["release_velocity_component"], "Measures visible product/model cadence."),
        ("ecosystem_component", "Hugging Face, GitHub, OpenAlex", "log-scaled public ecosystem signals", COMPONENT_WEIGHTS["ecosystem_component"], "Distribution/research pull proxy."),
        ("capability_surface_component", "OpenRouter context/output and Epoch compute", "log-scaled capability surface fields", COMPONENT_WEIGHTS["capability_surface_component"], "Product surface and disclosed scale proxy."),
        ("cost_efficiency_component", "OpenRouter output prices", "inverted log-scaled minimum output price", COMPONENT_WEIGHTS["cost_efficiency_component"], "Cost signal, not quality adjusted."),
        ("openness_component", "HF/Epoch/OpenRouter access labels", "weighted open-weight share", COMPONENT_WEIGHTS["openness_component"], "Inspectability and distribution proxy."),
    ]
    return pd.DataFrame(rows, columns=["component", "source_columns", "transform", "baseline_weight", "rationale"])


def build_company_score_sensitivity(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, weights in SENSITIVITY_WEIGHTS.items():
        total = sum(weights.values())
        for _, row in scores.iterrows():
            value = sum(float(row.get(component, 0) or 0) * weight for component, weight in weights.items()) / total
            rows.append({"scenario": scenario, "model_family": row["model_family"], "heuristic_index": round(value, 2)})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("scenario")["heuristic_index"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["scenario", "rank", "model_family"])


def infer_task_columns(mappings: pd.DataFrame) -> tuple[str | None, str | None]:
    lower_cols = {c.lower().strip(): c for c in mappings.columns}
    soc_col = lower_cols.get("o*net-soc code")
    task_col = lower_cols.get("task")
    if task_col is None:
        task_col = lower_cols.get("task statement")
    for c in mappings.columns:
        low = c.lower()
        if soc_col is None and ("soc" in low or "occupation" in low) and "title" not in low:
            soc_col = c
        if task_col is None and ("task" in low or "statement" in low) and "id" not in low:
            task_col = c
    return soc_col, task_col


def keyword_score(text: str, word_groups: dict[str, list[str]]) -> dict[str, float]:
    low = text.lower()
    scores = {}
    for group, words in word_groups.items():
        hits = sum(1 for word in words if word in low)
        scores[group] = min(1.0, hits / max(2, len(words) * 0.35))
    return scores


def build_job_exposure_scores(aei: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = aei["job_exposure"].copy()
    wage = aei["wage_data"].copy()
    bls = aei["bls_employment_may_2023"].copy()
    task_pen = aei["task_penetration"].copy()
    auto_task = aei["automation_by_task"].copy()
    mappings = aei["onet_task_statements"].copy()
    task_pct = aei["onet_task_mappings"].copy()

    jobs["soc_base"] = jobs["occ_code"].map(soc_base)
    wage["soc_base"] = wage["SOCcode"].map(soc_base)
    wage_roll = wage.sort_values("SOCcode").groupby("soc_base", as_index=False).agg(
        median_salary=("MedianSalary", "max"),
        job_forecast=("JobForecast", "max"),
        chance_auto=("ChanceAuto", "max"),
        job_zone=("JobZone", "max"),
        wage_job_name=("JobName", lambda s: sentence_join(list(s), 2)),
        job_family=("JobFamily", "first"),
        bright_outlook=("isBright", "max"),
        green_job=("isGreen", "max"),
    )
    jobs = jobs.merge(wage_roll, on="soc_base", how="left")

    if len(bls) > 0:
        bls = bls.rename(columns={bls.columns[0]: "bls_title", bls.columns[1]: "bls_employment"})
        bls["title_key"] = bls["bls_title"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        jobs["title_key"] = jobs["title"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        jobs = jobs.merge(bls[["title_key", "bls_employment"]], on="title_key", how="left")
        bls["job_family_key"] = bls["title_key"].str.replace(r"\s+occupations?$", "", regex=True).str.strip()
        jobs["job_family_key"] = jobs["job_family"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        jobs = jobs.merge(
            bls[["job_family_key", "bls_employment"]].rename(columns={"bls_employment": "bls_major_group_employment"}),
            on="job_family_key",
            how="left",
        )
        # The BLS file contains major-group totals, not occupation-level totals.
        # Repeating the full group total on every detailed occupation multiplies
        # employment by the number of occupations.  Allocate the group total
        # evenly only where an exact title match is unavailable, preserving the
        # aggregate order of magnitude without pretending it is observed detail.
        family_counts = jobs.groupby("job_family_key")["soc_base"].transform("nunique").clip(lower=1)
        jobs["bls_employment_exact"] = numeric(jobs["bls_employment"])
        jobs["bls_employment"] = jobs["bls_employment_exact"].fillna(
            numeric(jobs["bls_major_group_employment"]) / family_counts
        )
        jobs["employment_weight_provenance"] = np.where(
            jobs["bls_employment_exact"].notna(), "exact_title", np.where(jobs["bls_employment"].notna(), "allocated_major_group", "missing")
        )

    soc_col, task_col = infer_task_columns(mappings)
    task_features = pd.DataFrame()
    if soc_col and task_col:
        work = mappings[[soc_col, task_col]].rename(columns={soc_col: "soc_raw", task_col: "task"}).dropna()
        work["soc_base"] = work["soc_raw"].map(soc_base)
        work["task_key"] = work["task"].astype(str).str.lower().str.strip()
        task_pen["task_key"] = task_pen["task"].astype(str).str.lower().str.strip()
        auto_task["task_key"] = auto_task["task_name"].astype(str).str.lower().str.strip()
        work = work.merge(task_pen[["task_key", "penetration"]], on="task_key", how="left")
        if {"task_name", "pct"}.issubset(task_pct.columns):
            task_pct["task_key"] = task_pct["task_name"].astype(str).str.lower().str.strip()
            work = work.merge(task_pct[["task_key", "pct"]].rename(columns={"pct": "aei_task_pct"}), on="task_key", how="left")
        work = work.merge(
            auto_task[["task_key", "feedback_loop", "directive", "task_iteration", "validation", "learning", "filtered"]],
            on="task_key",
            how="left",
        )
        for group in DIGITAL_TASK_WORDS:
            work[group] = work["task"].astype(str).map(lambda x, g=group: keyword_score(x, {g: DIGITAL_TASK_WORDS[g]})[g])
        for group in BOTTLENECK_WORDS:
            work[group] = work["task"].astype(str).map(lambda x, g=group: keyword_score(x, {g: BOTTLENECK_WORDS[g]})[g])
        task_features = work.groupby("soc_base", as_index=False).agg(
            task_count=("task", "count"),
            mean_task_penetration=("penetration", "mean"),
            mean_aei_task_pct=("aei_task_pct", "mean"),
            directive_share=("directive", "mean"),
            feedback_loop_share=("feedback_loop", "mean"),
            task_iteration_share=("task_iteration", "mean"),
            validation_share=("validation", "mean"),
            learning_share=("learning", "mean"),
            filtered_share=("filtered", "mean"),
            language_task_share=("language", "mean"),
            code_task_share=("code", "mean"),
            analysis_task_share=("analysis", "mean"),
            visual_task_share=("visual", "mean"),
            agentic_task_share=("agentic", "mean"),
            physical_bottleneck=("physical", "mean"),
            human_trust_bottleneck=("human_trust", "mean"),
            regulated_bottleneck=("regulated", "mean"),
            example_tasks=("task", lambda s: sentence_join(list(s), 3)),
        )
        jobs = jobs.merge(task_features, on="soc_base", how="left")

    for group in DIGITAL_TASK_WORDS:
        col = f"{group}_task_share"
        fallback = jobs["title"].astype(str).map(lambda x, g=group: keyword_score(x, {g: DIGITAL_TASK_WORDS[g]})[g])
        current = jobs[col] if col in jobs.columns else pd.Series(np.nan, index=jobs.index)
        jobs[col] = current.fillna(fallback)
    for group in BOTTLENECK_WORDS:
        col = f"{group}_bottleneck"
        fallback = jobs["title"].astype(str).map(lambda x, g=group: keyword_score(x, {g: BOTTLENECK_WORDS[g]})[g])
        current = jobs[col] if col in jobs.columns else pd.Series(np.nan, index=jobs.index)
        jobs[col] = current.fillna(fallback)

    exposure = numeric(jobs["observed_exposure"]).fillna(0)
    chance = numeric(jobs.get("chance_auto", pd.Series(index=jobs.index))).replace(-1, np.nan) / 100
    directive = numeric(jobs.get("directive_share", pd.Series(index=jobs.index))).fillna(0)
    collaborative = (
        numeric(jobs.get("feedback_loop_share", pd.Series(index=jobs.index))).fillna(0)
        + numeric(jobs.get("task_iteration_share", pd.Series(index=jobs.index))).fillna(0)
        + numeric(jobs.get("validation_share", pd.Series(index=jobs.index))).fillna(0)
        + numeric(jobs.get("learning_share", pd.Series(index=jobs.index))).fillna(0)
    ).clip(0, 1)
    digital = jobs[[f"{g}_task_share" for g in DIGITAL_TASK_WORDS]].mean(axis=1).fillna(0)
    bottleneck = (
        jobs["physical_bottleneck"].fillna(0) * 0.45
        + jobs["human_trust_bottleneck"].fillna(0) * 0.35
        + jobs["regulated_bottleneck"].fillna(0) * 0.20
    ).clip(0, 1)

    task_penetration_signal = (
        numeric(jobs.get("mean_task_penetration", pd.Series(index=jobs.index))).fillna(0) * 0.65
        + numeric(jobs.get("mean_aei_task_pct", pd.Series(index=jobs.index))).fillna(0) * 0.35
    )
    jobs["capability_exposure_index"] = (100 * (0.56 * exposure + 0.24 * digital + 0.20 * task_penetration_signal)).round(2)
    jobs["substitution_pressure_index"] = (100 * (0.48 * exposure + 0.26 * directive + 0.26 * chance.fillna(chance.median())) * (1 - 0.62 * bottleneck)).round(2)
    jobs["augmentation_index"] = (100 * (0.50 * exposure + 0.32 * collaborative + 0.18 * (1 - jobs["physical_bottleneck"].fillna(0)))).round(2)
    jobs["human_bottleneck_index"] = (100 * bottleneck).round(2)
    jobs["near_term_disruption_index"] = (
        jobs["substitution_pressure_index"] * 0.48
        + jobs["augmentation_index"] * 0.30
        + minmax(jobs.get("median_salary", pd.Series(index=jobs.index))) * 0.12
        + minmax(jobs.get("job_forecast", pd.Series(index=jobs.index)), invert=True) * 0.10
    ).round(2)
    replacement_gate = (
        (1 - jobs["physical_bottleneck"].fillna(0) * 0.80)
        * (1 - jobs["human_trust_bottleneck"].fillna(0) * 0.70)
        * (1 - jobs["regulated_bottleneck"].fillna(0) * 0.75)
        * (0.50 + 0.50 * task_penetration_signal.clip(0, 1))
    ).clip(0, 1)
    jobs["full_job_automation_feasibility_index"] = (jobs["substitution_pressure_index"] * replacement_gate).round(2)
    jobs["augmentation_dominance_ratio"] = safe_divide(jobs["augmentation_index"], jobs["substitution_pressure_index"], fallback=0).round(3)
    jobs["dominant_outcome"] = np.select(
        [
            jobs["full_job_automation_feasibility_index"].ge(35),
            jobs["augmentation_dominance_ratio"].ge(1.15),
            jobs["human_bottleneck_index"].ge(25),
        ],
        ["replacement_candidate", "augmentation_first", "bottleneck_protected"],
        default="mixed_redesign",
    )
    jobs["scenario_2y_task_share_base"] = (jobs["substitution_pressure_index"] / 100 * 0.16 + jobs["augmentation_index"] / 100 * 0.22).round(3)
    jobs["scenario_5y_task_share_base"] = (jobs["substitution_pressure_index"] / 100 * 0.34 + jobs["augmentation_index"] / 100 * 0.39).round(3)
    jobs["scenario_10y_task_share_base"] = (jobs["substitution_pressure_index"] / 100 * 0.56 + jobs["augmentation_index"] / 100 * 0.58).clip(0, 0.92).round(3)
    jobs["risk_label"] = pd.cut(
        jobs["near_term_disruption_index"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["low", "moderate", "high", "very_high", "extreme"],
    ).astype(str)
    jobs = jobs.sort_values("near_term_disruption_index", ascending=False)

    domain_cols = [f"{g}_task_share" for g in DIGITAL_TASK_WORDS] + [f"{g}_bottleneck" for g in BOTTLENECK_WORDS]
    domain = jobs.groupby(jobs["soc_base"].str[:2]).agg({c: "mean" for c in domain_cols})
    domain["occupation_count"] = jobs.groupby(jobs["soc_base"].str[:2]).size()
    domain = domain.reset_index().rename(columns={"soc_base": "soc_major"})

    return write_table(jobs, "job_exposure_scores"), write_table(domain, "task_domain_exposure_heatmap")


def log_slope_by_year(df: pd.DataFrame, date_col: str, value_col: str, q: float = 0.9) -> tuple[float, pd.DataFrame]:
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = numeric(work[value_col])
    work = work.dropna()
    work = work[
        (work[date_col].dt.year >= 2018)
        & (work[date_col] <= pd.Timestamp(REFERENCE_DATE))
        & (work[value_col] > 0)
    ]
    if work.empty:
        return 0.0, pd.DataFrame()
    yearly = work.groupby(work[date_col].dt.year)[value_col].quantile(q).reset_index()
    yearly.columns = ["year", "value"]
    if len(yearly) < 3:
        return 0.0, yearly
    x = yearly["year"].to_numpy(dtype=float)
    y = np.log10(yearly["value"].to_numpy(dtype=float))
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, yearly


def capped_growth(raw_value: float, cap: float) -> tuple[float, bool]:
    if not np.isfinite(raw_value):
        return cap, True
    return min(raw_value, cap), raw_value > cap


def build_capability_forecasts(company_scores: pd.DataFrame, job_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch = read_csv_table("epoch_models_normalized")
    openrouter = read_csv_table("openrouter_models_catalog")
    lmarena = read_csv_table("lmarena_full")

    compute_slope, compute_history = log_slope_by_year(epoch, "release_date", "training_compute_flop", q=0.92)
    context_slope, context_history = log_slope_by_year(openrouter, "release_date", "context_window", q=0.85)

    openrouter["release_year"] = pd.to_datetime(openrouter["release_date"], errors="coerce").dt.year
    price_work = openrouter.dropna(subset=["release_year", "output_usd_per_1m"]).copy()
    price_work = price_work[price_work["release_year"].le(pd.Timestamp(REFERENCE_DATE).year)]
    price_work = price_work[price_work["output_usd_per_1m"] > 0]
    price_yearly = pd.DataFrame()
    price_slope = np.nan
    if len(price_work) >= 8:
        price_yearly = price_work.groupby("release_year")["output_usd_per_1m"].quantile(0.20).reset_index()
        if len(price_yearly) >= 2:
            price_slope = float(np.polyfit(price_yearly["release_year"], np.log10(price_yearly["output_usd_per_1m"]), 1)[0])
    # No fabricated fallback series: when the catalog cannot support a cohort
    # fit, the diagnostic records insufficient data and forward paths rely
    # solely on the explicit scenario slopes below.
    # OpenRouter is a current catalog. Grouping today's prices by model release
    # year is cross-sectional survivor/cohort evidence, not a historical price
    # series. Keep the slope as a diagnostic only and use explicit scenario
    # assumptions for forward price paths.
    price_assumed_slope = -0.18  # Base scenario; conservative/aggressive values are -0.08/-0.30.

    lmarena = lmarena.copy()
    lmarena["access_class"] = [access_from_text(n, o, lic) for n, o, lic in zip(lmarena.get("model_name", ""), lmarena.get("organization", ""), lmarena.get("license", ""))]
    # Measure the open-vs-closed gap on the most recent snapshot only; mixing
    # historical boards would compare ratings on drifted Elo scales.
    dated_arena = lmarena.dropna(subset=["leaderboard_publish_date"])
    if not dated_arena.empty:
        latest_date = dated_arena["leaderboard_publish_date"].max()
        lmarena = lmarena[lmarena["leaderboard_publish_date"].eq(latest_date)]
    best_open = numeric(lmarena.loc[lmarena["access_class"].isin(["open_weight", "likely_open_weight"]), "rating"]).max()
    best_closed = numeric(lmarena.loc[lmarena["access_class"].eq("closed_or_api"), "rating"]).max()
    open_gap = float(best_closed - best_open) if np.isfinite(best_open) and np.isfinite(best_closed) else 45.0

    # Concentration is measured, not simulated: a Herfindahl index over the
    # composite-index shares of the current snapshot.  No horizon growth term.
    composite = numeric(company_scores["frontier_momentum_heuristic_index"]).clip(lower=0)
    signal_concentration = float((composite / composite.sum()).pow(2).sum()) if float(composite.sum()) > 0 else np.nan
    p90_substitution = float(job_scores["substitution_pressure_index"].quantile(0.90) / 100)

    scenarios = {
        "conservative": {"compute": 0.55, "context": 0.45, "price_slope": -0.08, "gap": 0.45, "compute_cap": 80, "context_cap": 16},
        "base": {"compute": 1.00, "context": 1.00, "price_slope": -0.18, "gap": 1.00, "compute_cap": 400, "context_cap": 64},
        "aggressive": {"compute": 1.45, "context": 1.50, "price_slope": -0.30, "gap": 1.35, "compute_cap": 1200, "context_cap": 128},
    }
    rows = []
    diagnostics = [
        {
            "series": "epoch_training_compute_upper_tail",
            "source_table": "epoch_models_normalized",
            "years_used": len(compute_history),
            "fit_start_year": int(compute_history["year"].min()) if not compute_history.empty else None,
            "fit_end_year": int(compute_history["year"].max()) if not compute_history.empty else None,
            "raw_log10_slope_per_year": compute_slope,
            "fallback_or_cap_policy": "Raw extrapolation capped by scenario compute_cap.",
        },
        {
            "series": "openrouter_context_window_upper_tail",
            "source_table": "openrouter_models_catalog",
            "years_used": len(context_history),
            "fit_start_year": int(context_history["year"].min()) if not context_history.empty else None,
            "fit_end_year": int(context_history["year"].max()) if not context_history.empty else None,
            "raw_log10_slope_per_year": context_slope,
            "fallback_or_cap_policy": "Raw extrapolation capped by scenario context_cap.",
        },
        {
            "series": "openrouter_output_price_lower_quintile",
            "source_table": "openrouter_models_catalog",
            "years_used": len(price_yearly),
            "fit_start_year": int(price_yearly["release_year"].min()) if not price_yearly.empty else None,
            "fit_end_year": int(price_yearly["release_year"].max()) if not price_yearly.empty else None,
            "raw_log10_slope_per_year": price_slope,
            "observed_log10_slope_per_year": np.nan,
            "scenario_assumed_log10_slope_per_year": price_assumed_slope,
            "fallback_or_cap_policy": "Current catalog grouped by release cohort is not a historical price series; no fabricated fallback series is used when the fit window is too short. Forward paths use explicit -0.08/-0.18/-0.30 log10 scenario assumptions regardless of the diagnostic slope. Price factor floors at 0.05.",
        },
    ]
    for scenario, mult in scenarios.items():
        for horizon in [2, 5, 10]:
            raw_compute_gain = 10 ** (max(compute_slope, 0.0) * horizon * mult["compute"])
            raw_context_gain = 10 ** (max(context_slope, 0.0) * horizon * mult["context"])
            compute_gain, compute_capped = capped_growth(raw_compute_gain, mult["compute_cap"])
            context_gain, context_capped = capped_growth(raw_context_gain, mult["context_cap"])
            scenario_price_slope = float(mult["price_slope"])
            price_factor = max(0.05, 10 ** (scenario_price_slope * horizon))
            open_gap_remaining = max(0, open_gap * (1 - min(0.92, 0.12 * horizon * mult["gap"])))
            contact_assumption = TASK_CONTACT_ASSUMPTIONS[scenario][horizon]
            labor_tasks = min(contact_assumption, p90_substitution) if np.isfinite(p90_substitution) else contact_assumption
            rows.extend(
                [
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "frontier_training_compute_multiplier",
                        "value": round(compute_gain, 2),
                        "unit": "x current frontier trend",
                        "method": f"capped scenario from Epoch upper-tail slope; raw={raw_compute_gain:.2f}x capped={compute_capped}",
                    },
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "frontier_context_window_multiplier",
                        "value": round(context_gain, 2),
                        "unit": "x current API catalog trend",
                        "method": f"capped scenario from OpenRouter upper-tail slope; raw={raw_context_gain:.2f}x capped={context_capped}",
                    },
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "frontier_output_price_factor",
                        "value": round(price_factor, 4),
                        "unit": "fraction of current low-price frontier API output cost",
                        "method": f"Explicit price-decline scenario; current-catalog release-cohort diagnostic={price_slope:.3f}, scenario_assumed={scenario_price_slope:.3f}",
                    },
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "open_weight_lmarena_gap_remaining",
                        "value": round(open_gap_remaining, 1),
                        "unit": "arena rating points",
                        "method": "current open vs closed LMArena gap with scenario-specific closure speed",
                    },
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "share_of_us_occupation_tasks_materially_touched",
                        "value": round(labor_tasks, 3),
                        "unit": "share of task-weighted occupation activity",
                        "method": f"Explicit scenario task-contact assumption ({contact_assumption:.2f} at {horizon}y), capped by observed p90 substitution pressure; not a transition-model estimate.",
                    },
                    {
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "metric": "frontier_signal_concentration_hhi",
                        "value": round(signal_concentration, 3) if np.isfinite(signal_concentration) else np.nan,
                        "unit": "Herfindahl index of composite-signal shares (0-1)",
                        "method": "Sum of squared frontier-momentum composite shares across families in this snapshot; scale-free concentration diagnostic with no horizon dynamics.",
                    },
                ]
            )

    forecasts = pd.DataFrame(rows)
    history_rows = []
    for _, row in compute_history.iterrows():
        history_rows.append({"series": "epoch_training_compute_upper_tail", "year": int(row["year"]), "value": row["value"]})
    for _, row in context_history.iterrows():
        history_rows.append({"series": "openrouter_context_window_upper_tail", "year": int(row["year"]), "value": row["value"]})
    for _, row in price_yearly.iterrows():
        history_rows.append({"series": "openrouter_output_price_lower_quintile", "year": int(row["release_year"]), "value": row["output_usd_per_1m"]})
    history = pd.DataFrame(history_rows)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "company-next-best-model",
                "claim": f"{company_scores.iloc[0]['family']} has the strongest composite signal for near-term frontier leadership, but the top open-weight ecosystem score is not necessarily the same family.",
                "evidence": "Composite of LMArena, SWE-bench, OpenRouter, Epoch, Hugging Face, GitHub and OpenAlex indicators.",
                "confidence": "medium",
            },
            {
                "claim_id": "jobs-augmentation-not-total-replacement",
                "claim": "The labor signal is broad task contact, not full-job deletion: high-exposure occupations still retain bottlenecks from trust, regulation, physical work and accountability.",
                "evidence": "Anthropic Economic Index occupation exposure joined to O*NET task text, task collaboration modes and wage/job metadata.",
                "confidence": "medium-high",
            },
            {
                "claim_id": "open-source-catchup",
                "claim": "Open-weight systems look structurally advantaged on ecosystem and cost but still need repeated frontier jumps to erase closed/API benchmark gaps.",
                "evidence": "OpenRouter price fields, Hugging Face downloads/files, LMArena access-class split and Epoch open-weight release metadata.",
                "confidence": "medium",
            },
            {
                "claim_id": "ten-year-forecast",
                "claim": "The 10-year question is less whether AI touches most cognitive workflows and more whether institutions redesign jobs around verification, liability and human preference.",
                "evidence": "Scenario table combines capability trend, price decline, observed task exposure and bottleneck scoring.",
                "confidence": "speculative",
            },
        ]
    )

    write_table(pd.DataFrame(diagnostics), "forecast_input_diagnostics")
    return write_table(forecasts, "capability_forecasts"), write_table(history, "capability_frontier_history"), write_table(claims, "forecast_claims")


def livecodebench_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cached_text("livecodebench_space_api", LIVE_CODE_BENCH_API, overwrite=overwrite_sources)
    js = cached_text("livecodebench_leaderboard_js", LIVE_CODE_BENCH_SPACE, overwrite=overwrite_sources, timeout=180)
    match = re.search(r"Ji=JSON\.parse\('(.+?)'\),", js)
    if not match:
        return rows
    payload = json.loads(bytes(match.group(1), "utf-8").decode("unicode_escape"))
    performances = pd.DataFrame(payload.get("performances", []))
    models = pd.DataFrame(payload.get("models", []))
    if performances.empty or models.empty:
        return rows
    models["release_date_iso"] = models["release_date"].map(parse_any_date)
    release_map = dict(zip(models["model_repr"], models["release_date_iso"]))
    link_map = dict(zip(models["model_repr"], models.get("link", pd.Series("", index=models.index))))
    performances["pass@1"] = numeric(performances["pass@1"])
    grouped = performances.groupby(["model", "difficulty"], dropna=False).agg(
        pass_at_1=("pass@1", "mean"),
        questions=("question_id", "nunique"),
        first_problem_date=("date", "min"),
        last_problem_date=("date", "max"),
    ).reset_index()
    overall = performances.groupby("model", dropna=False).agg(
        pass_at_1=("pass@1", "mean"),
        questions=("question_id", "nunique"),
        first_problem_date=("date", "min"),
        last_problem_date=("date", "max"),
    ).reset_index()
    overall["difficulty"] = "all"
    grouped = pd.concat([overall, grouped], ignore_index=True)
    for _, row in grouped.iterrows():
        model = row["model"]
        append_domain_row(
            rows,
            source_id="livecodebench_leaderboard",
            source_name="LiveCodeBench leaderboard",
            benchmark="LiveCodeBench code generation",
            domain="software_engineering",
            task=f"code_generation_{row['difficulty']}",
            model_name=model,
            score=row["pass_at_1"],
            score_unit="percent_pass_at_1",
            source_url=link_map.get(model) or "https://livecodebench.github.io/",
            eval_date=release_map.get(model) or parse_any_date(row["last_problem_date"]),
            evidence_level="observed",
            sample_size=row["questions"],
            benchmark_weight=1.15 if row["difficulty"] == "all" else 0.72,
            limitations="LiveCodeBench is coding-specific and time-windowed; model release dates are used when present.",
            date_provenance="model_release_or_benchmark_window",
            temporal_eligible=bool(release_map.get(model) or parse_any_date(row["last_problem_date"])),
        )
    return rows


def open_medical_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    listing = json.loads(cached_text("open_medical_results_api", OPEN_MEDICAL_RESULTS_API, overwrite=overwrite_sources, timeout=120))
    siblings = [item.get("rfilename", "") for item in listing.get("siblings", []) if str(item.get("rfilename", "")).endswith(".json")]
    for filename in siblings:
        raw_path = RAW_DOMAIN / "open_medical_results" / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists() and not overwrite_sources:
            text = raw_path.read_text(encoding="utf-8")
        else:
            url = f"{OPEN_MEDICAL_RESULTS_RESOLVE}/{quote(filename)}"
            response = requests.get(url, headers={"User-Agent": "frontier-ai-domain-benchmark-analysis/0.1"}, timeout=90)
            if not response.ok:
                continue
            text = response.text
            raw_path.write_text(text, encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        config = payload.get("config", {})
        model_name = config.get("model_name", filename.split("/results_", 1)[0])
        eval_date_match = re.search(r"results[_\w/.-]*?(\d{4}[-_]\d{2}[-_]\d{2}(?:[ T_]\d{2}[:_-]\d{2}[:_-]\d{2})?)", filename)
        eval_date = eval_date_match.group(1).replace("_", "-") if eval_date_match else listing.get("lastModified", "")
        for benchmark, metrics in payload.get("results", {}).items():
            if not isinstance(metrics, dict):
                continue
            metric_name = next((key for key in ["acc,none", "exact_match,none", "score"] if key in metrics), next(iter(metrics), ""))
            score = metrics.get(metric_name)
            append_domain_row(
                rows,
                source_id="open_medical_llm_leaderboard",
                source_name="Open Medical-LLM Leaderboard",
                benchmark=benchmark,
                domain="medicine",
                task=benchmark.replace("mmlu_", "mmlu medical: "),
                model_name=model_name,
                score=score,
                score_unit="fraction_accuracy",
                source_url="https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard",
                eval_date=eval_date,
                evidence_level="observed",
                sample_size=np.nan,
                benchmark_weight=1.10 if benchmark in {"medqa_4options", "medmcqa", "pubmedqa"} else 0.82,
                limitations="Medical QA accuracy is not a clinical safety or deployment-readiness score.",
                date_provenance="evaluation_filename" if eval_date_match else "dataset_snapshot",
                temporal_eligible=bool(eval_date_match),
            )
    return rows


def terminal_bench_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = cached_text("terminal_bench_2_0_leaderboard", TERMINAL_BENCH_20_URL, overwrite=overwrite_sources, timeout=120)
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S):
        cells = [strip_html(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)]
        if cells and cells[0] == "":
            cells = cells[1:]
        if len(cells) < 7:
            continue
        rank, agent, model, date, agent_org, model_org, accuracy = cells[:7]
        score_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", accuracy)
        if not score_match:
            continue
        append_domain_row(
            rows,
            source_id="terminal_bench_2_0",
            source_name="Terminal-Bench 2.0 leaderboard",
            benchmark="Terminal-Bench 2.0",
            domain="agentic_terminal",
            task=agent,
            model_name=model,
            vendor=model_org,
            score=float(score_match.group(1)),
            score_unit="percent_accuracy",
            source_url=TERMINAL_BENCH_20_URL,
            eval_date=date,
            evidence_level="observed",
            sample_size=89,
            benchmark_weight=1.18,
            limitations="Agent, scaffold and model are entangled; do not attribute the whole score to model weights alone.",
            date_provenance="leaderboard_submission_date",
        )
    return rows


def finance_benchmark_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    listing = json.loads(cached_text("financebench_results_api", FINANCEBENCH_RESULTS_API, overwrite=overwrite_sources, timeout=90))
    for split in ["train", "valid", "test"]:
        filename = f"data/{split}-00000-of-00001.parquet"
        url = f"{FINANCEBENCH_RESOLVE}/{filename}"
        try:
            blob = cached_bytes(f"financebench_results_{split}", url, overwrite=overwrite_sources, timeout=90)
            frame = pd.read_parquet(io.BytesIO(blob))
        except Exception:
            continue
        if {"org", "model", "average"}.issubset(frame.columns):
            for _, row in frame.iterrows():
                append_domain_row(
                    rows,
                    source_id="financebench_results",
                    source_name="FinanceBench public results",
                    benchmark="FinanceBench",
                    domain="finance_quant",
                    task=split,
                    model_name=row["model"],
                    vendor=row.get("org", "financebench"),
                    score=row["average"],
                    score_unit="fraction_average",
                    source_url="https://huggingface.co/datasets/financebench/results",
                    eval_date=listing.get("lastModified", ""),
                    evidence_level="observed",
                    sample_size=len(frame),
                    benchmark_weight=0.82,
                    limitations="Small public result set; useful as a finance RAG signal, not a full domain trend.",
                    date_provenance="dataset_snapshot",
                    temporal_eligible=False,
                )
    return rows


def qfbench_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = cached_text("qfbench_homepage", QFBENCH_URL, overwrite=overwrite_sources, timeout=120)
    cards = re.findall(
        r"#<!-- -->(\d+).*?text-\[11px\][^>]*>([^<]+)</span>.*?text-\[15px\][^>]*>(.*?)</p>.*?via <!-- -->(.*?)</p>.*?>([0-9]+(?:\.[0-9]+)?)<!-- -->%</span>.*?pass@3 <!-- -->([0-9]+(?:\.[0-9]+)?)",
        page,
        flags=re.S,
    )
    for rank, date, model, agent, pass1, pass3 in cards[:30]:
        append_domain_row(
            rows,
            source_id="qfbench_v11",
            source_name="QFBench V11 leaderboard",
            benchmark="QFBench V11",
            domain="finance_quant",
            task=f"{strip_html(agent)} pass@1",
            model_name=strip_html(model),
            score=float(pass1),
            score_unit="percent_pass_at_1",
            source_url=QFBENCH_URL,
            eval_date=date,
            evidence_level="observed",
            sample_size=87,
            benchmark_weight=1.05,
            limitations="Agent and model are bundled; benchmark targets quantitative finance coding rather than all financial work.",
            date_provenance="leaderboard_submission_date",
        )
        append_domain_row(
            rows,
            source_id="qfbench_v11",
            source_name="QFBench V11 leaderboard",
            benchmark="QFBench V11 pass@3",
            domain="finance_quant",
            task=f"{strip_html(agent)} pass@3",
            model_name=strip_html(model),
            score=float(pass3),
            score_unit="percent_pass_at_3",
            source_url=QFBENCH_URL,
            eval_date=date,
            evidence_level="observed",
            sample_size=87,
            benchmark_weight=0.65,
            limitations="pass@3 captures recovery from repeated attempts; it is not one-shot productivity.",
            date_provenance="leaderboard_submission_date",
        )
    return rows


def legal_benchmark_rows(overwrite_sources: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = cached_text("lexometrica_legal_benchmark", LEXOMETRICA_URL, overwrite=overwrite_sources, timeout=90)
    text = strip_html(page)
    ranking_pattern = re.compile(
        r"(\d+)\s+([A-Za-z. ]+?)\s+([A-Za-z0-9.\-+ ]+?)\s+(0\.\d+|1\.00)\s+(\d+)%\s+(\d+)%\s+(0\.\d+|1\.00)"
    )
    for rank, provider, model, primary, safety, citations, composite in ranking_pattern.findall(text):
        if int(rank) > 30:
            continue
        append_domain_row(
            rows,
            source_id="lexometrica_legal_ru_v1",
            source_name="Lexometrica Ground Truth LegalBench RU",
            benchmark="Lexometrica legal-ru-v1 composite",
            domain="legal_reasoning",
            task="IRAC legal reasoning composite",
            model_name=model,
            vendor=provider,
            score=float(composite),
            score_unit="fraction_composite",
            source_url=LEXOMETRICA_URL,
            eval_date="2026-03-01",
            evidence_level="observed",
            sample_size=30,
            benchmark_weight=0.95,
            limitations="Russian legal domain; black-box task set and jurisdiction-specific results.",
            date_provenance="benchmark_release_date",
        )
        append_domain_row(
            rows,
            source_id="lexometrica_legal_ru_v1",
            source_name="Lexometrica Ground Truth LegalBench RU",
            benchmark="Lexometrica legal-ru-v1 citations",
            domain="legal_reasoning",
            task="citation validity",
            model_name=model,
            vendor=provider,
            score=float(citations),
            score_unit="percent_citations_ok",
            source_url=LEXOMETRICA_URL,
            eval_date="2026-03-01",
            evidence_level="observed",
            sample_size=30,
            benchmark_weight=0.55,
            limitations="Citation form is not the same as legal correctness.",
            date_provenance="benchmark_release_date",
        )
    return rows


def local_domain_benchmark_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    lmarena = read_csv_table("lmarena_full").copy()
    if not lmarena.empty:
        lmarena["rating"] = numeric(lmarena["rating"])
        lmarena["domain"] = [infer_capability_domain(category) for category in lmarena.get("category", "")]
        # Arena rows are historical leaderboard snapshots.  Normalize within a
        # snapshot, not over the entire archive, otherwise later snapshots and
        # frequently listed models dominate the scale.
        lmarena["score_normalized_local"] = (
            lmarena.groupby(["category", "leaderboard_publish_date"])["rating"].rank(method="average", pct=True) * 100
        )
        sample = lmarena.groupby(["model_name", "organization", "category", "leaderboard_publish_date", "domain"], dropna=False).agg(
            rating=("rating", "max"),
            score_norm=("score_normalized_local", "max"),
            vote_count=("vote_count", "max"),
            source_url=("source_url", "first"),
        ).reset_index()
        for _, row in sample.iterrows():
            append_domain_row(
                rows,
                source_id="lmarena_full",
                source_name="LMArena full leaderboard",
                benchmark=f"LMArena {row['category']}",
                domain=row["domain"],
                task=row["category"],
                model_name=row["model_name"],
                vendor=row.get("organization", ""),
                score=row["score_norm"],
                score_unit="percentile_normalized_rating",
                source_url=row.get("source_url", "https://lmarena.ai/"),
                eval_date=row["leaderboard_publish_date"],
                evidence_level="observed",
                sample_size=row.get("vote_count", np.nan),
                benchmark_weight=1.0,
                limitations="Arena preference rating is normalized within category; it is not an absolute accuracy score.",
                date_provenance="leaderboard_snapshot_date",
            )

    livebench = read_csv_table("livebench_judgments").copy()
    if not livebench.empty:
        livebench["score"] = numeric(livebench["score"])
        livebench["eval_date"] = pd.to_datetime(numeric(livebench["tstamp"]), unit="s", errors="coerce", utc=True).dt.date.astype(str)
        grouped = livebench.groupby(["model", "category", "task", "eval_date"], dropna=False).agg(score=("score", "mean"), judgments=("question_id", "count")).reset_index()
        for _, row in grouped.iterrows():
            domain = infer_capability_domain(row["category"], row["task"])
            append_domain_row(
                rows,
                source_id="livebench_judgments",
                source_name="LiveBench judgments",
                benchmark=f"LiveBench {row['category']}",
                domain=domain,
                task=row["task"],
                model_name=row["model"],
                score=row["score"],
                score_unit="fraction_judgment_score",
                source_url="https://huggingface.co/datasets/livebench/model_judgment",
                eval_date=row["eval_date"],
                evidence_level="observed",
                sample_size=row["judgments"],
                benchmark_weight=1.0,
                limitations="Judgment rows are task-level and may not reflect a complete model capability profile.",
                date_provenance="judgment_timestamp",
            )

    swe = read_csv_table("swebench_submissions").copy()
    if not swe.empty:
        for _, row in swe.iterrows():
            append_domain_row(
                rows,
                source_id="swebench_submissions",
                source_name="SWE-bench submissions",
                benchmark="SWE-bench",
                domain="software_engineering",
                task=row.get("system_name", "repository issue resolution"),
                model_name=row.get("model"),
                vendor=row.get("org", ""),
                score=row.get("score"),
                score_unit="percent_resolved",
                source_url=row.get("source_url", "https://www.swebench.com/"),
                eval_date="",
                evidence_level="observed",
                sample_size=row.get("total", np.nan),
                benchmark_weight=1.12,
                limitations="Submission-level benchmark; agent harness and scaffolding can affect score.",
                date_provenance="missing",
                temporal_eligible=False,
            )

    openllm = read_csv_table("openllm_leaderboard_metrics_long").copy()
    if not openllm.empty:
        openllm = openllm[~openllm["metric"].astype(str).str.contains("stderr|alias", case=False, na=False)].copy()
        openllm["value"] = numeric(openllm["value"])
        openllm = openllm[openllm["value"].between(0, 1, inclusive="both")]
        grouped = openllm.groupby(["model_name", "model_path", "benchmark"], dropna=False).agg(
            value=("value", "mean"),
            metric_count=("metric", "count"),
            source_url=("source_url", "first"),
        ).reset_index()
        for _, row in grouped.iterrows():
            domain = infer_capability_domain(row["benchmark"])
            append_domain_row(
                rows,
                source_id="open_llm_leaderboard_results",
                source_name="Open LLM Leaderboard results",
                benchmark=row["benchmark"],
                domain=domain,
                task=row["benchmark"].replace("leaderboard_", ""),
                model_name=row["model_name"],
                score=row["value"],
                score_unit="fraction_metric_mean",
                source_url=row.get("source_url", "https://huggingface.co/open-llm-leaderboard"),
                eval_date="",
                evidence_level="observed",
                sample_size=row.get("metric_count", np.nan),
                benchmark_weight=0.78,
                limitations="Open-weight leaderboard rows are broad but not always directly comparable to closed frontier APIs.",
                date_provenance="missing",
                temporal_eligible=False,
            )
    return rows


def build_domain_benchmark_analysis(overwrite_sources: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = local_domain_benchmark_rows()
    for loader in [
        livecodebench_rows,
        open_medical_rows,
        terminal_bench_rows,
        finance_benchmark_rows,
        qfbench_rows,
        legal_benchmark_rows,
    ]:
        try:
            rows.extend(loader(overwrite_sources=overwrite_sources))
        except Exception as exc:
            rows.append(
                {
                    "source_id": loader.__name__,
                    "source_name": loader.__name__,
                    "benchmark": "download_failed",
                    "domain": "science_reasoning",
                    "domain_label": domain_label("science_reasoning"),
                    "task": "download_failed",
                    "model_name": "n/a",
                    "model_family": "Other",
                    "vendor": "n/a",
                    "score": np.nan,
                    "score_unit": "n/a",
                    "score_normalized_0_100": np.nan,
                    "eval_date": "",
                    "source_url": "",
                    "evidence_level": "speculative",
                    "sample_size": np.nan,
                    "benchmark_weight": 0.0,
                    "limitations": f"Loader failed: {exc}",
                    "date_provenance": "missing",
                    "temporal_eligible": False,
                }
            )
    results = pd.DataFrame(rows)
    if results.empty:
        results = pd.DataFrame(columns=["source_id", "source_name", "benchmark", "domain", "domain_label", "task", "model_name", "model_family", "vendor", "score", "score_unit", "score_normalized_0_100", "eval_date", "source_url", "evidence_level", "sample_size", "benchmark_weight", "limitations", "date_provenance", "temporal_eligible"])
    results = filter_public_entities(results)
    results["score_normalized_0_100"] = numeric(results["score_normalized_0_100"]).clip(0, 100)
    results["eval_date_dt"] = pd.to_datetime(results["eval_date"], errors="coerce", utc=True)
    results["eval_year"] = results["eval_date_dt"].dt.year
    reference_ts = pd.Timestamp(REFERENCE_DATE, tz="UTC")
    results["as_of_eligible"] = results["eval_date_dt"].isna() | results["eval_date_dt"].le(reference_ts)
    results["temporal_eligible"] = results.get("temporal_eligible", False).fillna(False).astype(bool) & results["eval_date_dt"].notna() & results["eval_date_dt"].le(reference_ts)
    results = results[results["score_normalized_0_100"].notna()].copy()
    results = results[results["model_family"].ne("Command")].copy()
    results = results[results["as_of_eligible"]].copy()

    effective_keys = ["source_id", "benchmark", "task", "model_name", "eval_date"]
    results["effective_observation_id"] = pd.util.hash_pandas_object(
        results[effective_keys].fillna("").astype(str), index=False
    ).astype(str)

    coverage = results.groupby("domain", as_index=False).agg(
        domain_label=("domain_label", "first"),
        normalized_result_rows=("score_normalized_0_100", "count"),
        effective_observations=("effective_observation_id", "nunique"),
        source_count=("source_id", "nunique"),
        benchmark_count=("benchmark", "nunique"),
        model_count=("model_name", "nunique"),
        family_count=("model_family", "nunique"),
        dated_rows=("eval_date_dt", lambda s: int(s.notna().sum())),
        temporal_rows=("temporal_eligible", "sum"),
        earliest_eval_date=("eval_date_dt", lambda s: s.dropna().min().date().isoformat() if s.notna().any() else ""),
        latest_eval_date=("eval_date_dt", lambda s: s.dropna().max().date().isoformat() if s.notna().any() else ""),
        median_score=("score_normalized_0_100", "median"),
        p90_score=("score_normalized_0_100", lambda s: float(np.quantile(s.dropna(), 0.90)) if s.notna().any() else np.nan),
    )
    coverage["interpretation"] = coverage["domain"].map(lambda d: CAPABILITY_DOMAINS.get(d, {}).get("interpretation", ""))
    coverage["forecast_caveat"] = coverage["domain"].map(lambda d: CAPABILITY_DOMAINS.get(d, {}).get("forecast_caveat", ""))
    coverage["coverage_label"] = np.select(
        [
            coverage["source_count"].ge(3) & coverage["benchmark_count"].ge(3) & coverage["model_count"].ge(25),
            (coverage["source_count"].ge(2) & coverage["benchmark_count"].ge(2)) | (coverage["benchmark_count"].ge(5) & coverage["model_count"].ge(25)),
        ],
        ["broad", "moderate"],
        default="thin",
    )

    # Build one annual observation per model/benchmark, then one frontier value
    # per benchmark.  This prevents historical snapshots and large leaderboards
    # from receiving tens of thousands of implicit votes.
    dated = results[results["temporal_eligible"]].dropna(subset=["eval_year"]).copy()
    dated = dated[(dated["eval_year"] >= 2023) & (dated["eval_year"] <= 2026)]
    dated = dated.sort_values("eval_date_dt").drop_duplicates(
        ["domain", "source_id", "benchmark", "task", "model_name", "eval_year"], keep="last"
    )
    benchmark_year = dated.groupby(["domain", "domain_label", "source_id", "benchmark", "eval_year"], as_index=False).agg(
        frontier_score=("score_normalized_0_100", lambda s: float(np.quantile(s.dropna(), 0.95)) if s.notna().any() else np.nan),
        best_score=("score_normalized_0_100", "max"),
        median_score=("score_normalized_0_100", "median"),
        result_rows=("score_normalized_0_100", "count"),
        benchmark_weight=("benchmark_weight", "median"),
    )
    frontier = benchmark_year.groupby(["domain", "domain_label", "eval_year"], as_index=False).agg(
        frontier_score=("frontier_score", "mean"),
        best_score=("best_score", "max"),
        median_score=("median_score", "median"),
        result_rows=("result_rows", "sum"),
        source_count=("source_id", "nunique"),
        benchmark_count=("benchmark", "nunique"),
    )
    frontier = frontier.sort_values(["domain", "eval_year"])

    velocity_rows = []
    for _, coverage_row in coverage.iterrows():
        domain = coverage_row["domain"]
        group = frontier[frontier["domain"].eq(domain)].copy()
        group = group.dropna(subset=["frontier_score"]).sort_values("eval_year")
        domain_panel = benchmark_year[benchmark_year["domain"].eq(domain)]
        slopes = []
        for _, benchmark_group in domain_panel.groupby(["source_id", "benchmark"]):
            benchmark_group = benchmark_group.sort_values("eval_year")
            if benchmark_group["eval_year"].nunique() >= 2:
                slopes.append(float(np.polyfit(benchmark_group["eval_year"], benchmark_group["frontier_score"], 1)[0]))
        slope = float(np.median(slopes)) if slopes else np.nan
        source = "median_within_benchmark_slope" if slopes else "insufficient_comparable_history"
        latest_by_benchmark = domain_panel.sort_values("eval_year").groupby(["source_id", "benchmark"], as_index=False).tail(1)
        current = float(latest_by_benchmark["frontier_score"].mean()) if len(latest_by_benchmark) else float(coverage_row["p90_score"])
        current = float(np.clip(current, 0, 99.5)) if np.isfinite(current) else 45.0
        velocity_rows.append(
            {
                "domain": domain,
                "domain_label": domain_label(domain),
                "current_frontier_score": round(current, 2),
                "annual_frontier_point_gain_observed": round(slope, 3) if np.isfinite(slope) else np.nan,
                "slope_source": source,
                "years_observed": int(group["eval_year"].nunique()) if len(group) else 0,
                "longitudinal_benchmark_count": len(slopes),
                "source_count": int(coverage_row["source_count"]),
                "benchmark_count": int(coverage_row["benchmark_count"]),
                "model_count": int(coverage_row["model_count"]),
                "coverage_label": coverage_row["coverage_label"],
                "interpretation": coverage_row["interpretation"],
                "forecast_caveat": coverage_row["forecast_caveat"],
            }
        )
    velocity = pd.DataFrame(velocity_rows)
    if not velocity.empty:
        velocity["annual_frontier_point_gain_used"] = numeric(velocity["annual_frontier_point_gain_observed"]).clip(lower=0.0, upper=ANNUAL_GAIN_CAP).fillna(0.0).round(3)
        velocity["forecast_enabled"] = velocity["longitudinal_benchmark_count"].ge(1) & numeric(velocity["annual_frontier_point_gain_observed"]).gt(0)
        velocity["annual_gap_closure_rate_base"] = (
            velocity["annual_frontier_point_gain_used"] / (100 - numeric(velocity["current_frontier_score"]).clip(upper=98.5)).clip(lower=8)
        ).clip(GAP_CLOSURE_RATE_MIN, GAP_CLOSURE_RATE_MAX).round(4)
        velocity["forecast_confidence"] = np.select(
            [
                velocity["coverage_label"].eq("broad") & velocity["longitudinal_benchmark_count"].ge(2),
                velocity["forecast_enabled"],
            ],
            ["medium", "low"],
            default="insufficient_history",
        )
        velocity = velocity.sort_values(["current_frontier_score", "source_count"], ascending=False)

    scenarios = {"conservative": 0.55, "base": 1.0, "aggressive": 1.45}
    forecast_rows = []
    threshold_rows = []
    for _, row in velocity.iterrows():
        current = float(row["current_frontier_score"])
        rate = float(row["annual_gap_closure_rate_base"])
        enabled = bool(row["forecast_enabled"])
        for scenario, mult in scenarios.items():
            for horizon in [2, 5, 10]:
                forecast = 100 - (100 - current) * math.exp(-(rate * mult) * horizon) if enabled else current
                forecast_rows.append(
                    {
                        "domain": row["domain"],
                        "domain_label": row["domain_label"],
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "forecast_frontier_score": round(min(99.5, forecast), 2),
                        "current_frontier_score": current,
                        "annual_gap_closure_rate": round(rate * mult, 4),
                        "confidence": row["forecast_confidence"],
                        "forecast_enabled": enabled,
                        "method": "Bounded exponential gap-closure scenario from within-benchmark longitudinal trends." if enabled else "No extrapolation: insufficient comparable longitudinal benchmark history.",
                        "caveat": row["forecast_caveat"],
                    }
                )
        for threshold in [80, 90, 95]:
            if current >= threshold:
                years = 0.0
            elif current >= 99.0 or not enabled or rate <= 0:
                years = np.nan
            else:
                years = -math.log((100 - threshold) / max(0.1, 100 - current)) / max(rate, 0.001)
            threshold_rows.append(
                {
                    "domain": row["domain"],
                    "domain_label": row["domain_label"],
                    "threshold_score": threshold,
                    "base_years_to_threshold": round(years, 2) if np.isfinite(years) else np.nan,
                    "estimated_threshold_year": int(2026 + math.ceil(years)) if np.isfinite(years) else np.nan,
                    "confidence": row["forecast_confidence"],
                    "forecast_enabled": enabled,
                    "caveat": row["forecast_caveat"],
                }
            )
    forecasts = pd.DataFrame(forecast_rows)
    thresholds = pd.DataFrame(threshold_rows)

    catalog = coverage[
        [
            "domain",
            "domain_label",
            "normalized_result_rows",
            "effective_observations",
            "source_count",
            "benchmark_count",
            "model_count",
            "family_count",
            "earliest_eval_date",
            "latest_eval_date",
            "temporal_rows",
            "coverage_label",
            "interpretation",
            "forecast_caveat",
        ]
    ].sort_values(["coverage_label", "source_count", "normalized_result_rows"], ascending=[True, False, False])

    return (
        write_table(catalog, "domain_benchmark_catalog"),
        write_table(results.drop(columns=["eval_date_dt"], errors="ignore"), "domain_benchmark_results"),
        write_table(frontier, "domain_capability_frontier"),
        write_table(velocity, "domain_improvement_velocity"),
        write_table(forecasts, "domain_capability_forecasts"),
        write_table(thresholds, "domain_forecast_thresholds"),
    )


def build_historical_analogy_index() -> pd.DataFrame:
    # Every dimension score below is an author-assigned subjective prior, not a
    # measured quantity.  The table is published with that provenance attached
    # so the similarity output can never be mistaken for observed data.
    waves = pd.DataFrame(
        [
            {"wave": "spreadsheets", "period": "1979-1995", "speed": 78, "cost_decline": 62, "generality": 68, "labor_scope": 74, "capital_intensity": 28, "network_effects": 46, "regulatory_friction": 18},
            {"wave": "internet", "period": "1993-2010", "speed": 82, "cost_decline": 76, "generality": 86, "labor_scope": 73, "capital_intensity": 55, "network_effects": 94, "regulatory_friction": 31},
            {"wave": "cloud_saas", "period": "2006-2022", "speed": 72, "cost_decline": 79, "generality": 71, "labor_scope": 58, "capital_intensity": 69, "network_effects": 78, "regulatory_friction": 25},
            {"wave": "smartphones", "period": "2007-2020", "speed": 88, "cost_decline": 54, "generality": 76, "labor_scope": 48, "capital_intensity": 73, "network_effects": 91, "regulatory_friction": 34},
            {"wave": "industrial_robotics", "period": "1961-2020", "speed": 38, "cost_decline": 51, "generality": 29, "labor_scope": 44, "capital_intensity": 90, "network_effects": 22, "regulatory_friction": 47},
            {"wave": "electricity", "period": "1882-1930", "speed": 31, "cost_decline": 71, "generality": 94, "labor_scope": 88, "capital_intensity": 96, "network_effects": 83, "regulatory_friction": 58},
            {"wave": "search_ads", "period": "1998-2015", "speed": 83, "cost_decline": 83, "generality": 62, "labor_scope": 42, "capital_intensity": 61, "network_effects": 96, "regulatory_friction": 22},
            {"wave": "containerization", "period": "1956-1990", "speed": 43, "cost_decline": 88, "generality": 56, "labor_scope": 63, "capital_intensity": 86, "network_effects": 79, "regulatory_friction": 41},
        ]
    )
    ai = np.array([86, 91, 93, 86, 82, 88, 52], dtype=float)
    dims = ["speed", "cost_decline", "generality", "labor_scope", "capital_intensity", "network_effects", "regulatory_friction"]
    matrix = waves[dims].to_numpy(dtype=float)
    similarity = (matrix @ ai) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(ai))
    waves["ai_similarity_score"] = (similarity * 100).round(2)
    waves["evidence_level"] = "speculative"
    waves["input_basis"] = "author_assigned_subjective_prior"
    waves["ai_profile_basis"] = "author_assigned_subjective_prior (frontier-AI vector)"
    waves["interpretation"] = [
        "Best analogy for occupational task rebundling and sudden knowledge-worker productivity jumps.",
        "Best analogy for general-purpose diffusion, platform creation and strange second-order labor demand.",
        "Best analogy for enterprise adoption lags and API-first business-model shift.",
        "Best analogy for consumer pull, app ecosystems and fast behavioral rewiring.",
        "Useful negative analogy: physical deployment is slower and more capital-locked than software AI.",
        "Best analogy for long-run production reorganization, not near-term speed.",
        "Best analogy for advertising-funded discovery and winner-take-most information layers.",
        "Best analogy for cost shock in a hidden infrastructure layer.",
    ]
    return write_table(waves.sort_values("ai_similarity_score", ascending=False), "historical_analogy_index")


def build_open_closed_gap_by_category() -> tuple[pd.DataFrame, pd.DataFrame]:
    lmarena = read_csv_table("lmarena_full")
    lmarena = lmarena.copy()
    lmarena["model_family"] = [family_from_text(n, o) for n, o in zip(lmarena.get("model_name", ""), lmarena.get("organization", ""))]
    lmarena["access_class"] = [access_from_text(n, o, lic) for n, o, lic in zip(lmarena.get("model_name", ""), lmarena.get("organization", ""), lmarena.get("license", ""))]
    lmarena["access_bucket"] = np.where(lmarena["access_class"].isin(["open_weight", "likely_open_weight"]), "open_weight", "closed_or_api")
    lmarena["rating"] = numeric(lmarena["rating"])
    # Compare open vs closed inside each category's most recent snapshot only.
    # Mixing every historical snapshot would let stale boards and repeated
    # listings decide the gap.
    dated = lmarena.dropna(subset=["leaderboard_publish_date"])
    latest_snapshots = dated.sort_values("leaderboard_publish_date").groupby("category", dropna=False).tail(1)[
        ["category", "leaderboard_publish_date"]
    ]
    if not latest_snapshots.empty:
        lmarena = lmarena.merge(latest_snapshots, on=["category", "leaderboard_publish_date"], how="inner")
    grouped = lmarena.groupby(["category", "access_bucket"], as_index=False).agg(
        best_rating=("rating", "max"),
        top10_median_rating=("rating", lambda s: s.dropna().sort_values(ascending=False).head(10).median()),
        model_count=("model_name", "nunique"),
        vote_count=("vote_count", "sum"),
    )
    wide = grouped.pivot(index="category", columns="access_bucket", values="best_rating").reset_index()
    for col in ["closed_or_api", "open_weight"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["open_closed_best_gap"] = (wide["closed_or_api"] - wide["open_weight"]).round(2)
    wide["open_closed_gap_pct_of_closed"] = safe_divide(wide["open_closed_best_gap"], wide["closed_or_api"]).round(4)
    wide["comparison_note"] = np.where(
        wide[["closed_or_api", "open_weight"]].notna().all(axis=1),
        "Open and closed best ratings compared within the category's most recent leaderboard snapshot.",
        "No comparable open-weight or closed/API row in the category's most recent snapshot.",
    )
    family_category = lmarena.sort_values("rating", ascending=False).groupby(["category", "model_family", "access_bucket"], as_index=False).head(1)
    family_category = family_category[
        ["category", "model_family", "access_bucket", "model_name", "organization", "rating", "rank", "vote_count"]
    ].sort_values(["category", "rating"], ascending=[True, False])
    return write_table(wide.sort_values("open_closed_best_gap", ascending=False), "open_closed_gap_by_category"), write_table(family_category, "lmarena_category_leaders")


def build_price_performance_frontier() -> pd.DataFrame:
    openrouter = read_csv_table("openrouter_models_catalog").copy()
    lmarena = read_csv_table("lmarena_full").copy()
    lmarena["model_family"] = [family_from_text(n, o) for n, o in zip(lmarena.get("model_name", ""), lmarena.get("organization", ""))]
    family_rating = lmarena.groupby("model_family", as_index=False).agg(
        family_best_lmarena=("rating", "max"),
        family_top20_lmarena=("rating", lambda s: numeric(s).dropna().sort_values(ascending=False).head(20).median()),
    )
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    openrouter["access_class"] = [
        classify_access(name, " ".join([clean_text(vendor), clean_text(mid)]))
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    work = openrouter.merge(family_rating, on="model_family", how="left")
    work["output_price_clean"] = numeric(work["output_usd_per_1m"]).replace(0, np.nan)
    work["input_price_clean"] = numeric(work["input_usd_per_1m"]).replace(0, np.nan)
    work["blended_price_usd_per_1m"] = (work["input_price_clean"].fillna(work["output_price_clean"]) * 0.35 + work["output_price_clean"].fillna(work["input_price_clean"]) * 0.65)
    work["price_performance_index"] = (
        minmax(work["family_best_lmarena"]) * 0.70
        + minmax(work["context_window"], log=True) * 0.15
        + minmax(work["blended_price_usd_per_1m"], invert=True, log=True) * 0.15
    ).round(2)
    work["quality_per_dollar"] = safe_divide(work["family_best_lmarena"], np.log1p(work["blended_price_usd_per_1m"])).round(2)
    work["quality_proxy_level"] = "family_level_proxy"
    work["quality_proxy_note"] = "LMArena rating is attached at model-family level; do not read this as direct model-level benchmark evidence."

    frontier_flags = []
    candidates = work[["family_best_lmarena", "blended_price_usd_per_1m"]].copy()
    for idx, row in candidates.iterrows():
        rating = row["family_best_lmarena"]
        price = row["blended_price_usd_per_1m"]
        if not np.isfinite(rating) or not np.isfinite(price):
            frontier_flags.append(False)
            continue
        dominates = candidates[
            (candidates["family_best_lmarena"] >= rating)
            & (candidates["blended_price_usd_per_1m"] <= price)
            & ((candidates["family_best_lmarena"] > rating) | (candidates["blended_price_usd_per_1m"] < price))
        ]
        frontier_flags.append(dominates.empty)
    work["price_performance_frontier"] = frontier_flags
    cols = [
        "openrouter_id",
        "canonical_model",
        "vendor",
        "model_family",
        "access_class",
        "release_date",
        "context_window",
        "input_usd_per_1m",
        "output_usd_per_1m",
        "blended_price_usd_per_1m",
        "family_best_lmarena",
        "family_top20_lmarena",
        "quality_proxy_level",
        "quality_proxy_note",
        "price_performance_index",
        "quality_per_dollar",
        "price_performance_frontier",
        "source_url",
    ]
    return write_table(work[cols].sort_values("price_performance_index", ascending=False), "price_performance_frontier")


def benchmark_candidate_tables() -> dict[str, list[dict[str, Any]]]:
    lmarena = read_csv_table("lmarena_full").copy()
    lmarena["family"] = [family_from_text(n, o) for n, o in zip(lmarena.get("model_name", ""), lmarena.get("organization", ""))]
    lmarena["rating"] = numeric(lmarena["rating"])
    lmarena_candidates = []
    for model_name, group in lmarena.sort_values("rating", ascending=False).groupby("model_name", dropna=True):
        best = group.iloc[0]
        lmarena_candidates.append(
            {
                "model_name": model_name,
                "model_id": model_name,
                "family": best.get("family"),
                "benchmark_score": best.get("rating"),
                "benchmark_metric": "lmarena_rating",
                "benchmark_category": best.get("category"),
                "sort_score": best.get("rating"),
                "source_url": best.get("source_url"),
            }
        )

    swe = read_csv_table("swebench_submissions").copy()
    swe["family"] = [family_from_text(m, s, sub) for m, s, sub in zip(swe.get("model", ""), swe.get("system_name", ""), swe.get("submission", ""))]
    swe_candidates = []
    for model_name, group in swe.sort_values("score", ascending=False).groupby("model", dropna=True):
        best = group.iloc[0]
        swe_candidates.append(
            {
                "model_name": model_name,
                "model_id": best.get("submission"),
                "family": best.get("family"),
                "benchmark_score": best.get("score"),
                "benchmark_metric": "swebench_percent_resolved",
                "benchmark_category": "software_engineering",
                "sort_score": best.get("score"),
                "source_url": best.get("source_url"),
            }
        )

    livebench = read_csv_table("livebench_judgments").copy()
    livebench["family"] = [family_from_text(m) for m in livebench.get("model", "")]
    livebench["score"] = numeric(livebench["score"])
    livebench_group = livebench.groupby("model", dropna=True).agg(
        family=("family", "first"),
        benchmark_score=("score", "mean"),
        best_score=("score", "max"),
        categories=("category", lambda s: sentence_join(sorted(set(clean_text(x) for x in s)), 4)),
    ).reset_index()
    livebench_candidates = [
        {
            "model_name": row["model"],
            "model_id": row["model"],
            "family": row["family"],
            "benchmark_score": row["benchmark_score"],
            "benchmark_metric": "livebench_mean_judgment_score",
            "benchmark_category": row["categories"],
            "sort_score": row["best_score"],
            "source_url": "https://huggingface.co/datasets/livebench/model_judgment",
        }
        for _, row in livebench_group.iterrows()
    ]

    openllm = read_csv_table("openllm_leaderboard_metrics_long").copy()
    openllm = openllm[~openllm["metric"].astype(str).str.contains("stderr", case=False, na=False)].copy()
    openllm["value"] = numeric(openllm["value"])
    openllm = openllm[openllm["value"].between(0, 1, inclusive="both")]
    openllm["family"] = [family_from_text(n, p) for n, p in zip(openllm.get("model_name", ""), openllm.get("model_path", ""))]
    openllm_group = openllm.groupby(["model_name", "model_path"], dropna=True).agg(
        family=("family", "first"),
        benchmark_score=("value", "mean"),
        best_score=("value", "max"),
        metrics=("metric", lambda s: sentence_join(sorted(set(clean_text(x) for x in s)), 4)),
        source_url=("source_url", "first"),
    ).reset_index()
    openllm_candidates = [
        {
            "model_name": row["model_name"],
            "model_id": row["model_path"],
            "family": row["family"],
            "benchmark_score": row["benchmark_score"],
            "benchmark_metric": "open_llm_leaderboard_mean_metric",
            "benchmark_category": row["metrics"],
            "sort_score": row["best_score"],
            "source_url": row["source_url"],
        }
        for _, row in openllm_group.iterrows()
    ]

    return {
        "lmarena": lmarena_candidates,
        "swebench": swe_candidates,
        "livebench": livebench_candidates,
        "open_llm_leaderboard": openllm_candidates,
    }


def build_model_benchmark_match_audit() -> pd.DataFrame:
    openrouter = read_csv_table("openrouter_models_catalog").copy()
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    candidate_tables = benchmark_candidate_tables()
    matchers = {source: PreparedModelMatcher(candidates) for source, candidates in candidate_tables.items()}
    rows = []
    for _, model in openrouter.iterrows():
        for source, matcher in matchers.items():
            match = matcher.match(
                model.get("canonical_model"),
                model.get("openrouter_id"),
                clean_text(model.get("model_family")),
            )
            record = match.benchmark_record or {}
            rows.append(
                {
                    "openrouter_id": model.get("openrouter_id"),
                    "canonical_model": model.get("canonical_model"),
                    "vendor": model.get("vendor"),
                    "model_family": model.get("model_family"),
                    "benchmark_source": source,
                    "benchmark_model_name": match.benchmark_model_name,
                    "benchmark_family": match.benchmark_family,
                    "match_confidence": match.confidence,
                    "match_rank": MATCH_CONFIDENCE_ORDER[match.confidence],
                    "match_key": match.match_key,
                    "direct_model_match": match.direct_model_match,
                    "benchmark_score": record.get("benchmark_score"),
                    "benchmark_metric": record.get("benchmark_metric", ""),
                    "benchmark_category": record.get("benchmark_category", ""),
                    "benchmark_source_url": record.get("source_url", ""),
                    "normalized_openrouter_name": normalize_model_name(model.get("canonical_model")),
                    "normalized_benchmark_name": normalize_model_name(match.benchmark_model_name),
                }
            )
    audit = pd.DataFrame(rows).sort_values(["openrouter_id", "match_rank", "benchmark_source"])
    return write_table(audit, "model_benchmark_match_audit")


def build_direct_model_price_performance(match_audit: pd.DataFrame, proxy_frontier: pd.DataFrame) -> pd.DataFrame:
    openrouter = read_csv_table("openrouter_models_catalog").copy()
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    direct = match_audit[match_audit["direct_model_match"].astype(bool)].copy()
    if direct.empty:
        cols = [
            "openrouter_id",
            "canonical_model",
            "vendor",
            "model_family",
            "direct_evidence_sources",
            "direct_match_count",
            "direct_lmarena_rating",
            "direct_price_performance_index",
            "direct_price_performance_frontier",
        ]
        return write_table(pd.DataFrame(columns=cols), "direct_model_price_performance")

    source_summary = direct.groupby("openrouter_id").agg(
        direct_evidence_sources=("benchmark_source", lambda s: ",".join(sorted(set(s)))),
        direct_match_count=("benchmark_source", "nunique"),
        best_direct_match_confidence=("match_rank", "min"),
    ).reset_index()
    lmarena = direct[direct["benchmark_source"].eq("lmarena")].copy()
    lmarena["benchmark_score"] = numeric(lmarena["benchmark_score"])
    lmarena_best = lmarena.sort_values("benchmark_score", ascending=False).groupby("openrouter_id", as_index=False).head(1)
    lmarena_best = lmarena_best[
        [
            "openrouter_id",
            "benchmark_model_name",
            "benchmark_score",
            "benchmark_category",
            "match_confidence",
            "benchmark_source_url",
        ]
    ].rename(
        columns={
            "benchmark_model_name": "direct_lmarena_model_name",
            "benchmark_score": "direct_lmarena_rating",
            "benchmark_category": "direct_lmarena_category",
            "match_confidence": "direct_lmarena_match_confidence",
            "benchmark_source_url": "direct_lmarena_source_url",
        }
    )
    work = openrouter.merge(source_summary, on="openrouter_id", how="inner").merge(lmarena_best, on="openrouter_id", how="left")
    work["output_price_clean"] = numeric(work["output_usd_per_1m"]).replace(0, np.nan)
    work["input_price_clean"] = numeric(work["input_usd_per_1m"]).replace(0, np.nan)
    work["blended_price_usd_per_1m"] = work["input_price_clean"].fillna(work["output_price_clean"]) * 0.35 + work["output_price_clean"].fillna(work["input_price_clean"]) * 0.65
    work["direct_price_performance_index"] = (
        minmax(work["direct_lmarena_rating"]) * 0.76
        + minmax(work["context_window"], log=True) * 0.12
        + minmax(work["blended_price_usd_per_1m"], invert=True, log=True) * 0.12
    ).round(2)
    candidates = work[["direct_lmarena_rating", "blended_price_usd_per_1m"]].copy()
    frontier_flags = []
    for _, row in candidates.iterrows():
        rating = row["direct_lmarena_rating"]
        price = row["blended_price_usd_per_1m"]
        if not np.isfinite(rating) or not np.isfinite(price):
            frontier_flags.append(False)
            continue
        dominates = candidates[
            (candidates["direct_lmarena_rating"] >= rating)
            & (candidates["blended_price_usd_per_1m"] <= price)
            & ((candidates["direct_lmarena_rating"] > rating) | (candidates["blended_price_usd_per_1m"] < price))
        ]
        frontier_flags.append(dominates.empty)
    work["direct_price_performance_frontier"] = frontier_flags
    proxy_cols = proxy_frontier[["openrouter_id", "family_best_lmarena", "price_performance_index"]].rename(
        columns={"family_best_lmarena": "proxy_family_best_lmarena", "price_performance_index": "proxy_price_performance_index"}
    )
    work = work.merge(proxy_cols, on="openrouter_id", how="left")
    work["quality_proxy_level"] = np.where(work["direct_lmarena_rating"].notna(), "direct_model_lmarena", "direct_non_lmarena_benchmark")
    work["evidence_type"] = "direct_match"
    cols = [
        "openrouter_id",
        "canonical_model",
        "vendor",
        "model_family",
        "release_date",
        "context_window",
        "input_usd_per_1m",
        "output_usd_per_1m",
        "blended_price_usd_per_1m",
        "direct_evidence_sources",
        "direct_match_count",
        "best_direct_match_confidence",
        "direct_lmarena_model_name",
        "direct_lmarena_rating",
        "direct_lmarena_category",
        "direct_lmarena_match_confidence",
        "proxy_family_best_lmarena",
        "direct_price_performance_index",
        "proxy_price_performance_index",
        "direct_price_performance_frontier",
        "quality_proxy_level",
        "evidence_type",
        "direct_lmarena_source_url",
        "source_url",
    ]
    return write_table(work[cols].sort_values("direct_price_performance_index", ascending=False), "direct_model_price_performance")


def build_model_price_panel() -> pd.DataFrame:
    openrouter = read_csv_table("openrouter_models_catalog").copy()
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    openrouter["release_date_dt"] = pd.to_datetime(openrouter["release_date"], errors="coerce", utc=True)
    openrouter["release_year"] = openrouter["release_date_dt"].dt.year
    openrouter["input_price_clean"] = numeric(openrouter["input_usd_per_1m"]).replace(0, np.nan)
    openrouter["output_price_clean"] = numeric(openrouter["output_usd_per_1m"]).replace(0, np.nan)
    openrouter["blended_price_usd_per_1m"] = (
        openrouter["input_price_clean"].fillna(openrouter["output_price_clean"]) * 0.45
        + openrouter["output_price_clean"].fillna(openrouter["input_price_clean"]) * 0.55
    )
    modal_cols = [col for col in ["modality", "input_modalities", "output_modalities"] if col in openrouter.columns]
    if modal_cols:
        modal_text = openrouter[modal_cols].astype(str).agg(" ".join, axis=1).str.lower()
        text_like = modal_text.str.contains("text", na=False) | modal_text.str.strip().isin(["", "nan"])
        openrouter = openrouter[text_like].copy()
    openrouter = openrouter[
        openrouter["input_price_clean"].gt(0)
        & openrouter["output_price_clean"].gt(0)
        & openrouter["blended_price_usd_per_1m"].gt(0)
        & openrouter["model_family"].ne("Command")
    ].copy()
    openrouter["price_percentile_rank"] = openrouter["blended_price_usd_per_1m"].rank(pct=True)
    return openrouter


def task_cost_usd(input_tokens: float, output_tokens: float, input_price: float, output_price: float) -> float:
    if not all(np.isfinite(value) for value in [input_tokens, output_tokens, input_price, output_price]):
        return np.nan
    return float(input_tokens) * float(input_price) / 1_000_000 + float(output_tokens) * float(output_price) / 1_000_000


def build_domain_family_quality(domain_results: pd.DataFrame) -> pd.DataFrame:
    work = domain_results.copy()
    work["score_normalized_0_100"] = numeric(work["score_normalized_0_100"]).clip(0, 100)
    work = work[work["score_normalized_0_100"].notna() & work["model_family"].ne("Command")].copy()
    if work.empty:
        return pd.DataFrame(columns=["domain", "model_family", "family_domain_score", "family_domain_best_score", "domain_source_count", "domain_benchmark_count", "domain_result_rows"])
    # Collapse repeated snapshots/model rows inside each benchmark before
    # combining benchmarks.  A 40k-row leaderboard must not outweigh an
    # independent 100-row benchmark simply because it publishes more often.
    keys = ["domain", "model_family", "source_id", "benchmark"]
    grouped = work.groupby(keys, observed=True)
    per_benchmark = grouped.agg(
        benchmark_family_best=("score_normalized_0_100", "max"),
        benchmark_weight=("benchmark_weight", "median"),
        raw_result_rows=("score_normalized_0_100", "count"),
    ).reset_index()
    quantiles = grouped["score_normalized_0_100"].quantile(0.90).rename("benchmark_family_score").reset_index()
    per_benchmark = per_benchmark.merge(quantiles, on=keys, how="left")
    rows = []
    for (domain, family), group in per_benchmark.groupby(["domain", "model_family"]):
        weights = numeric(group["benchmark_weight"]).fillna(1.0).clip(0.25, 1.5)
        rows.append(
            {
                "domain": domain,
                "model_family": family,
                "family_domain_score": round(float(np.average(group["benchmark_family_score"], weights=weights)), 3),
                "family_domain_best_score": round(float(group["benchmark_family_best"].max()), 3),
                "domain_source_count": int(group["source_id"].nunique()),
                "domain_benchmark_count": int(group["benchmark"].nunique()),
                "domain_result_rows": int(group["raw_result_rows"].sum()),
                "effective_benchmark_observations": int(len(group)),
                "aggregation_method": "benchmark-first weighted mean; repeated rows collapsed within benchmark",
            }
        )
    return pd.DataFrame(rows)


def build_cost_external_evidence() -> pd.DataFrame:
    rows = [
        {
            "source_id": "openrouter_models_api",
            "name": "OpenRouter Models API",
            "url": "https://openrouter.ai/docs/guides/overview/models",
            "used_for": "Current model price, context-window and modality catalog fields.",
            "evidence_note": "The API exposes model identifiers, context length and lowest pricing fields; local dataset snapshots normalize those fields into per-million-token prices.",
        },
        {
            "source_id": "epoch_llm_inference_price_trends",
            "name": "Epoch AI LLM inference price trends",
            "url": "https://epoch.ai/data-insights/llm-inference-price-trends/",
            "used_for": "External prior that quality-adjusted inference prices can fall faster than raw frontier price lists.",
            "evidence_note": "Used as scenario context, not as a hidden numeric override of the local catalog.",
        },
        {
            "source_id": "price_of_progress_arxiv_2511_23455",
            "name": "The Price of Progress: Algorithmic Efficiency and the Falling Cost of AI Inference",
            "url": "https://arxiv.org/abs/2511.23455",
            "used_for": "Quality-adjusted fixed-task cost-decline prior.",
            "evidence_note": "Supports separating fixed benchmark/task cost from the average cost of frontier workloads.",
        },
        {
            "source_id": "agentic_token_consumption_arxiv_2604_22750",
            "name": "How Do AI Agents Spend Your Money?",
            "url": "https://arxiv.org/abs/2604.22750",
            "used_for": "Token-amplification caveat for agentic coding and multi-step workflows.",
            "evidence_note": "Used to justify tracking message/task workload complexity separately from per-token prices.",
        },
        {
            "source_id": "price_reversal_arxiv_2603_23971",
            "name": "The Price Reversal Phenomenon",
            "url": "https://arxiv.org/abs/2603.23971",
            "used_for": "Caveat that listed price can be a weak proxy for realized cost when thinking tokens and retry variance differ by model.",
            "evidence_note": "The analysis therefore reports token-budget assumptions, not only model list prices.",
        },
        {
            "source_id": "anthropic_economic_index_arxiv_2511_15080",
            "name": "Anthropic Economic Index report: Uneven geographic and enterprise AI adoption",
            "url": "https://arxiv.org/abs/2511.15080",
            "used_for": "External support for rising directive delegation and more autonomous AI task use.",
            "evidence_note": "Used as narrative context for workload mix; the local labor/task tables remain the primary quantitative input.",
        },
    ]
    return write_table(pd.DataFrame(rows), "cost_external_evidence")


def select_task_candidate(candidates: pd.DataFrame, score_col: str, cost_col: str, threshold: float) -> tuple[pd.Series, str]:
    scored = candidates[candidates[score_col].notna() & candidates[cost_col].notna()].copy()
    adequate = scored[scored[score_col].ge(threshold)].copy()
    if not adequate.empty:
        return adequate.sort_values([cost_col, score_col], ascending=[True, False]).iloc[0], "adequate"
    if not scored.empty:
        return scored.sort_values([score_col, cost_col], ascending=[False, True]).iloc[0], "best_available_below_threshold"
    return pd.Series(dtype=object), "no_candidate"


def build_llm_cost_task_analysis(
    forecasts: pd.DataFrame,
    domain_results: pd.DataFrame,
    domain_velocity: pd.DataFrame,
    domain_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_panel = build_model_price_panel()
    domain_quality = build_domain_family_quality(domain_results)

    message_rows = []
    message_component_rows = []
    for year, mix in MESSAGE_MIX_BY_YEAR.items():
        available = price_panel[
            price_panel["release_year"].notna()
            & price_panel["release_year"].le(year)
            & price_panel["release_year"].ge(2020)
        ].copy()
        if available.empty:
            available = price_panel.copy()
        weighted_cost = 0.0
        weighted_tokens = 0.0
        for profile in MESSAGE_WORKLOAD_PROFILES:
            share = float(mix.get(profile["profile"], 0.0))
            q = float(profile["price_quantile"])
            input_price = float(available["input_price_clean"].quantile(q))
            output_price = float(available["output_price_clean"].quantile(q))
            cost = task_cost_usd(profile["input_tokens"], profile["output_tokens"], input_price, output_price)
            tokens = profile["input_tokens"] + profile["output_tokens"]
            weighted_cost += share * cost
            weighted_tokens += share * tokens
            message_component_rows.append(
                {
                    "year": year,
                    "profile": profile["profile"],
                    "display_name": profile["display_name"],
                    "complexity_label": profile["complexity_label"],
                    "mix_share": share,
                    "input_tokens": profile["input_tokens"],
                    "output_tokens": profile["output_tokens"],
                    "price_quantile": q,
                    "input_price_usd_per_1m": round(input_price, 6),
                    "output_price_usd_per_1m": round(output_price, 6),
                    "profile_cost_usd": round(cost, 6),
                    "weighted_cost_contribution_usd": round(share * cost, 6),
                    "method": "Modeled workload mix over current listed-price cohorts; not observed invoice data.",
                }
            )
        message_rows.append(
            {
                "year": year,
                "released_model_count": int(len(available)),
                "low_cost_blended_price_usd_per_1m": round(float(available["blended_price_usd_per_1m"].quantile(0.20)), 6),
                "median_blended_price_usd_per_1m": round(float(available["blended_price_usd_per_1m"].quantile(0.50)), 6),
                "frontier_blended_price_usd_per_1m": round(float(available["blended_price_usd_per_1m"].quantile(0.90)), 6),
                "simple_chat_share": mix.get("simple_chat", 0.0),
                "knowledge_work_share": mix.get("knowledge_work_message", 0.0),
                "long_context_share": mix.get("long_context_analysis", 0.0),
                "agentic_workflow_share": mix.get("agentic_workflow", 0.0),
                "average_effective_tokens": round(weighted_tokens, 0),
                "modeled_average_message_cost_usd": round(weighted_cost, 6),
                "message_cost_index_2023_100": np.nan,
                "assumption_basis": "authored workload-mix shares over listed-price cohorts (survivor-biased catalog)",
                "method": "Weighted scenario mix of simple chat, knowledge work, long-context analysis and agentic workflow runs.",
            }
        )
    message_trends = pd.DataFrame(message_rows)
    base_2023 = float(message_trends.loc[message_trends["year"].eq(2023), "modeled_average_message_cost_usd"].iloc[0]) if not message_trends.empty else np.nan
    if np.isfinite(base_2023) and base_2023 > 0:
        message_trends["message_cost_index_2023_100"] = (message_trends["modeled_average_message_cost_usd"] / base_2023 * 100).round(1)
    message_components = pd.DataFrame(message_component_rows)

    current_rows = []
    candidate_rows = []
    forecast_rows = []
    domain_velocity_index = domain_velocity.set_index("domain") if not domain_velocity.empty else pd.DataFrame()
    for profile in FIXED_TASK_PROFILES:
        domain = profile["domain"]
        quality = domain_quality[domain_quality["domain"].eq(domain)].copy()
        candidates = price_panel.merge(quality, on="model_family", how="inner")
        candidates = candidates[~candidates["model_family"].isin(["Other", "unknown", ""])].copy()
        context_needed = profile["input_tokens"] + profile["output_tokens"]
        candidates = candidates[numeric(candidates.get("context_window", pd.Series(index=candidates.index))).fillna(0).ge(context_needed)].copy()
        candidates["current_task_cost_usd"] = [
            task_cost_usd(profile["input_tokens"], profile["output_tokens"], inp, out)
            for inp, out in zip(candidates["input_price_clean"], candidates["output_price_clean"])
        ]
        candidates["current_quality_gap"] = profile["required_domain_score"] - candidates["family_domain_score"]
        candidates["current_adequacy_status"] = np.where(candidates["family_domain_score"].ge(profile["required_domain_score"]), "adequate", "below_threshold")
        candidates = candidates.sort_values(["current_adequacy_status", "current_task_cost_usd", "family_domain_score"], ascending=[True, True, False])
        top_candidates = candidates.sort_values(
            ["current_adequacy_status", "current_task_cost_usd", "family_domain_score"],
            ascending=[True, True, False],
        ).head(20)
        for _, row in top_candidates.iterrows():
            candidate_rows.append(
                {
                    "task_profile": profile["task_profile"],
                    "display_name": profile["display_name"],
                    "domain": domain,
                    "domain_label": domain_label(domain),
                    "required_domain_score": profile["required_domain_score"],
                    "openrouter_id": row["openrouter_id"],
                    "canonical_model": row["canonical_model"],
                    "vendor": row["vendor"],
                    "model_family": row["model_family"],
                    "context_window": row.get("context_window"),
                    "input_usd_per_1m": row["input_price_clean"],
                    "output_usd_per_1m": row["output_price_clean"],
                    "family_domain_score": row["family_domain_score"],
                    "current_quality_gap": round(float(row["current_quality_gap"]), 3),
                    "current_task_cost_usd": round(float(row["current_task_cost_usd"]), 6),
                    "current_adequacy_status": row["current_adequacy_status"],
                    "evidence_level": "family_proxy",
                    "method": "OpenRouter prices joined to domain benchmark family scores; candidate must fit task token budget.",
                }
            )
        selected, status = select_task_candidate(candidates, "family_domain_score", "current_task_cost_usd", profile["required_domain_score"])
        if selected.empty:
            continue
        current_cost = float(selected["current_task_cost_usd"])
        current_rows.append(
            {
                "task_profile": profile["task_profile"],
                "display_name": profile["display_name"],
                "domain": domain,
                "domain_label": domain_label(domain),
                "scenario": "current",
                "horizon_years": 0,
                "target_year": 2026,
                "selected_model": selected["canonical_model"],
                "selected_family": selected["model_family"],
                "selected_vendor": selected["vendor"],
                "required_domain_score": profile["required_domain_score"],
                "selected_domain_score": round(float(selected["family_domain_score"]), 3),
                "forecast_task_cost_usd": round(current_cost, 6),
                "current_task_cost_usd": round(current_cost, 6),
                "cost_factor_vs_current": 1.0,
                "adequacy_status": status,
                "input_tokens": profile["input_tokens"],
                "output_tokens": profile["output_tokens"],
                "human_gate": profile["human_gate"],
                "method": "Cheapest current model family meeting the task-domain score threshold; family-level quality proxy.",
            }
        )
        current_frontier = (
            float(domain_velocity_index.loc[domain, "current_frontier_score"])
            if domain in domain_velocity_index.index and pd.notna(domain_velocity_index.loc[domain, "current_frontier_score"])
            else float(candidates["family_domain_score"].max())
        )
        for scenario, prior in QUALITY_ADJUSTED_COST_PRIORS.items():
            for horizon in [2, 5, 10]:
                domain_match = domain_forecasts[
                    domain_forecasts["domain"].eq(domain)
                    & domain_forecasts["scenario"].eq(scenario)
                    & domain_forecasts["horizon_years"].eq(horizon)
                ]
                forecast_score = (
                    float(domain_match["forecast_frontier_score"].iloc[0])
                    if not domain_match.empty
                    else min(99.5, current_frontier + horizon * 3.0)
                )
                gain = max(0.0, forecast_score - current_frontier)
                work = candidates.copy()
                rank = work["current_task_cost_usd"].rank(pct=True).fillna(1.0)
                catchup_multiplier = (0.70 + (1 - rank) * 0.45).clip(0.65, 1.15)
                work["forecast_domain_score"] = (numeric(work["family_domain_score"]) + gain * catchup_multiplier).clip(0, 99.5)
                quality_factor = max(0.025, float(prior["annual_factor"]) ** horizon)
                work["forecast_task_cost_usd"] = work["current_task_cost_usd"] * quality_factor
                future, future_status = select_task_candidate(work, "forecast_domain_score", "forecast_task_cost_usd", profile["required_domain_score"])
                if future.empty:
                    continue
                forecast_rows.append(
                    {
                        "task_profile": profile["task_profile"],
                        "display_name": profile["display_name"],
                        "domain": domain,
                        "domain_label": domain_label(domain),
                        "scenario": scenario,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "selected_model": future["canonical_model"],
                        "selected_family": future["model_family"],
                        "selected_vendor": future["vendor"],
                        "required_domain_score": profile["required_domain_score"],
                        "selected_domain_score": round(float(future["forecast_domain_score"]), 3),
                        "forecast_task_cost_usd": round(float(future["forecast_task_cost_usd"]), 6),
                        "current_task_cost_usd": round(current_cost, 6),
                        "cost_factor_vs_current": round(float(future["forecast_task_cost_usd"]) / current_cost, 6) if current_cost > 0 else np.nan,
                        "adequacy_status": future_status,
                        "input_tokens": profile["input_tokens"],
                        "output_tokens": profile["output_tokens"],
                        "human_gate": profile["human_gate"],
                        "method": f"Cheapest adequate candidate after domain score gain and quality-adjusted cost factor; prior={prior['annual_factor']} per year.",
                    }
                )
    fixed_curves = pd.concat([pd.DataFrame(current_rows), pd.DataFrame(forecast_rows)], ignore_index=True)
    fixed_candidates = pd.DataFrame(candidate_rows)

    divergence_rows = []
    current_message = float(message_trends[message_trends["year"].eq(2026)]["modeled_average_message_cost_usd"].iloc[0]) if not message_trends.empty else np.nan
    current_fixed = fixed_curves[fixed_curves["scenario"].eq("current")]["forecast_task_cost_usd"].median() if not fixed_curves.empty else np.nan
    for scenario, prior in QUALITY_ADJUSTED_COST_PRIORS.items():
        for horizon in [2, 5, 10]:
            metric_rows = forecasts[forecasts["scenario"].eq(scenario) & forecasts["horizon_years"].eq(horizon)]
            task_share = metric_rows.loc[metric_rows["metric"].eq("share_of_us_occupation_tasks_materially_touched"), "value"]
            context_mult = metric_rows.loc[metric_rows["metric"].eq("frontier_context_window_multiplier"), "value"]
            task_share_value = float(task_share.iloc[0]) if len(task_share) else 0.12 * horizon
            context_value = float(context_mult.iloc[0]) if len(context_mult) else 1 + horizon
            complexity_multiplier = 1 + task_share_value * 8.0 + math.log1p(max(context_value, 1.0)) * 0.45
            frontier_price_factor = float(prior["frontier_price_factor"]) ** horizon
            future_message = current_message * complexity_multiplier * frontier_price_factor
            fixed_subset = fixed_curves[
                fixed_curves["scenario"].eq(scenario)
                & fixed_curves["horizon_years"].eq(horizon)
                & fixed_curves["adequacy_status"].eq("adequate")
            ]
            if fixed_subset.empty:
                fixed_subset = fixed_curves[fixed_curves["scenario"].eq(scenario) & fixed_curves["horizon_years"].eq(horizon)]
            future_fixed = float(fixed_subset["forecast_task_cost_usd"].median()) if not fixed_subset.empty else np.nan
            divergence_rows.append(
                {
                    "scenario": scenario,
                    "horizon_years": horizon,
                    "target_year": 2026 + horizon,
                    "modeled_average_message_cost_usd": round(future_message, 6),
                    "message_cost_factor_vs_2026": round(future_message / current_message, 4) if current_message > 0 else np.nan,
                    "median_fixed_task_cost_usd": round(future_fixed, 6) if np.isfinite(future_fixed) else np.nan,
                    "fixed_task_cost_factor_vs_2026": round(future_fixed / current_fixed, 6) if current_fixed > 0 and np.isfinite(future_fixed) else np.nan,
                    "frontier_workload_complexity_multiplier": round(complexity_multiplier, 4),
                    "frontier_price_factor": round(frontier_price_factor, 4),
                    "task_share_touched_input": round(task_share_value, 4),
                    "context_multiplier_input": round(context_value, 4),
                    "interpretation": "Average task/message cost can rise if workload complexity and frontier routing grow faster than fixed-task unit costs fall.",
                    "method": "Scenario transform combining capability forecast task contact, context multiplier and quality-adjusted fixed-task cost priors.",
                }
            )

    return (
        write_table(message_trends, "llm_message_cost_trends"),
        write_table(message_components, "llm_message_cost_profile_components"),
        write_table(fixed_candidates, "fixed_task_cost_candidates"),
        write_table(fixed_curves, "fixed_task_cost_curves"),
        write_table(pd.DataFrame(divergence_rows), "cost_divergence_scenarios"),
    )


def build_vendor_frontier_scores(company_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = company_scores.copy()
    scores["vendor"] = scores["model_family"].map(FAMILY_VENDOR_MAP).fillna(scores["model_family"])
    rows = []
    component_rows = []
    for vendor, group in scores.groupby("vendor"):
        group = group.sort_values("frontier_momentum_heuristic_index", ascending=False)
        flagship = group.iloc[0]
        weights = numeric(group["evidence_count"]).fillna(1).clip(lower=1)
        row: dict[str, Any] = {
            "vendor": vendor,
            "flagship_family": flagship["model_family"],
            "portfolio_families": ",".join(group["model_family"].astype(str)),
            "family_count": len(group),
            "evidence_count": round(float(weights.sum()), 2),
        }
        for component in COMPONENT_WEIGHTS:
            flagship_value = float(flagship.get(component, 0) or 0)
            portfolio_mean = float(np.average(numeric(group[component]).fillna(0), weights=weights))
            row[component] = round(0.65 * flagship_value + 0.35 * portfolio_mean, 2)
            component_rows.append(
                {
                    "vendor": vendor,
                    "flagship_family": flagship["model_family"],
                    "component": component,
                    "flagship_component_value": round(flagship_value, 2),
                    "evidence_weighted_portfolio_mean": round(portfolio_mean, 2),
                    "vendor_component_value": row[component],
                    "baseline_weight": COMPONENT_WEIGHTS[component],
                }
            )
        row["vendor_frontier_portfolio_score"] = round(sum(row[c] * w for c, w in COMPONENT_WEIGHTS.items()), 2)
        rows.append(row)
    vendor_scores = pd.DataFrame(rows).sort_values("vendor_frontier_portfolio_score", ascending=False)
    vendor_scores["rank"] = vendor_scores["vendor_frontier_portfolio_score"].rank(ascending=False, method="min").astype(int)
    component_df = pd.DataFrame(component_rows)
    return write_table(vendor_scores, "vendor_frontier_scores"), write_table(component_df, "vendor_score_components")


def source_latest_date(df: pd.DataFrame) -> str:
    date_cols = [c for c in df.columns if "date" in c.lower() or c.lower().endswith("_at")]
    latest = pd.NaT
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().any():
            candidate = parsed.max()
            latest = candidate if pd.isna(latest) or candidate > latest else latest
    return latest.date().isoformat() if pd.notna(latest) else ""


def field_coverage(df: pd.DataFrame, candidates: list[str], positive: bool = False) -> float:
    cols = [col for col in candidates if col in df.columns]
    if not cols or df.empty:
        return 0.0
    covered = pd.Series(False, index=df.index)
    for col in cols:
        values = df[col]
        if positive:
            covered = covered | numeric(values).gt(0).fillna(False)
        else:
            covered = covered | values.notna() & values.astype(str).str.strip().ne("")
    return round(float(covered.mean()), 4)


def build_coverage_diagnostics(company_scores: pd.DataFrame, match_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = DATASET / "dataset_manifest.csv"
    source_path = DATASET / "source_registry_rich.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    sources = pd.read_csv(source_path) if source_path.exists() else pd.DataFrame(columns=["source_id", "name", "url", "type"])
    source_names = dict(zip(sources.get("source_id", []), sources.get("name", [])))
    rows = []
    for _, row in manifest.iterrows():
        path = ROOT / clean_text(row.get("csv_path"))
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        source_ids = clean_text(row.get("source_ids")).split(",") if clean_text(row.get("source_ids")) else []
        rows.append(
            {
                "table": row.get("table"),
                "rows": int(row.get("rows", len(df))),
                "columns": int(row.get("columns", len(df.columns))),
                "captured_at": row.get("captured_at"),
                "source_ids": ",".join(source_ids),
                "source_names": sentence_join([source_names.get(source_id, source_id) for source_id in source_ids], 5),
                "latest_source_date": source_latest_date(df),
                "release_date_coverage": field_coverage(df, ["release_date", "release_or_created_at", "leaderboard_publish_date", "created_at"]),
                "price_coverage": field_coverage(df, ["input_usd_per_1m", "output_usd_per_1m"], positive=True),
                "access_class_coverage": field_coverage(df, ["access_class", "license"]),
                "benchmark_value_coverage": field_coverage(df, ["rating", "score", "value"], positive=False),
                "context_window_coverage": field_coverage(df, ["context_window"], positive=True),
                "organization_vendor_coverage": field_coverage(df, ["vendor", "organization", "org", "author", "vendor_or_author"]),
                "columns_present": ",".join(df.columns[:40]),
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values("table")

    openrouter = read_csv_table("openrouter_models_catalog").copy()
    openrouter["model_family"] = [
        frontier_family_from_model(name, mid, vendor)
        for name, mid, vendor in zip(openrouter.get("canonical_model", ""), openrouter.get("openrouter_id", ""), openrouter.get("vendor", ""))
    ]
    direct_counts = match_audit[match_audit["direct_model_match"].astype(bool)].groupby("model_family").size().rename("direct_benchmark_match_count")
    family_rows = []
    for _, row in company_scores.iterrows():
        family = row["model_family"]
        family_openrouter = openrouter[openrouter["model_family"].eq(family)]
        price_cov = field_coverage(family_openrouter, ["input_usd_per_1m", "output_usd_per_1m"], positive=True)
        release_cov = field_coverage(family_openrouter, ["release_date"])
        access_cov = field_coverage(family_openrouter, ["access_class"])
        context_cov = field_coverage(family_openrouter, ["context_window"], positive=True)
        direct_count = int(direct_counts.get(family, 0))
        proxy_value = row.get("lmarena_models", 0)
        proxy_count = int(proxy_value) if pd.notna(proxy_value) else 0
        api_value = row.get("api_model_count", 0)
        api_count = int(api_value) if pd.notna(api_value) else 0
        coverage_parts = [
            price_cov,
            release_cov,
            access_cov,
            context_cov,
            min(1.0, direct_count / 3),
            min(1.0, proxy_count / 5),
        ]
        source_gap_count = int(sum(part <= 0 for part in coverage_parts))
        family_rows.append(
            {
                "model_family": family,
                "vendor": FAMILY_VENDOR_MAP.get(family, family),
                "api_model_count": api_count,
                "price_coverage": price_cov,
                "release_date_coverage": release_cov,
                "access_class_coverage": access_cov,
                "context_window_coverage": context_cov,
                "direct_benchmark_match_count": direct_count,
                "family_proxy_benchmark_count": proxy_count,
                "labor_signal_coverage": 1.0,
                "source_gap_count": source_gap_count,
                "coverage_score": round(float(np.mean(coverage_parts) * 100), 2),
            }
        )
    family_matrix = pd.DataFrame(family_rows).sort_values("coverage_score", ascending=False)
    return write_table(diagnostics, "source_coverage_diagnostics"), write_table(family_matrix, "family_coverage_matrix")


def build_rank_stability(company_scores: pd.DataFrame, draws: int = 700) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260516)
    component_cols = list(COMPONENT_WEIGHTS)
    evidence_col = "evidence_source_count" if "evidence_source_count" in company_scores.columns else "evidence_count"
    base = company_scores[["model_family", "rank", evidence_col, *component_cols]].copy().fillna(0)
    rows = []
    for draw in range(draws):
        weights = rng.dirichlet(np.array(list(COMPONENT_WEIGHTS.values())) * 120)
        evidence = numeric(base[evidence_col]).clip(lower=1)
        # Raw leaderboard rows are pseudo-replicates, so uncertainty shrinks by
        # independent source families rather than by row count.
        noise_scale = 2.2 + 12 / np.sqrt(evidence)
        simulated_components = base[component_cols].to_numpy(dtype=float) + rng.normal(0, noise_scale.to_numpy()[:, None], size=(len(base), len(component_cols)))
        simulated_components = np.clip(simulated_components, 0, 100)
        simulated_scores = simulated_components @ weights
        ranks = pd.Series(simulated_scores).rank(ascending=False, method="min").astype(int)
        for i, family in enumerate(base["model_family"]):
            rows.append(
                {
                    "draw": draw,
                    "model_family": family,
                    "bootstrap_frontier_momentum_heuristic_index": round(float(simulated_scores[i]), 2),
                    "bootstrap_rank": int(ranks.iloc[i]),
                    "method": "Deterministic component bootstrap with evidence-scaled noise and Dirichlet component weights; not a calibrated probability model.",
                }
            )
    bootstrap = pd.DataFrame(rows)
    intervals = bootstrap.groupby("model_family").agg(
        score_p10=("bootstrap_frontier_momentum_heuristic_index", lambda s: round(float(np.quantile(s, 0.10)), 2)),
        score_p50=("bootstrap_frontier_momentum_heuristic_index", lambda s: round(float(np.quantile(s, 0.50)), 2)),
        score_p90=("bootstrap_frontier_momentum_heuristic_index", lambda s: round(float(np.quantile(s, 0.90)), 2)),
        best_rank=("bootstrap_rank", "min"),
        median_rank=("bootstrap_rank", "median"),
        worst_rank=("bootstrap_rank", "max"),
        rank_iqr=("bootstrap_rank", lambda s: round(float(np.quantile(s, 0.75) - np.quantile(s, 0.25)), 2)),
    ).reset_index()
    intervals = intervals.merge(base[["model_family", "rank"]].rename(columns={"rank": "current_rank"}), on="model_family", how="left")
    intervals["rank_stability_label"] = np.select(
        [intervals["rank_iqr"].le(1), intervals["rank_iqr"].le(3)],
        ["stable", "moderate"],
        default="unstable",
    )
    intervals = intervals.sort_values(["median_rank", "score_p50"])
    return write_table(bootstrap, "frontier_score_bootstrap"), write_table(intervals, "rank_stability_intervals")


def build_failure_mode_audits(company_scores: pd.DataFrame, match_audit: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "claim_id": "family-frontier-ranking",
            "claim": "Frontier-family ranking is useful for comparing public signals.",
            "assumption": "Benchmark, release, ecosystem, cost and openness signals are directionally informative.",
            "failure_mode": "A private model or undisclosed benchmark result changes the frontier without appearing in public data.",
            "evidence_that_would_break_it": "Official release or benchmark snapshot with materially higher direct model evidence.",
            "mitigation": "Keep heuristic label, show sensitivity, and refresh source snapshots before external use.",
            "affected_artifact": "company_frontier_scores.csv",
            "severity": "high",
        },
        {
            "claim_id": "direct-model-evidence",
            "claim": "Direct model matches are stronger than family-level proxies.",
            "assumption": "Normalized names and curated aliases correctly identify equivalent model rows.",
            "failure_mode": "Vendor naming drift or hidden routing makes same-looking model IDs non-equivalent.",
            "evidence_that_would_break_it": "Provider documentation or benchmark metadata showing the row is a different checkpoint.",
            "mitigation": "Expose match confidence and keep unmatched/family-only rows in the audit table.",
            "affected_artifact": "model_benchmark_match_audit.csv",
            "severity": "high",
        },
        {
            "claim_id": "vendor-portfolio-ranking",
            "claim": "Vendor portfolio scores are easier for business readers than family scores.",
            "assumption": "Flagship plus evidence-weighted portfolio mean is a reasonable vendor aggregation.",
            "failure_mode": "A vendor has one dominant model family and several weak rows that should not affect portfolio interpretation.",
            "evidence_that_would_break_it": "Manual portfolio review showing one product line carries nearly all deployment relevance.",
            "mitigation": "Report flagship family, family count and components beside the vendor score.",
            "affected_artifact": "vendor_frontier_scores.csv",
            "severity": "medium",
        },
        {
            "claim_id": "labor-domain-pressure",
            "claim": "Business domain pressure summarizes occupation-level AI exposure in business language.",
            "assumption": "Keyword domain mapping is sufficient for a portfolio-level translation layer.",
            "failure_mode": "Occupation titles hide domain-specific workflows or regulated subdomains.",
            "evidence_that_would_break_it": "Manual O*NET task audit contradicting assigned domain for high-weight occupations.",
            "mitigation": "Keep occupation examples and human gates visible for each domain.",
            "affected_artifact": "business_domain_ai_pressure.csv",
            "severity": "medium",
        },
        {
            "claim_id": "forecast-envelope",
            "claim": "Scenario envelopes show plausible directional range.",
            "assumption": "Conservative/base/aggressive scenarios bound the intended stress test.",
            "failure_mode": "A structural market break makes historical slopes irrelevant.",
            "evidence_that_would_break_it": "Large price shock, regulation shock, or discontinuous capability release.",
            "mitigation": "Label forecast bands as non-calibrated scenario envelopes.",
            "affected_artifact": "forecast_uncertainty_bands.png",
            "severity": "high",
        },
    ]
    failure_modes = pd.DataFrame(rows)

    direct_counts = match_audit[match_audit["direct_model_match"].astype(bool)].groupby("model_family").size().rename("direct_benchmark_match_count")
    coverage_map = coverage.set_index("model_family") if not coverage.empty else pd.DataFrame()
    audit_rows = []
    for _, row in company_scores.iterrows():
        family = row["model_family"]
        direct_count = int(direct_counts.get(family, 0))
        coverage_score = float(coverage_map.loc[family, "coverage_score"]) if family in coverage_map.index else 0.0
        evidence_count = float(row.get("evidence_count", 0) or 0)
        reasons = []
        if direct_count < 2:
            reasons.append("few_direct_model_matches")
        if coverage_score < 60:
            reasons.append("low_source_coverage")
        if evidence_count < company_scores["evidence_count"].median():
            reasons.append("below_median_evidence_depth")
        audit_rows.append(
            {
                "model_family": family,
                "vendor": FAMILY_VENDOR_MAP.get(family, family),
                "evidence_count": round(evidence_count, 2),
                "direct_benchmark_match_count": direct_count,
                "coverage_score": round(coverage_score, 2),
                "underobserved": bool(reasons),
                "underobserved_reasons": ",".join(reasons) if reasons else "none",
                "audit_note": "Use family ranking cautiously when direct benchmark and coverage fields are sparse." if reasons else "Coverage is comparatively strong in this snapshot.",
            }
        )
    underobserved = pd.DataFrame(audit_rows).sort_values(["underobserved", "coverage_score"], ascending=[False, True])
    return write_table(failure_modes, "claim_failure_modes"), write_table(underobserved, "underobserved_family_audit")


def business_domain_for_job(row: pd.Series) -> str:
    text = " ".join(clean_text(row.get(col)) for col in ["title", "job_family", "example_tasks"]).lower()
    scores = {
        domain: sum(1 for keyword in keywords if keyword in text)
        for domain, keywords in BUSINESS_DOMAIN_RULES.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "operations_back_office"


def build_business_domain_implications(job_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = job_scores.copy()
    jobs["business_domain"] = jobs.apply(business_domain_for_job, axis=1)
    # Growth forecasts are percentages, not population weights.  Never use them
    # as a fallback denominator for labor-weighted domain averages.
    weights = numeric(jobs.get("bls_employment", pd.Series(index=jobs.index))).fillna(1).clip(lower=1)
    jobs["domain_labor_weight"] = weights
    rows = []
    for domain in BUSINESS_DOMAIN_RULES:
        group = jobs[jobs["business_domain"].eq(domain)].copy()
        if group.empty:
            group = jobs.sort_values("near_term_disruption_index", ascending=False).head(1).copy()
        weight = numeric(group["domain_labor_weight"]).fillna(1)
        rows.append(
            {
                "business_domain": domain,
                "occupation_count": len(group),
                "labor_weight_sum": round(float(weight.sum()), 2),
                "disruption_index": round(float(np.average(group["near_term_disruption_index"], weights=weight)), 2),
                "augmentation_index": round(float(np.average(group["augmentation_index"], weights=weight)), 2),
                "replacement_feasibility_index": round(float(np.average(group["full_job_automation_feasibility_index"], weights=weight)), 2),
                "human_bottleneck_index": round(float(np.average(group["human_bottleneck_index"], weights=weight)), 2),
                "dominant_outcome": group["dominant_outcome"].mode().iloc[0] if not group["dominant_outcome"].mode().empty else "mixed_redesign",
                "example_occupations": sentence_join(group.sort_values("near_term_disruption_index", ascending=False)["title"].astype(str).tolist(), 4),
            }
        )
    domain_pressure = pd.DataFrame(rows).sort_values("disruption_index", ascending=False)
    domain_pressure["pressure_label"] = pd.cut(
        domain_pressure["disruption_index"],
        bins=[-1, 25, 45, 65, 101],
        labels=["low", "moderate", "high", "very_high"],
    ).astype(str)
    workflow_rows = [
        ("software_engineering", "Issue triage, code review, test generation and migration planning", "draft patches and review checklists", "senior engineer approval and production ownership"),
        ("customer_support", "Ticket summarization, answer drafting and escalation routing", "suggest responses and detect repeated issues", "human review for refunds, safety and account actions"),
        ("legal_compliance", "Contract review, policy comparison and evidence packet preparation", "extract clauses and compare requirements", "licensed professional signoff"),
        ("marketing_content", "Campaign briefs, SEO drafts, localization and asset variants", "generate drafts and performance hypotheses", "brand, legal and factual review"),
        ("finance_analysis", "Variance explanations, spreadsheet checks and memo drafting", "surface anomalies and draft analysis", "accountability for assumptions and controls"),
        ("healthcare_administration", "Claims coding, prior authorization packets and patient-message drafts", "prepare structured documentation", "clinical and compliance review"),
        ("education", "Lesson planning, tutoring support and feedback drafting", "adapt materials and summarize progress", "teacher judgment and student relationship"),
        ("operations_back_office", "Data entry cleanup, scheduling, reconciliation and process documentation", "automate routine transformations", "exception handling and vendor/customer accountability"),
    ]
    workflows = pd.DataFrame(
        [
            {
                "business_domain": domain,
                "workflow_example": workflow,
                "likely_ai_role": ai_role,
                "human_gate": gate,
                "evidence_basis": "Mapped from occupation exposure, O*NET task features, augmentation/substitution mode shares and bottleneck indexes.",
            }
            for domain, workflow, ai_role, gate in workflow_rows
        ]
    )
    return write_table(domain_pressure, "business_domain_ai_pressure"), write_table(workflows, "domain_workflow_examples")


def build_release_cadence() -> tuple[pd.DataFrame, pd.DataFrame]:
    openrouter = read_csv_table("openrouter_models_catalog").copy()
    epoch = read_csv_table("epoch_models_normalized").copy()
    rows = []
    for _, row in openrouter.iterrows():
        family = frontier_family_from_model(row.get("canonical_model"), row.get("openrouter_id"), row.get("vendor"))
        release_date = pd.to_datetime(row.get("release_date"), errors="coerce")
        if pd.isna(release_date):
            continue
        rows.append(
            {
                "release_date": release_date.date().isoformat(),
                "model_family": family,
                "vendor": FAMILY_VENDOR_MAP.get(family, clean_text(row.get("vendor"), "Unknown")),
                "product_line": clean_text(row.get("product_line"), family),
                "model_name": row.get("canonical_model"),
                "release_type": "open_weight" if clean_text(row.get("access_class")).startswith("open") else "api_catalog",
                "source_table": "openrouter_models_catalog",
            }
        )
    epoch_family = family_column(epoch)
    for _, row in epoch.iterrows():
        family = clean_text(row.get(epoch_family))
        release_date = pd.to_datetime(row.get("release_date"), errors="coerce")
        if pd.isna(release_date) or family in {"", "Other", "unknown"}:
            continue
        rows.append(
            {
                "release_date": release_date.date().isoformat(),
                "model_family": family,
                "vendor": FAMILY_VENDOR_MAP.get(family, clean_text(row.get("vendor"), "Unknown")),
                "product_line": clean_text(row.get("product_line"), family),
                "model_name": row.get("canonical_model", row.get("model")),
                "release_type": "open_weight" if "open" in clean_text(row.get("access_class")).lower() else "research_metadata",
                "source_table": "epoch_models_normalized",
            }
        )
    releases = pd.DataFrame(rows)
    releases = releases[releases["model_family"].isin(set(FAMILY_VENDOR_MAP) | set(FRONTIER_FAMILIES))]
    releases = releases.drop_duplicates(["release_date", "model_family", "vendor", "product_line", "model_name", "release_type"])
    reference = pd.Timestamp(REFERENCE_DATE)

    def cadence_summary(group: pd.DataFrame, label_col: str) -> dict[str, Any]:
        dates = pd.to_datetime(group["release_date"], errors="coerce").dropna().sort_values()
        recent = dates[dates >= reference - pd.Timedelta(days=365)]
        gaps = dates.drop_duplicates().diff().dt.days.dropna()
        latest = dates.max() if len(dates) else pd.NaT
        median_gap = float(gaps.median()) if len(gaps) else np.nan
        cadence_label = "fast" if len(recent) >= 5 or (np.isfinite(median_gap) and median_gap <= 75) else "steady" if len(recent) >= 2 else "slow_or_sparse"
        source_mix = group.groupby("release_type").size().sort_values(ascending=False)
        return {
            label_col: group[label_col].iloc[0],
            "vendor": group["vendor"].iloc[0] if label_col == "model_family" else group[label_col].iloc[0],
            "total_releases": len(group),
            "recent_releases_365d": len(recent),
            "median_days_between_releases": round(median_gap, 1) if np.isfinite(median_gap) else np.nan,
            "days_since_latest_release": int((reference - latest).days) if pd.notna(latest) else np.nan,
            "latest_release_date": latest.date().isoformat() if pd.notna(latest) else "",
            "source_mix": ",".join(f"{idx}:{value}" for idx, value in source_mix.items()),
            "cadence_label": cadence_label,
        }

    family_rows = [cadence_summary(group, "model_family") for _, group in releases.groupby("model_family") if len(group)]
    vendor_rows = []
    for vendor, group in releases.groupby("vendor"):
        summary = cadence_summary(group.assign(vendor=vendor), "vendor")
        summary["portfolio_families"] = ",".join(sorted(set(group["model_family"])))
        vendor_rows.append(summary)
    family_cadence = pd.DataFrame(family_rows).sort_values("recent_releases_365d", ascending=False)
    vendor_cadence = pd.DataFrame(vendor_rows).sort_values("recent_releases_365d", ascending=False)
    return write_table(family_cadence, "release_cadence_by_family"), write_table(vendor_cadence, "release_cadence_by_vendor")


def scenario_weight_vector(spec: dict[str, Any], horizon: int) -> np.ndarray:
    """Interpolate a scenario's baseline multipliers at the given horizon."""
    t = (horizon - 2) / 8.0
    vector = []
    for component, base_weight in COMPONENT_WEIGHTS.items():
        early_m = float(spec["early"].get(component, 1.0))
        late_m = float(spec["late"].get(component, 1.0))
        multiplier = math.exp(math.log(early_m) * (1 - t) + math.log(late_m) * t)
        vector.append(base_weight * multiplier)
    array = np.array(vector, dtype=float)
    return array / array.sum()


def build_company_leadership_simulation(company_scores: pd.DataFrame, draws: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(20260516)
    components = list(COMPONENT_WEIGHTS)
    base = company_scores[["model_family", *components]].copy().fillna(0)
    rows = []
    raw_scores = base[components].to_numpy(dtype=float)
    for scenario, spec in LEADERSHIP_SCENARIO_MULTIPLIERS.items():
        description = str(spec["description"])
        for horizon in [2, 5, 10]:
            weights = scenario_weight_vector(spec, horizon)
            alpha = np.maximum(weights * 140, 2.0)
            wins = defaultdict(int)
            score_store = defaultdict(list)
            for _ in range(draws):
                sampled_weights = rng.dirichlet(alpha)
                noise = rng.normal(0, 2.5 + horizon * 0.18, size=raw_scores.shape[0])
                simulated = raw_scores @ sampled_weights + noise
                winner = int(np.argmax(simulated))
                family = base.iloc[winner]["model_family"]
                wins[family] += 1
                for i, fam in enumerate(base["model_family"]):
                    score_store[fam].append(float(simulated[i]))
            for fam in base["model_family"]:
                values = np.array(score_store[fam])
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_description": description,
                        "horizon_years": horizon,
                        "target_year": 2026 + horizon,
                        "model_family": fam,
                        "simulation_win_share": wins[fam] / draws,
                        "simulated_score_mean": round(float(values.mean()), 2),
                        "simulated_score_p10": round(float(np.quantile(values, 0.10)), 2),
                        "simulated_score_p90": round(float(np.quantile(values, 0.90)), 2),
                        "draws": draws,
                        "method": "Dirichlet around scenario weights derived from documented multipliers over the published baseline COMPONENT_WEIGHTS; evidence-scaled noise; not a calibrated probability model.",
                    }
                )
    out = pd.DataFrame(rows).sort_values(["scenario", "horizon_years", "simulation_win_share"], ascending=[True, True, False])
    return write_table(out, "company_next_frontier_probabilities")


def build_leadership_model_audit(company_scores: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    scored = company_scores.copy()
    # Component ranks feed the audit notes so commentary is derived from data.
    for col, rank_col in [
        ("performance_component", "performance_rank"),
        ("release_velocity_component", "release_velocity_rank"),
        ("ecosystem_component", "ecosystem_rank"),
        ("cost_efficiency_component", "cost_efficiency_rank"),
        ("openness_component", "openness_rank"),
    ]:
        scored[rank_col] = numeric(scored[col]).rank(ascending=False, method="min")
    scored["family_total"] = len(scored)
    # Audit the currently strongest families instead of a hardcoded lab list.
    audited_families = scored.sort_values("rank")["model_family"].head(8).tolist()
    rows = []
    for family in audited_families:
        match = scored[scored["model_family"].eq(family)]
        if match.empty:
            continue
        row = match.iloc[0]
        prob_summary = probabilities[probabilities["model_family"].eq(family)].pivot_table(
            index="scenario", columns="horizon_years", values="simulation_win_share", aggfunc="first"
        )
        rows.append(
            {
                "model_family": family,
                "frontier_momentum_heuristic_index": row.get("frontier_momentum_heuristic_index"),
                "current_rank": row.get("rank"),
                "performance_component": row.get("performance_component"),
                "release_velocity_component": row.get("release_velocity_component"),
                "ecosystem_component": row.get("ecosystem_component"),
                "capability_surface_component": row.get("capability_surface_component"),
                "cost_efficiency_component": row.get("cost_efficiency_component"),
                "openness_component": row.get("openness_component"),
                "frontier_quality_10y_probability": prob_summary.loc["frontier_quality", 10] if "frontier_quality" in prob_summary.index and 10 in prob_summary.columns else np.nan,
                "open_ecosystem_10y_probability": prob_summary.loc["open_ecosystem_upside", 10] if "open_ecosystem_upside" in prob_summary.index and 10 in prob_summary.columns else np.nan,
                "audit_note": leadership_audit_note(row),
            }
        )
    audit = pd.DataFrame(rows).sort_values("frontier_quality_10y_probability", ascending=False)
    return write_table(audit, "leadership_model_audit")


def leadership_audit_note(row: pd.Series) -> str:
    # Notes are derived from this snapshot's component ranks, never hand-written
    # per family, so the narrative cannot silently outlive the data.
    parts = []

    def rank_of(col: str) -> float:
        value = row.get(col)
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    perf = rank_of("performance_rank")
    openness = rank_of("openness_rank")
    cost = rank_of("cost_efficiency_rank")
    velocity = rank_of("release_velocity_rank")
    families = rank_of("family_total") or np.nan

    if np.isfinite(perf):
        parts.append(f"performance signal ranked {int(perf)} of {int(families)}" if np.isfinite(families) else f"performance signal ranked {int(perf)}")
        if perf == 1:
            parts.append("strongest current benchmark/performance component in this snapshot")
    if np.isfinite(velocity):
        if velocity <= 2:
            parts.append("top release-velocity signal")
        elif velocity >= max(3.0, (families or 0) - 1):
            parts.append("release-velocity signal near the bottom of the panel")
    if np.isfinite(openness) and openness <= 2:
        parts.append("leading openness/ecosystem profile")
    if np.isfinite(cost) and cost <= 2:
        parts.append("strong cost-efficiency signal")
    if not parts:
        return "Interpret with caution; public signals are incomplete and component-dependent."
    parts.append("derived from this snapshot's component ranks")
    return "; ".join(parts) + "."


def simple_kmeans(matrix: np.ndarray, k: int, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260516)
    finite = np.nan_to_num(matrix, nan=0.0)
    if len(finite) < k:
        labels = np.zeros(len(finite), dtype=int)
        return labels, finite[:1]
    initial_idx = rng.choice(len(finite), size=k, replace=False)
    centers = finite[initial_idx].copy()
    labels = np.zeros(len(finite), dtype=int)
    for _ in range(iterations):
        distances = ((finite[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            points = finite[labels == cluster]
            if len(points):
                centers[cluster] = points.mean(axis=0)
    return labels, centers


def label_cluster(center: pd.Series) -> str:
    if center.get("full_job_automation_feasibility_index", 0) >= 18:
        return "replacement-prone clerical/transaction work"
    if center.get("augmentation_index", 0) >= center.get("substitution_pressure_index", 0) + 8:
        return "augmentation-heavy expert work"
    if center.get("human_bottleneck_index", 0) >= 22:
        return "trust/physical bottleneck work"
    if center.get("code_task_share", 0) >= 0.08 or center.get("analysis_task_share", 0) >= 0.12:
        return "technical analysis work"
    if center.get("agentic_task_share", 0) >= 0.10:
        return "coordination and management work"
    return "mixed redesign work"


def build_labor_deep_dive(job_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [
        "observed_exposure",
        "capability_exposure_index",
        "substitution_pressure_index",
        "augmentation_index",
        "human_bottleneck_index",
        "full_job_automation_feasibility_index",
        "language_task_share",
        "code_task_share",
        "analysis_task_share",
        "visual_task_share",
        "agentic_task_share",
        "physical_bottleneck",
        "human_trust_bottleneck",
        "regulated_bottleneck",
    ]
    work = job_scores.copy()
    matrix = work[features].apply(numeric).fillna(0)
    standardized = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
    labels, centers = simple_kmeans(standardized.to_numpy(dtype=float), k=7)
    work["labor_cluster_id"] = labels
    centers_original = pd.DataFrame(
        [matrix[work["labor_cluster_id"].eq(i)].mean() for i in range(7)]
    ).reset_index().rename(columns={"index": "labor_cluster_id"})
    centers_original["cluster_label"] = centers_original.apply(label_cluster, axis=1)
    cluster_sizes = work.groupby("labor_cluster_id").agg(
        occupation_count=("title", "count"),
        example_occupations=("title", lambda s: sentence_join(list(s.head(5)), 5)),
        median_salary_median=("median_salary", "median"),
        job_forecast_sum=("job_forecast", "sum"),
    ).reset_index()
    cluster_profiles = centers_original.merge(cluster_sizes, on="labor_cluster_id", how="left").sort_values("full_job_automation_feasibility_index", ascending=False)

    work["allocated_bls_employment"] = np.nan
    if "bls_employment" in work.columns:
        counts = work.groupby("job_family")["title"].transform("count").replace(0, np.nan)
        work["allocated_bls_employment"] = numeric(work["bls_employment"]) / counts
    # Statistical-audit contract: job-growth percentages are forecasts, never
    # population weights.  Rows without allocated employment fall back to a
    # neutral unit weight and carry an explicit provenance label instead of
    # silently borrowing a growth rate as if it were headcount.
    allocated = numeric(work["allocated_bls_employment"])
    work["labor_weight"] = allocated.fillna(1.0).clip(lower=1.0)
    work["labor_weight_provenance"] = np.where(allocated.notna(), "allocated_employment", "neutral_unit_weight")
    summary_rows = []
    for group_col in ["job_family", "dominant_outcome", "risk_label"]:
        grouped = work.groupby(group_col, dropna=False)
        for name, group in grouped:
            weight = numeric(group["labor_weight"]).fillna(1)
            summary_rows.append(
                {
                    "grouping": group_col,
                    "group": clean_text(name, "unknown"),
                    "occupation_count": len(group),
                    "labor_weight_sum": round(float(weight.sum()), 2),
                    "weighted_disruption_index": round(float(np.average(group["near_term_disruption_index"], weights=weight)), 2),
                    "weighted_replacement_feasibility": round(float(np.average(group["full_job_automation_feasibility_index"], weights=weight)), 2),
                    "weighted_augmentation_index": round(float(np.average(group["augmentation_index"], weights=weight)), 2),
                    "weighted_human_bottleneck": round(float(np.average(group["human_bottleneck_index"], weights=weight)), 2),
                }
            )
    market_summary = pd.DataFrame(summary_rows).sort_values(["grouping", "weighted_disruption_index"], ascending=[True, False])
    replacement = work[
        [
            "occ_code",
            "title",
            "job_family",
            "dominant_outcome",
            "near_term_disruption_index",
            "substitution_pressure_index",
            "augmentation_index",
            "human_bottleneck_index",
            "full_job_automation_feasibility_index",
            "scenario_2y_task_share_base",
            "scenario_5y_task_share_base",
            "scenario_10y_task_share_base",
            "example_tasks",
            "labor_cluster_id",
        ]
    ].sort_values("full_job_automation_feasibility_index", ascending=False)
    assignments = work[["occ_code", "title", "job_family", "labor_cluster_id", "dominant_outcome"]].merge(
        cluster_profiles[["labor_cluster_id", "cluster_label"]], on="labor_cluster_id", how="left"
    )
    # Persist the weight provenance alongside the scores so weighted summaries
    # stay auditable from the published CSV alone.
    write_table(work, "job_exposure_scores")
    return (
        write_table(cluster_profiles, "labor_cluster_profiles"),
        write_table(market_summary, "labor_market_exposure_summary"),
        write_table(replacement, "job_replacement_feasibility"),
    )


def build_counterintuitive_findings(
    company_scores: pd.DataFrame,
    probabilities: pd.DataFrame,
    gap: pd.DataFrame,
    price_frontier: pd.DataFrame,
    labor_summary: pd.DataFrame,
    replacement: pd.DataFrame,
) -> pd.DataFrame:
    fq_10y = probabilities[(probabilities["horizon_years"].eq(10)) & (probabilities["scenario"].eq("frontier_quality"))].sort_values("simulation_win_share", ascending=False).head(1).iloc[0]
    open_10y = probabilities[(probabilities["horizon_years"].eq(10)) & (probabilities["scenario"].eq("open_ecosystem_upside"))].sort_values("simulation_win_share", ascending=False).head(1).iloc[0]
    widest_gap = gap.dropna(subset=["open_closed_best_gap"]).sort_values("open_closed_best_gap", ascending=False).head(1).iloc[0]
    efficient = price_frontier[price_frontier["price_performance_frontier"]].head(5)
    repl_top = replacement.head(1).iloc[0]
    aug = labor_summary[(labor_summary["grouping"].eq("dominant_outcome")) & (labor_summary["group"].eq("augmentation_first"))]
    repl = labor_summary[(labor_summary["grouping"].eq("dominant_outcome")) & (labor_summary["group"].eq("replacement_candidate"))]
    aug_weight = float(aug["labor_weight_sum"].sum()) if len(aug) else 0.0
    repl_weight = float(repl["labor_weight_sum"].sum()) if len(repl) else 0.0
    rows = [
        {
            "finding": "Raw frontier leadership and open-distribution upside are different questions.",
            "evidence": f"In the 10-year frontier-quality scenario, {fq_10y['model_family']} leads ({fq_10y['simulation_win_share']:.1%} of simulation draws); in the open-ecosystem-upside scenario, {open_10y['model_family']} leads ({open_10y['simulation_win_share']:.1%} of simulation draws).",
            "why_it_is_interesting": "The previous single 10-year number was misleading because it mixed best-model simulation share with adoption economics.",
            "artifact": "company_next_frontier_probabilities.csv",
        },
        {
            "finding": "The open-vs-closed gap is not one gap.",
            "evidence": f"The largest measured LMArena category gap is {widest_gap['category']} at {widest_gap['open_closed_best_gap']:.1f} rating points.",
            "why_it_is_interesting": "Open-source catchup can be true in one domain and false in another; a single headline benchmark hides where closed labs still have moat.",
            "artifact": "open_closed_gap_by_category.csv",
        },
        {
            "finding": "Cheap models can sit on the efficient frontier without being the raw best model.",
            "evidence": f"Efficient frontier examples include {sentence_join(efficient['canonical_model'].astype(str).tolist(), 3)}.",
            "why_it_is_interesting": "Enterprise adoption often follows sufficient capability per dollar, not absolute leaderboard rank.",
            "artifact": "price_performance_frontier.csv",
        },
        {
            "finding": "The top whole-job automation candidates are narrower than the top task-exposure jobs.",
            "evidence": f"The highest replacement-feasibility occupation is {repl_top['title']} with feasibility index {repl_top['full_job_automation_feasibility_index']:.1f}.",
            "why_it_is_interesting": "A job can be heavily touched by AI but still mostly redesigned around human review rather than deleted.",
            "artifact": "job_replacement_feasibility.csv",
        },
        {
            "finding": "Augmentation can be a larger labor-weighted mode than replacement.",
            "evidence": f"Available labor-weight proxy: augmentation-first={aug_weight:,.0f}, replacement-candidate={repl_weight:,.0f}.",
            "why_it_is_interesting": "This pushes the labor forecast toward workflow redesign, wage compression and productivity dispersion before mass full automation.",
            "artifact": "labor_market_exposure_summary.csv",
        },
    ]
    return write_table(pd.DataFrame(rows), "counterintuitive_findings")


PALETTE = {
    "ink": "#18202a",
    "muted": "#657181",
    "grid": "#d8dee7",
    "blue": "#2f5d7c",
    "teal": "#248277",
    "green": "#5f7f36",
    "gold": "#c58b2b",
    "orange": "#b65f35",
    "red": "#a23b3b",
    "purple": "#725a9c",
    "slate": "#53606f",
}

COMPONENT_COLORS = {
    "performance_component": "#2f5d7c",
    "release_velocity_component": "#248277",
    "ecosystem_component": "#5f7f36",
    "capability_surface_component": "#725a9c",
    "cost_efficiency_component": "#c58b2b",
    "openness_component": "#b65f35",
}

SCENARIO_COLORS = {
    "conservative": "#53606f",
    "base": "#2f5d7c",
    "aggressive": "#a23b3b",
}


def apply_chart_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c7ced8",
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "font.size": 10,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / path, dpi=220)
    plt.close(fig)


def soften_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c7ced8")
    ax.spines["bottom"].set_color("#c7ced8")


def wrap_tick_labels(labels: pd.Series | list[Any], width: int = 24) -> list[str]:
    return [textwrap.fill(clean_text(label), width=width) for label in labels]


def add_note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0,
        -0.16,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["muted"],
        fontsize=9,
        wrap=True,
    )


def plot_company_scores(scores: pd.DataFrame) -> None:
    top = scores.sort_values("frontier_momentum_heuristic_index", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    colors = np.where(top["sensitivity_label"].eq("stable_top_tier"), PALETTE["blue"], PALETTE["slate"])
    ax.barh(top["model_family"], top["frontier_momentum_heuristic_index"], color=colors)
    ax.set_title("Frontier Momentum Heuristic Index by Model Family")
    ax.set_xlabel("Heuristic index (0-100)")
    ax.grid(axis="x", alpha=0.25)
    for _, row in top.iterrows():
        ax.text(row["frontier_momentum_heuristic_index"] + 1, row["model_family"], f"{row['frontier_momentum_heuristic_index']:.1f}", va="center", fontsize=9, color=PALETTE["muted"])
    add_note(ax, "Stable top-tier families remain near the top across sensitivity variants; weight-sensitive families move materially when price or ecosystem weights change.")
    soften_axes(ax)
    finish_figure(fig, "company_frontier_scores.png")

    component_cols = list(COMPONENT_WEIGHTS)
    work = scores.sort_values("frontier_momentum_heuristic_index", ascending=False).head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    left = np.zeros(len(work))
    y = np.arange(len(work))
    for col in component_cols:
        values = work[col].to_numpy(dtype=float) * COMPONENT_WEIGHTS[col]
        ax.barh(y, values, left=left, color=COMPONENT_COLORS[col], label=col.replace("_component", "").replace("_", " "))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(work["model_family"])
    ax.set_xlabel("Weighted contribution to heuristic index")
    ax.set_title("What Drives Each Frontier-Family Score")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=3, loc="lower right", fontsize=8)
    add_note(ax, "The stacked bars show contribution after baseline weights. They make the ranking auditable: a high score can come from benchmark performance, ecosystem pull, cost efficiency, or openness.")
    soften_axes(ax)
    finish_figure(fig, "company_score_component_stack.png")

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    scatter = ax.scatter(
        scores["evidence_count"].clip(lower=1),
        scores["frontier_momentum_heuristic_index"],
        s=np.sqrt(scores["api_model_count"].fillna(0).clip(lower=1)) * 32,
        c=scores["openness_component"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Evidence rows across benchmarks, APIs and ecosystem sources (log)")
    ax.set_ylabel("Frontier momentum heuristic index")
    ax.set_title("Signal Strength vs Evidence Depth")
    for _, row in scores.sort_values("frontier_momentum_heuristic_index", ascending=False).head(8).iterrows():
        ax.annotate(str(row["model_family"]), (max(row["evidence_count"], 1), row["frontier_momentum_heuristic_index"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    cb = fig.colorbar(scatter, ax=ax, fraction=0.035)
    cb.set_label("Openness component")
    add_note(ax, "Bigger dots indicate more API catalog rows. This chart separates strong scores backed by broad evidence from sparse but visually impressive outliers.")
    soften_axes(ax)
    finish_figure(fig, "company_score_evidence_scatter.png")


def plot_job_scores(jobs: pd.DataFrame) -> None:
    top = jobs.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8.4))
    colors = top["dominant_outcome"].map(
        {
            "replacement_candidate": PALETTE["red"],
            "augmentation_first": PALETTE["teal"],
            "bottleneck_protected": PALETTE["green"],
            "mixed_redesign": PALETTE["gold"],
        }
    ).fillna(PALETTE["slate"])
    ax.barh(wrap_tick_labels(top["title"], 30), top["near_term_disruption_index"], color=colors)
    ax.set_title("Highest Near-Term AI Disruption Pressure by Occupation")
    ax.set_xlabel("Index (0-100)")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "Color encodes the dominant modeled outcome, so the chart separates task disruption from whole-job replacement.")
    soften_axes(ax)
    finish_figure(fig, "job_exposure_top.png")

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    wages = numeric(jobs["median_salary"])
    scatter = ax.scatter(
        wages,
        jobs["near_term_disruption_index"],
        s=numeric(jobs.get("full_job_automation_feasibility_index", pd.Series(index=jobs.index))).fillna(0).clip(lower=4) * 1.4,
        alpha=0.48,
        c=jobs["augmentation_index"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.25,
    )
    ax.set_xscale("log")
    ax.set_title("AI Disruption Pressure vs Median Salary")
    ax.set_xlabel("Median salary / wage proxy (log scale)")
    ax.set_ylabel("Near-term disruption index")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(scatter, ax=ax, fraction=0.035)
    cb.set_label("Augmentation index")
    add_note(ax, "Point size tracks whole-job replacement feasibility. High salary plus high disruption is a redesign signal, not automatically a deletion signal.")
    soften_axes(ax)
    finish_figure(fig, "job_exposure_wage_scatter.png")


def plot_forecasts(forecasts: pd.DataFrame) -> None:
    for metric, path, title, ylabel in [
        ("share_of_us_occupation_tasks_materially_touched", "labor_task_forecast.png", "Scenario: Share of Occupation Tasks Materially Touched by AI", "Task share"),
        ("frontier_output_price_factor", "cost_forecast_scenarios.png", "Scenario: Frontier Output Cost Factor", "Fraction of current cost"),
        ("open_weight_lmarena_gap_remaining", "open_closed_catchup.png", "Scenario: Open-Weight Gap Remaining", "Arena rating points"),
    ]:
        work = forecasts[forecasts["metric"].eq(metric)]
        fig, ax = plt.subplots(figsize=(9.6, 5.8))
        for scenario, group in work.groupby("scenario"):
            group = group.sort_values("horizon_years")
            ax.plot(group["target_year"], group["value"], marker="o", linewidth=2.5, markersize=6, color=SCENARIO_COLORS.get(scenario, PALETTE["slate"]), label=scenario)
        ax.set_title(title)
        ax.set_xlabel("Target year")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        soften_axes(ax)
        finish_figure(fig, path)

    dashboard_metrics = [
        ("frontier_context_window_multiplier", "Context window multiplier"),
        ("frontier_output_price_factor", "Output price factor"),
        ("open_weight_lmarena_gap_remaining", "Open-weight gap remaining"),
        ("share_of_us_occupation_tasks_materially_touched", "Task share touched"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.flatten(), dashboard_metrics):
        work = forecasts[forecasts["metric"].eq(metric)]
        for scenario, group in work.groupby("scenario"):
            group = group.sort_values("horizon_years")
            ax.plot(group["target_year"], group["value"], marker="o", linewidth=2.2, color=SCENARIO_COLORS.get(scenario, PALETTE["slate"]), label=scenario)
        ax.set_title(title)
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.22)
        soften_axes(ax)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Scenario Dashboard: 2, 5 and 10 Year Frontier-AI Pressure", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    finish_figure(fig, "forecast_scenario_dashboard.png")


def plot_llm_cost_task_analysis(
    message_trends: pd.DataFrame,
    fixed_curves: pd.DataFrame,
    fixed_candidates: pd.DataFrame,
    divergence: pd.DataFrame,
) -> None:
    if not message_trends.empty:
        fig, ax1 = plt.subplots(figsize=(10.2, 6.2))
        ax1.plot(
            message_trends["year"],
            message_trends["modeled_average_message_cost_usd"],
            marker="o",
            linewidth=2.8,
            color=PALETTE["blue"],
            label="modeled average message/task cost",
        )
        ax1.set_ylabel("USD per modeled message/task")
        ax1.set_xlabel("Model release cohort year")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(
            message_trends["year"],
            message_trends["average_effective_tokens"],
            marker="s",
            linewidth=2.0,
            color=PALETTE["orange"],
            label="average effective tokens",
        )
        ax2.set_ylabel("Effective tokens in workload mix")
        ax1.set_title("Modeled LLM Cost per Message/Task")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
        add_note(ax1, "This is a workload-mix model over listed-price cohorts, not observed billing data. The point is to separate per-token price from task complexity.")
        soften_axes(ax1)
        soften_axes(ax2)
        finish_figure(fig, "llm_message_cost_trends.png")

    if not fixed_curves.empty:
        work = fixed_curves[fixed_curves["scenario"].isin(["current", "base"])].copy()
        fig, ax = plt.subplots(figsize=(11, 6.8))
        for task, group in work.groupby("display_name"):
            group = group.sort_values("target_year")
            ax.plot(group["target_year"], group["forecast_task_cost_usd"], marker="o", linewidth=2.1, label=task)
        ax.set_yscale("log")
        ax.set_xlabel("Target year")
        ax.set_ylabel("Cheapest adequate task cost, USD (log)")
        ax.set_title("Fixed Task Cost Curves")
        ax.grid(alpha=0.24)
        ax.legend(ncol=2, fontsize=8)
        add_note(ax, "A fixed task can get cheaper when more families clear the required capability threshold and quality-adjusted unit cost falls.")
        soften_axes(ax)
        finish_figure(fig, "fixed_task_cost_curves.png")

    if not divergence.empty:
        fig, ax = plt.subplots(figsize=(10.8, 6.3))
        for scenario, group in divergence.groupby("scenario"):
            group = group.sort_values("target_year")
            ax.plot(group["target_year"], group["message_cost_factor_vs_2026"], marker="o", linewidth=2.3, color=SCENARIO_COLORS.get(scenario, PALETTE["slate"]), label=f"{scenario} message/task")
            ax.plot(group["target_year"], group["fixed_task_cost_factor_vs_2026"], marker="s", linestyle="--", linewidth=2.0, color=SCENARIO_COLORS.get(scenario, PALETTE["slate"]), alpha=0.75, label=f"{scenario} fixed task")
        ax.axhline(1.0, color="#333333", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("Target year")
        ax.set_ylabel("Factor vs 2026 (log)")
        ax.set_title("Why Average Message Cost Can Rise While Fixed Task Cost Falls")
        ax.grid(alpha=0.24)
        ax.legend(ncol=2, fontsize=8)
        add_note(ax, "Solid lines track the workload mix becoming harder; dashed lines track the cheapest adequate model for fixed task profiles.")
        soften_axes(ax)
        finish_figure(fig, "cost_task_message_divergence.png")

    if not fixed_candidates.empty:
        work = fixed_candidates.copy()
        fig, ax = plt.subplots(figsize=(10.6, 6.4))
        colors = work["current_adequacy_status"].map({"adequate": PALETTE["teal"], "below_threshold": PALETTE["orange"]}).fillna(PALETTE["slate"])
        ax.scatter(
            work["current_task_cost_usd"],
            work["family_domain_score"],
            s=48,
            c=colors,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Current task cost, USD (log)")
        ax.set_ylabel("Family-domain benchmark score")
        ax.set_title("Task Quality-Cost Candidate Ladder")
        for _, row in work.sort_values("current_task_cost_usd").head(10).iterrows():
            ax.annotate(str(row["model_family"]), (row["current_task_cost_usd"], row["family_domain_score"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        add_note(ax, "Each point is a task-model candidate. Adequacy uses family-domain benchmark scores, so this remains a deployability screen rather than direct model proof.")
        ax.grid(alpha=0.24)
        soften_axes(ax)
        finish_figure(fig, "fixed_task_quality_cost_ladder.png")


def plot_analogy(analogies: pd.DataFrame) -> None:
    top = analogies.sort_values("ai_similarity_score").copy()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.barh(top["wave"], top["ai_similarity_score"], color=PALETTE["green"])
    ax.set_title("Historical Technology Wave Similarity to Frontier AI")
    ax.set_xlabel("Cosine similarity index")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "Speculative analogy index built from author-assigned subjective priors, not measured data. It frames where AI may resemble prior waves; treat every number as an opinion made inspectable.")
    soften_axes(ax)
    finish_figure(fig, "historical_analogy_index.png")


def plot_domain_heatmap(domain: pd.DataFrame) -> None:
    cols = [c for c in domain.columns if c.endswith("_share") or c.endswith("_bottleneck")]
    if not cols:
        return
    work = domain.sort_values("soc_major").set_index("soc_major")[cols].fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(work.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_task_share", "").replace("_bottleneck", " bottleneck") for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(work.index)))
    ax.set_yticklabels(work.index)
    ax.set_title("Task Domain Exposure by SOC Major Group")
    fig.colorbar(im, ax=ax, fraction=0.025)
    soften_axes(ax)
    finish_figure(fig, "task_domain_exposure_heatmap.png")


def plot_open_closed_gap(gap: pd.DataFrame) -> None:
    work = gap.dropna(subset=["open_closed_best_gap"]).sort_values("open_closed_best_gap").copy()
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.barh(work["category"], work["open_closed_best_gap"], color=PALETTE["purple"])
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Open vs Closed LMArena Best-Model Gap by Category")
    ax.set_xlabel("Closed/API best minus open-weight best, rating points")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "Categories without a comparable open-weight row are excluded here and called out in the table.")
    soften_axes(ax)
    finish_figure(fig, "open_closed_gap_by_category.png")

    levels = gap.dropna(subset=["closed_or_api", "open_weight"]).sort_values("closed_or_api").copy()
    if not levels.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(levels))
        ax.barh(y + 0.18, levels["closed_or_api"], height=0.34, color=PALETTE["blue"], label="closed/API best")
        ax.barh(y - 0.18, levels["open_weight"], height=0.34, color=PALETTE["orange"], label="open-weight best")
        ax.set_yticks(y)
        ax.set_yticklabels(levels["category"])
        ax.set_xlabel("Best observed LMArena rating")
        ax.set_title("Open and Closed Best Ratings by Category")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
        soften_axes(ax)
        finish_figure(fig, "open_closed_category_levels.png")


def plot_price_frontier(frontier: pd.DataFrame) -> None:
    work = frontier.dropna(subset=["family_best_lmarena", "blended_price_usd_per_1m"]).copy()
    work = work[work["blended_price_usd_per_1m"] > 0]
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    colors = np.where(work["price_performance_frontier"], PALETTE["red"], PALETTE["blue"])
    sizes = np.sqrt(numeric(work["context_window"]).fillna(1).clip(lower=1)) / 20
    ax.scatter(work["blended_price_usd_per_1m"], work["family_best_lmarena"], s=sizes.clip(18, 220), alpha=0.66, c=colors, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_title("Price-Performance Frontier")
    ax.set_xlabel("Blended output-heavy price, USD / 1M tokens (log)")
    ax.set_ylabel("Family best LMArena rating proxy")
    ax.grid(alpha=0.25)
    for _, row in work[work["price_performance_frontier"]].head(8).iterrows():
        ax.annotate(str(row["model_family"]), (row["blended_price_usd_per_1m"], row["family_best_lmarena"]), fontsize=8, alpha=0.8)
    add_note(ax, "Red points are non-dominated by this family-level proxy. Bubble size tracks context window, so cheap long-context models stand out without pretending they have direct model-level benchmark ratings.")
    soften_axes(ax)
    finish_figure(fig, "price_performance_frontier.png")

    context = work.dropna(subset=["context_window"]).copy()
    fig, ax = plt.subplots(figsize=(10.2, 6.5))
    scatter = ax.scatter(
        context["context_window"],
        context["blended_price_usd_per_1m"],
        c=context["family_best_lmarena"],
        cmap="viridis",
        s=np.where(context["price_performance_frontier"], 90, 24),
        alpha=0.7,
        edgecolor=np.where(context["price_performance_frontier"], PALETTE["red"], "white"),
        linewidth=np.where(context["price_performance_frontier"], 1.1, 0.35),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context window (tokens, log)")
    ax.set_ylabel("Blended output-heavy price, USD / 1M tokens (log)")
    ax.set_title("Context Window, Price and Rating Proxy")
    cb = fig.colorbar(scatter, ax=ax, fraction=0.035)
    cb.set_label("Family best LMArena rating proxy")
    add_note(ax, "This view shows why context length, price and rating proxy must be kept separate: large context is a product surface, not a benchmark score.")
    soften_axes(ax)
    finish_figure(fig, "price_context_rating_map.png")


def plot_leadership_probabilities(probabilities: pd.DataFrame) -> None:
    work = probabilities[probabilities["scenario"].eq("frontier_quality")].copy()
    top_families = work.groupby("model_family")["simulation_win_share"].max().sort_values(ascending=False).head(10).index
    work = work[work["model_family"].isin(top_families)].copy()
    pivot = work.pivot(index="model_family", columns="horizon_years", values="simulation_win_share").fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    y = np.arange(len(pivot))
    width = 0.24
    for i, horizon in enumerate(sorted(pivot.columns)):
        ax.barh(y + (i - 1) * width, pivot[horizon], height=width, label=f"{horizon}y")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Share of simulation draws")
    ax.set_title("Frontier-Quality Leadership Simulation Share")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    add_note(ax, "These are simulation shares under explicit component-weight scenarios. They are useful for stress-testing assumptions, not forecasting market odds.")
    soften_axes(ax)
    finish_figure(fig, "company_next_frontier_probabilities.png")

    rows = []
    for (scenario, horizon), group in probabilities.groupby(["scenario", "horizon_years"]):
        leader = group.sort_values("simulation_win_share", ascending=False).iloc[0]
        rows.append({"scenario": scenario, "horizon_years": horizon, "leader": leader["model_family"], "share": leader["simulation_win_share"]})
    matrix = pd.DataFrame(rows)
    pivot_share = matrix.pivot(index="scenario", columns="horizon_years", values="share").loc[
        ["frontier_quality", "balanced_lab_execution", "open_ecosystem_upside"]
    ]
    pivot_leader = matrix.pivot(index="scenario", columns="horizon_years", values="leader").loc[pivot_share.index]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(pivot_share.to_numpy(dtype=float), cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot_share.columns)))
    ax.set_xticklabels([f"{int(c)}y" for c in pivot_share.columns])
    ax.set_yticks(range(len(pivot_share.index)))
    ax.set_yticklabels([idx.replace("_", " ") for idx in pivot_share.index])
    ax.set_title("Scenario Leaders and Simulation Share")
    for i, scenario in enumerate(pivot_share.index):
        for j, horizon in enumerate(pivot_share.columns):
            ax.text(j, i, f"{pivot_leader.loc[scenario, horizon]}\n{pivot_share.loc[scenario, horizon] * 100:.0f}%", ha="center", va="center", color="white" if pivot_share.loc[scenario, horizon] > 0.45 else PALETTE["ink"], fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, label="Top family simulation share")
    finish_figure(fig, "leadership_scenario_matrix.png")


def plot_labor_clusters(cluster_profiles: pd.DataFrame) -> None:
    work = cluster_profiles.sort_values("full_job_automation_feasibility_index").copy()
    labels = work["cluster_label"] + " (" + work["labor_cluster_id"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.barh(labels, work["full_job_automation_feasibility_index"], color=PALETTE["orange"], label="replacement feasibility")
    ax.scatter(work["augmentation_index"], labels, color=PALETTE["blue"], s=70, label="augmentation")
    ax.set_title("Labor Clusters: Replacement Feasibility vs Augmentation")
    ax.set_xlabel("Cluster mean index")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    add_note(ax, "Bars and dots separate the two labor stories: replacement feasibility can stay lower than augmentation pressure even in exposed clusters.")
    soften_axes(ax)
    finish_figure(fig, "labor_cluster_profiles.png")


def plot_labor_outcome_mix(labor_summary: pd.DataFrame) -> None:
    work = labor_summary[labor_summary["grouping"].eq("dominant_outcome")].copy()
    if work.empty:
        return
    work = work.sort_values("labor_weight_sum", ascending=True)
    colors = work["group"].map(
        {
            "replacement_candidate": PALETTE["red"],
            "augmentation_first": PALETTE["teal"],
            "bottleneck_protected": PALETTE["green"],
            "mixed_redesign": PALETTE["gold"],
        }
    ).fillna(PALETTE["slate"])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(work["group"].str.replace("_", " "), work["labor_weight_sum"], color=colors)
    ax.set_xlabel("Labor-weight proxy")
    ax.set_title("Labor-Weighted Dominant Outcome Mix")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "The weight is a public-data proxy, not an employment forecast. It helps keep the report from over-indexing on a few eye-catching occupations.")
    soften_axes(ax)
    finish_figure(fig, "labor_outcome_mix.png")


def plot_replacement_feasibility(replacement: pd.DataFrame) -> None:
    top = replacement.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    ax.barh(wrap_tick_labels(top["title"], 30), top["full_job_automation_feasibility_index"], color=PALETTE["red"])
    ax.set_title("Highest Whole-Job Automation Feasibility")
    ax.set_xlabel("Feasibility index after bottleneck gates")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "This chart is intentionally stricter than task exposure: substitution pressure is gated by physical, trust, regulatory and task-coverage bottlenecks.")
    soften_axes(ax)
    finish_figure(fig, "job_replacement_feasibility.png")


def plot_direct_vs_proxy_price_performance(direct: pd.DataFrame, proxy: pd.DataFrame) -> None:
    direct_work = direct.dropna(subset=["direct_lmarena_rating", "blended_price_usd_per_1m"]).copy()
    proxy_work = proxy.dropna(subset=["family_best_lmarena", "blended_price_usd_per_1m"]).copy()
    if direct_work.empty or proxy_work.empty:
        return
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.scatter(
        proxy_work["blended_price_usd_per_1m"],
        proxy_work["family_best_lmarena"],
        s=28,
        alpha=0.24,
        color=PALETTE["slate"],
        label="family proxy",
    )
    ax.scatter(
        direct_work["blended_price_usd_per_1m"],
        direct_work["direct_lmarena_rating"],
        s=np.where(direct_work["direct_price_performance_frontier"], 88, 44),
        alpha=0.78,
        color=PALETTE["teal"],
        edgecolor="white",
        linewidth=0.5,
        label="direct model match",
    )
    for _, row in direct_work.sort_values("direct_price_performance_index", ascending=False).head(8).iterrows():
        ax.annotate(str(row["model_family"]), (row["blended_price_usd_per_1m"], row["direct_lmarena_rating"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Blended output-heavy price, USD / 1M tokens (log)")
    ax.set_ylabel("LMArena rating")
    ax.set_title("Direct Model Evidence vs Family Proxy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    add_note(ax, "Direct points require model-level name matches. Proxy points keep the broader deployability screen visible without upgrading family evidence to model-level proof.")
    soften_axes(ax)
    finish_figure(fig, "direct_vs_proxy_price_performance.png")


def plot_vendor_scores(vendor_scores: pd.DataFrame, company_scores: pd.DataFrame) -> None:
    top = vendor_scores.sort_values("vendor_frontier_portfolio_score").copy()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(top["vendor"], top["vendor_frontier_portfolio_score"], color=PALETTE["blue"])
    ax.set_title("Vendor Frontier Portfolio Score")
    ax.set_xlabel("Heuristic vendor portfolio score")
    ax.grid(axis="x", alpha=0.25)
    for _, row in top.iterrows():
        ax.text(row["vendor_frontier_portfolio_score"] + 1, row["vendor"], f"{row['vendor_frontier_portfolio_score']:.1f}", va="center", fontsize=8, color=PALETTE["muted"])
    add_note(ax, "Vendor score combines flagship family signal with evidence-weighted portfolio breadth. It is a business-facing companion to family rank, not a replacement.")
    soften_axes(ax)
    finish_figure(fig, "vendor_frontier_scores.png")

    family = company_scores[["model_family", "rank"]].copy()
    family["vendor"] = family["model_family"].map(FAMILY_VENDOR_MAP).fillna(family["model_family"])
    rank_shift = family.merge(vendor_scores[["vendor", "rank"]].rename(columns={"rank": "vendor_rank"}), on="vendor", how="left")
    rank_shift["rank_shift"] = rank_shift["vendor_rank"] - rank_shift["rank"]
    work = rank_shift.sort_values("rank_shift").copy()
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    colors = np.where(work["rank_shift"] > 0, PALETTE["orange"], PALETTE["teal"])
    ax.barh(work["model_family"], work["rank_shift"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Family Rank vs Vendor Portfolio Rank Shift")
    ax.set_xlabel("Vendor rank minus family rank")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "Negative values mean the vendor portfolio rank is stronger than the individual family rank; positive values mean the opposite.")
    soften_axes(ax)
    finish_figure(fig, "family_vs_vendor_rank_shift.png")


def plot_source_coverage(source_coverage: pd.DataFrame, family_coverage: pd.DataFrame) -> None:
    if not source_coverage.empty:
        cols = ["release_date_coverage", "price_coverage", "access_class_coverage", "benchmark_value_coverage", "context_window_coverage", "organization_vendor_coverage"]
        work = source_coverage.set_index("table")[cols].fillna(0)
        work = work.loc[work.mean(axis=1).sort_values().tail(12).index]
        fig, ax = plt.subplots(figsize=(12.2, 6.8))
        im = ax.imshow(work.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("_coverage", "").replace("_", " ") for c in cols], rotation=35, ha="right")
        ax.set_yticks(range(len(work.index)))
        ax.set_yticklabels(work.index)
        ax.set_title("Source Freshness and Coverage Dashboard")
        fig.colorbar(im, ax=ax, fraction=0.025, label="Coverage share")
        soften_axes(ax)
        finish_figure(fig, "source_coverage_dashboard.png")

    if not family_coverage.empty:
        cols = ["price_coverage", "release_date_coverage", "access_class_coverage", "context_window_coverage", "labor_signal_coverage"]
        work = family_coverage.sort_values("coverage_score").set_index("model_family")[cols].fillna(0)
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        im = ax.imshow(work.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("_coverage", "").replace("_", " ") for c in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(work.index)))
        ax.set_yticklabels(work.index)
        ax.set_title("Family Signal Coverage Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.03, label="Coverage share")
        soften_axes(ax)
        finish_figure(fig, "family_signal_coverage_heatmap.png")


def plot_rank_uncertainty(rank_intervals: pd.DataFrame) -> None:
    work = rank_intervals.sort_values("score_p50").copy()
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    xerr = np.vstack([work["score_p50"] - work["score_p10"], work["score_p90"] - work["score_p50"]])
    colors = work["rank_stability_label"].map({"stable": PALETTE["teal"], "moderate": PALETTE["gold"], "unstable": PALETTE["red"]}).fillna(PALETTE["slate"])
    ax.errorbar(work["score_p50"], work["model_family"], xerr=xerr, fmt="none", ecolor="#9aa7b4", elinewidth=2, capsize=4)
    ax.scatter(work["score_p50"], work["model_family"], color=colors, s=60, zorder=3)
    ax.set_title("Frontier Rank Uncertainty")
    ax.set_xlabel("Bootstrap heuristic index interval")
    ax.grid(axis="x", alpha=0.25)
    add_note(ax, "Intervals are evidence-scaled bootstrap stress tests, not calibrated confidence intervals. Labels come from rank IQR.")
    soften_axes(ax)
    finish_figure(fig, "frontier_rank_uncertainty.png")


def plot_forecast_uncertainty_bands(forecasts: pd.DataFrame) -> None:
    metrics = [
        "frontier_output_price_factor",
        "open_weight_lmarena_gap_remaining",
        "share_of_us_occupation_tasks_materially_touched",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric in zip(axes, metrics):
        pivot = forecasts[forecasts["metric"].eq(metric)].pivot(index="target_year", columns="scenario", values="value").sort_index()
        if pivot.empty:
            continue
        lower = pivot.min(axis=1)
        upper = pivot.max(axis=1)
        base = pivot["base"] if "base" in pivot.columns else pivot.mean(axis=1)
        ax.fill_between(pivot.index.astype(int), lower, upper, color=PALETTE["blue"], alpha=0.18)
        ax.plot(pivot.index.astype(int), base, marker="o", color=PALETTE["blue"], linewidth=2)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Target year")
        ax.grid(alpha=0.22)
        soften_axes(ax)
    fig.suptitle("Forecast Scenario Envelopes, Not Calibrated Confidence Bands", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    finish_figure(fig, "forecast_uncertainty_bands.png")


def plot_business_domain_pressure(domain_pressure: pd.DataFrame) -> None:
    work = domain_pressure.sort_values("disruption_index").copy()
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.barh(work["business_domain"].str.replace("_", " "), work["disruption_index"], color=PALETTE["teal"], label="disruption")
    ax.scatter(work["augmentation_index"], work["business_domain"].str.replace("_", " "), color=PALETTE["blue"], s=58, label="augmentation")
    ax.scatter(work["replacement_feasibility_index"], work["business_domain"].str.replace("_", " "), color=PALETTE["red"], s=58, label="replacement feasibility")
    ax.set_title("Business Domain AI Pressure Matrix")
    ax.set_xlabel("Weighted index")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    add_note(ax, "The domain view translates occupation-level exposure into business language while keeping augmentation and replacement separate.")
    soften_axes(ax)
    finish_figure(fig, "business_domain_pressure_matrix.png")


def plot_release_cadence(family_cadence: pd.DataFrame, vendor_cadence: pd.DataFrame) -> None:
    if not family_cadence.empty:
        work = family_cadence.sort_values("recent_releases_365d").copy()
        fig, ax = plt.subplots(figsize=(10.4, 6.2))
        ax.barh(work["model_family"], work["recent_releases_365d"], color=PALETTE["gold"])
        ax.set_title("Recent Release Velocity by Family")
        ax.set_xlabel("Visible releases in last 365 days")
        ax.grid(axis="x", alpha=0.25)
        soften_axes(ax)
        finish_figure(fig, "recent_release_velocity.png")

    if not vendor_cadence.empty:
        work = vendor_cadence.sort_values("total_releases", ascending=False).head(10).copy()
        fig, ax = plt.subplots(figsize=(10.6, 6.2))
        ax.scatter(
            work["median_days_between_releases"].fillna(work["median_days_between_releases"].max()),
            work["recent_releases_365d"],
            s=np.sqrt(work["total_releases"].clip(lower=1)) * 70,
            color=PALETTE["purple"],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in work.iterrows():
            ax.annotate(str(row["vendor"]), (row["median_days_between_releases"], row["recent_releases_365d"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title("Release Cadence Timeline Summary")
        ax.set_xlabel("Median days between visible releases")
        ax.set_ylabel("Recent releases in last 365 days")
        ax.grid(alpha=0.25)
        add_note(ax, "Bubble size tracks total visible releases. This is visible public cadence, not a complete internal product roadmap.")
        soften_axes(ax)
        finish_figure(fig, "release_cadence_timeline.png")


def plot_domain_benchmark_analysis(
    domain_catalog: pd.DataFrame,
    domain_frontier: pd.DataFrame,
    domain_velocity: pd.DataFrame,
    domain_forecasts: pd.DataFrame,
    domain_thresholds: pd.DataFrame,
) -> None:
    if not domain_catalog.empty:
        work = domain_catalog.sort_values("normalized_result_rows").copy()
        fig, ax = plt.subplots(figsize=(11, 6.8))
        colors = work["coverage_label"].map({"broad": PALETTE["teal"], "moderate": PALETTE["gold"], "thin": PALETTE["red"]}).fillna(PALETTE["slate"])
        ax.barh(work["domain_label"], work["normalized_result_rows"], color=colors)
        ax.set_xscale("log")
        ax.set_title("Benchmark Coverage by Capability Domain")
        ax.set_xlabel("Normalized benchmark result rows (log)")
        ax.grid(axis="x", alpha=0.25)
        for _, row in work.iterrows():
            ax.text(row["normalized_result_rows"] * 1.05, row["domain_label"], f"{int(row['source_count'])} sources", va="center", fontsize=8, color=PALETTE["muted"])
        add_note(ax, "Coverage is deliberately visible because domain forecasts are only as useful as the source breadth behind them.")
        soften_axes(ax)
        finish_figure(fig, "domain_benchmark_coverage.png")

        matrix = domain_catalog.set_index("domain_label")[["source_count", "benchmark_count", "model_count", "family_count"]].fillna(0)
        matrix = matrix.loc[matrix["source_count"].sort_values().index]
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        im = ax.imshow(np.log1p(matrix.to_numpy(dtype=float)), aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([c.replace("_", " ") for c in matrix.columns], rotation=25, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_title("Domain Source Matrix")
        for i, label in enumerate(matrix.index):
            for j, col in enumerate(matrix.columns):
                ax.text(j, i, f"{int(matrix.loc[label, col])}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
        fig.colorbar(im, ax=ax, fraction=0.028, label="log(1 + count)")
        soften_axes(ax)
        finish_figure(fig, "domain_source_matrix.png")

    if not domain_frontier.empty:
        top_domains = domain_velocity.sort_values("current_frontier_score", ascending=False).head(10)["domain"].tolist()
        work = domain_frontier[domain_frontier["domain"].isin(top_domains)].copy()
        fig, ax = plt.subplots(figsize=(11.5, 6.7))
        for domain, group in work.groupby("domain"):
            group = group.sort_values("eval_year")
            ax.plot(group["eval_year"], group["frontier_score"], marker="o", linewidth=2, label=domain_label(domain))
        ax.set_ylim(0, 104)
        ax.set_title("Observed Domain Frontier Trend")
        ax.set_xlabel("Evaluation year")
        ax.set_ylabel("Frontier score, normalized 0-100")
        ax.grid(alpha=0.24)
        ax.legend(ncol=2, fontsize=8)
        add_note(ax, "Scores are normalized across heterogeneous benchmarks. The trend is useful for direction and relative velocity, not exact cross-domain psychometrics.")
        soften_axes(ax)
        finish_figure(fig, "domain_frontier_trends.png")

    if not domain_velocity.empty:
        work = domain_velocity.sort_values("current_frontier_score").copy()
        fig, ax = plt.subplots(figsize=(11, 6.8))
        ax.barh(work["domain_label"], work["current_frontier_score"], color=PALETTE["blue"], label="current frontier score")
        ax.scatter(work["annual_frontier_point_gain_used"] * 4, work["domain_label"], color=PALETTE["orange"], s=68, label="annual point gain used x4")
        ax.set_xlim(0, 105)
        ax.set_title("Current Capability vs Improvement Velocity by Domain")
        ax.set_xlabel("Normalized score / scaled annual gain")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False)
        add_note(ax, "Orange dots are scaled so velocity can be read beside current level. Thin domains use the cross-domain fallback and are labeled in the table.")
        soften_axes(ax)
        finish_figure(fig, "domain_current_velocity.png")

    if not domain_forecasts.empty:
        base = domain_forecasts[domain_forecasts["scenario"].eq("base")].copy()
        top_domains = domain_velocity.sort_values("current_frontier_score", ascending=False).head(10)["domain"].tolist()
        base = base[base["domain"].isin(top_domains)]
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        for domain, group in base.groupby("domain"):
            group = group.sort_values("target_year")
            ax.plot(group["target_year"], group["forecast_frontier_score"], marker="o", linewidth=2.1, label=domain_label(domain))
        ax.set_ylim(0, 104)
        ax.set_title("Base Scenario Domain Capability Forecast")
        ax.set_xlabel("Target year")
        ax.set_ylabel("Forecast frontier score, normalized 0-100")
        ax.grid(alpha=0.24)
        ax.legend(ncol=2, fontsize=8)
        add_note(ax, "Forecasts use bounded gap closure from observed domain trends. They become more speculative when source coverage is thin.")
        soften_axes(ax)
        finish_figure(fig, "domain_forecast_base.png")

        selected = domain_forecasts[domain_forecasts["domain"].isin(domain_velocity.head(6)["domain"])].copy()
        if not selected.empty:
            domains = list(dict.fromkeys(selected["domain"].tolist()))[:6]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharey=True)
            for ax, domain in zip(axes.flatten(), domains):
                subset = selected[selected["domain"].eq(domain)]
                for scenario, group in subset.groupby("scenario"):
                    group = group.sort_values("target_year")
                    ax.plot(group["target_year"], group["forecast_frontier_score"], marker="o", linewidth=1.9, color=SCENARIO_COLORS.get(scenario, PALETTE["slate"]), label=scenario)
                ax.set_title(domain_label(domain))
                ax.set_ylim(0, 104)
                ax.grid(alpha=0.20)
                soften_axes(ax)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle("Domain Forecast Scenario Small Multiples", fontsize=15, fontweight="bold", color=PALETTE["ink"])
            finish_figure(fig, "domain_forecast_scenarios.png")

    if not domain_thresholds.empty:
        work = domain_thresholds[domain_thresholds["threshold_score"].eq(90)].copy()
        work = work.sort_values("base_years_to_threshold", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6.4))
        values = numeric(work["base_years_to_threshold"]).clip(upper=12)
        ax.barh(work["domain_label"], values, color=PALETTE["purple"])
        ax.set_title("Estimated Years to 90/100 Domain Frontier Score")
        ax.set_xlabel("Base scenario years from 2026 (capped at 12 for display)")
        ax.grid(axis="x", alpha=0.25)
        add_note(ax, "A zero means the normalized 90 threshold is already observed in this public benchmark panel. Thin-domain estimates should be read cautiously.")
        soften_axes(ax)
        finish_figure(fig, "domain_threshold_timeline.png")


def build_plots(
    company_scores: pd.DataFrame,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    analogies: pd.DataFrame,
    domain: pd.DataFrame,
    gap: pd.DataFrame,
    price_frontier: pd.DataFrame,
    probabilities: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    labor_summary: pd.DataFrame,
    replacement: pd.DataFrame,
    match_audit: pd.DataFrame,
    direct_price: pd.DataFrame,
    vendor_scores: pd.DataFrame,
    source_coverage: pd.DataFrame,
    family_coverage: pd.DataFrame,
    rank_intervals: pd.DataFrame,
    business_domain_pressure: pd.DataFrame,
    release_cadence_family: pd.DataFrame,
    release_cadence_vendor: pd.DataFrame,
    domain_catalog: pd.DataFrame,
    domain_frontier: pd.DataFrame,
    domain_velocity: pd.DataFrame,
    domain_forecasts: pd.DataFrame,
    domain_thresholds: pd.DataFrame,
    message_cost_trends: pd.DataFrame,
    fixed_task_cost_curves: pd.DataFrame,
    fixed_task_cost_candidates: pd.DataFrame,
    cost_divergence_scenarios: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_chart_theme()
    plot_company_scores(company_scores)
    plot_job_scores(jobs)
    plot_forecasts(forecasts)
    plot_analogy(analogies)
    plot_domain_heatmap(domain)
    plot_open_closed_gap(gap)
    plot_price_frontier(price_frontier)
    plot_leadership_probabilities(probabilities)
    plot_labor_clusters(cluster_profiles)
    plot_labor_outcome_mix(labor_summary)
    plot_replacement_feasibility(replacement)
    plot_direct_vs_proxy_price_performance(direct_price, price_frontier)
    plot_vendor_scores(vendor_scores, company_scores)
    plot_source_coverage(source_coverage, family_coverage)
    plot_rank_uncertainty(rank_intervals)
    plot_forecast_uncertainty_bands(forecasts)
    plot_business_domain_pressure(business_domain_pressure)
    plot_release_cadence(release_cadence_family, release_cadence_vendor)
    plot_domain_benchmark_analysis(domain_catalog, domain_frontier, domain_velocity, domain_forecasts, domain_thresholds)
    plot_llm_cost_task_analysis(message_cost_trends, fixed_task_cost_curves, fixed_task_cost_candidates, cost_divergence_scenarios)


def source_registry() -> pd.DataFrame:
    rows = [
        {
            "source_id": "anthropic_economic_index",
            "name": "Anthropic Economic Index public dataset",
            "url": "https://huggingface.co/datasets/Anthropic/EconomicIndex",
            "used_for": "Observed occupation exposure, task penetration, augmentation/automation modes, wage and employment companion data.",
            "license_or_access": "Public Hugging Face dataset; verify dataset card for current license.",
        },
        {
            "source_id": "onet_task_statements",
            "name": "O*NET task statements via Anthropic Economic Index release files",
            "url": "https://www.onetcenter.org/database.html",
            "used_for": "Task text features and bottleneck scoring by occupation.",
            "license_or_access": "O*NET Database public files, redistributed in AEI release inputs.",
        },
        {
            "source_id": "bls_oews_wage_employment",
            "name": "BLS employment and wage companion files in AEI release",
            "url": "https://www.bls.gov/oes/",
            "used_for": "Occupation wage proxy and labor-market scale where available.",
            "license_or_access": "Public BLS source data. BLS anti-bot policy blocked direct xlsx download in this runtime, so cached public AEI companion files are used.",
        },
        {
            "source_id": "ai_capability_signals_rich_dataset",
            "name": "Local rich frontier AI dataset package",
            "url": str((DATASET / "README.md").relative_to(ROOT)),
            "used_for": "Company scoring, model benchmarks, prices, release cadence, research and ecosystem indicators.",
            "license_or_access": "Derived from public APIs and datasets listed in data/dataset/source_registry_rich.csv.",
        },
        {
            "source_id": "livecodebench_leaderboard",
            "name": "LiveCodeBench public leaderboard and data",
            "url": "https://livecodebench.github.io/",
            "used_for": "Coding-domain pass@1 rows by model, difficulty and release window.",
            "license_or_access": "Public project data; benchmark paper and Hugging Face assets list license details.",
        },
        {
            "source_id": "open_medical_llm_leaderboard",
            "name": "Open Medical-LLM Leaderboard result files",
            "url": "https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard",
            "used_for": "Medical-domain MedQA, MedMCQA, PubMedQA and MMLU medical subset accuracy rows.",
            "license_or_access": "Public Hugging Face space and results dataset.",
        },
        {
            "source_id": "terminal_bench_2_0",
            "name": "Terminal-Bench 2.0 leaderboard",
            "url": TERMINAL_BENCH_20_URL,
            "used_for": "Agentic terminal-work benchmark scores across agents/models.",
            "license_or_access": "Public leaderboard; agent/model scores are source-visible.",
        },
        {
            "source_id": "financebench_results",
            "name": "FinanceBench public results",
            "url": "https://huggingface.co/datasets/financebench/results",
            "used_for": "Finance RAG benchmark rows in the finance-domain panel.",
            "license_or_access": "Public Hugging Face dataset.",
        },
        {
            "source_id": "qfbench_v11",
            "name": "QFBench V11 public leaderboard",
            "url": QFBENCH_URL,
            "used_for": "Quantitative-finance coding/agent benchmark scores.",
            "license_or_access": "Public project website and GitHub repository.",
        },
        {
            "source_id": "lexometrica_legal_ru_v1",
            "name": "Lexometrica Ground Truth LegalBench RU",
            "url": LEXOMETRICA_URL,
            "used_for": "Legal reasoning composite and citation-validity rows.",
            "license_or_access": "Public black-box leaderboard; prompts/cases are intentionally not published.",
        },
        {
            "source_id": "cost_external_evidence",
            "name": "Cost-per-message and fixed-task cost evidence notes",
            "url": str((ANALYSIS / "cost_external_evidence.csv").relative_to(ROOT)),
            "used_for": "Separating observed/listed API price signals from scenario priors about quality-adjusted fixed-task cost decline and agentic token amplification.",
            "license_or_access": "Source URLs are listed in the generated evidence table.",
        },
    ]
    return write_table(pd.DataFrame(rows), "deep_analysis_source_registry")


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    return report_table(df[cols].head(n))


def report_table(df: pd.DataFrame) -> str:
    out = df.copy()
    if "simulation_win_share" in out.columns:
        out["simulation_win_share"] = out["simulation_win_share"].map(
            lambda value: f"{float(value) * 100:.1f}%" if pd.notna(value) and np.isfinite(float(value)) else "n/a"
        )
    out = out.astype(object).where(pd.notna(out), "n/a")
    return out.to_markdown(index=False)


def format_number(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def format_share(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def build_dashboard_key_findings(
    company_scores: pd.DataFrame,
    jobs: pd.DataFrame,
    probabilities: pd.DataFrame,
    gap: pd.DataFrame,
    direct_price: pd.DataFrame,
    source_coverage: pd.DataFrame,
    family_coverage: pd.DataFrame,
    rank_intervals: pd.DataFrame,
    failure_modes: pd.DataFrame,
    business_domain_pressure: pd.DataFrame,
    release_cadence_family: pd.DataFrame,
    domain_catalog: pd.DataFrame,
    domain_velocity: pd.DataFrame,
    domain_forecasts: pd.DataFrame,
    message_cost_trends: pd.DataFrame,
    fixed_task_cost_curves: pd.DataFrame,
    cost_divergence_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        question: str,
        headline: str,
        metric: str,
        reading: str,
        evidence_level: str,
        primary_artifact: str,
        priority_order: int,
    ) -> None:
        rows.append(
            {
                "section": section,
                "question": question,
                "headline": headline,
                "metric": metric,
                "reading": reading,
                "evidence_level": evidence_level,
                "primary_artifact": primary_artifact,
                "priority_order": priority_order,
            }
        )

    top_company = company_scores.iloc[0]
    stable_top_count = int(company_scores["sensitivity_label"].eq("stable_top_tier").sum())
    add(
        "Models",
        "Who leads the current frontier-family signal?",
        str(top_company["model_family"]),
        f"{format_number(top_company['frontier_momentum_heuristic_index'])} heuristic index",
        f"{stable_top_count} families stay in the top tier across sensitivity variants; read the rank as a weighted public-signal index, not a universal model score.",
        "observed",
        "company_frontier_scores.csv",
        1,
    )

    if not domain_velocity.empty:
        fast_domain = domain_velocity.sort_values("annual_frontier_point_gain_used", ascending=False).iloc[0]
        add(
            "Domains",
            "Which capability field is improving fastest in the public panel?",
            str(fast_domain["domain_label"]),
            f"{format_number(fast_domain['annual_frontier_point_gain_used'])} points/year used",
            "The domain panel normalizes coding, medicine, math, legal, finance, vision, search/document and agentic benchmark scores into one auditable long table before forecasting.",
            "observed" if fast_domain["slope_source"] == "median_within_benchmark_slope" else "scenario",
            "domain_improvement_velocity.csv",
            2,
        )

    if not domain_catalog.empty:
        broad_count = int(domain_catalog["coverage_label"].eq("broad").sum())
        add(
            "Coverage",
            "How many capability domains have broad benchmark coverage?",
            f"{broad_count} broad domains",
            f"{int(domain_catalog['normalized_result_rows'].sum()):,} normalized rows",
            "Thin domains are not hidden: legal and finance are useful but carry stronger caveats than coding, language or arena-backed categories.",
            "observed",
            "domain_benchmark_catalog.csv",
            3,
        )

    direct_counts = family_coverage.sort_values("direct_benchmark_match_count", ascending=False).iloc[0]
    add(
        "Evidence",
        "Where is model-level evidence strongest?",
        str(direct_counts["model_family"]),
        f"{int(direct_counts['direct_benchmark_match_count'])} direct benchmark matches",
        "Direct matches are the cleanest deployability evidence; family-only rows stay visible so the report does not borrow certainty from proxy data.",
        "direct_match",
        "family_coverage_matrix.csv",
        4,
    )

    fq_10y = probabilities[(probabilities["scenario"].eq("frontier_quality")) & (probabilities["horizon_years"].eq(10))].sort_values(
        "simulation_win_share",
        ascending=False,
    )
    if not fq_10y.empty:
        leader = fq_10y.iloc[0]
        add(
            "Forecast",
            "Who wins the 10-year frontier-quality stress test?",
            str(leader["model_family"]),
            f"{format_share(leader['simulation_win_share'])} simulation share",
            "This is a sensitivity share from perturbed component weights, not a calibrated probability or prediction-market number.",
            "scenario",
            "company_next_frontier_probabilities.csv",
            5,
        )

    open_10y = probabilities[
        (probabilities["scenario"].eq("open_ecosystem_upside")) & (probabilities["horizon_years"].eq(10))
    ].sort_values("simulation_win_share", ascending=False)
    if not open_10y.empty:
        leader = open_10y.iloc[0]
        add(
            "Open Ecosystem",
            "Who benefits if distribution, openness and cost matter more?",
            str(leader["model_family"]),
            f"{format_share(leader['simulation_win_share'])} simulation share",
            "The answer can differ from raw frontier-quality leadership because openness, cost and ecosystem pull are separate adoption axes.",
            "scenario",
            "company_next_frontier_probabilities.csv",
            6,
        )

    if not domain_forecasts.empty:
        base_2036 = domain_forecasts[(domain_forecasts["scenario"].eq("base")) & (domain_forecasts["horizon_years"].eq(10))].sort_values("forecast_frontier_score", ascending=False)
        base_2036 = base_2036[~base_2036["confidence"].eq("low")] if (~base_2036["confidence"].eq("low")).any() else base_2036
        if not base_2036.empty:
            leader = base_2036.iloc[0]
            add(
                "Domain Forecast",
                "Which field has the highest 2036 base-case frontier score?",
                str(leader["domain_label"]),
                f"{format_number(leader['forecast_frontier_score'])}/100 forecast score",
                "The forecast is bounded by a 100-point frontier scale and uses gap closure, so it cannot grow without limit.",
                "scenario",
                "domain_capability_forecasts.csv",
                7,
            )

    gap_values = pd.to_numeric(gap["open_closed_best_gap"], errors="coerce")
    if gap_values.notna().any():
        gap_row = gap.loc[gap_values.idxmax()]
        add(
            "Open vs Closed",
            "Where is the open-vs-closed gap largest?",
            str(gap_row["category"]).replace("_", " "),
            f"{format_number(gap_row['open_closed_best_gap'])} arena rating points",
            "The report treats catch-up as category-specific. A single open-vs-closed headline hides large differences by task domain.",
            "observed",
            "open_closed_gap_by_category.csv",
            8,
        )

    if not direct_price.empty:
        efficient = direct_price.sort_values("direct_price_performance_index", ascending=False).iloc[0]
        add(
            "Economics",
            "Which directly matched model sits highest on price-performance?",
            str(efficient["canonical_model"]),
            f"{format_number(efficient['direct_price_performance_index'])} direct index",
            "This view uses direct model-level benchmark evidence, avoiding the family-proxy shortcut used in the broader efficient-frontier screen.",
            "direct_match",
            "direct_model_price_performance.csv",
            9,
        )

    if not message_cost_trends.empty:
        latest_message = message_cost_trends.sort_values("year").tail(1).iloc[0]
        add(
            "Economics",
            "Is average message/task cost rising in the modeled workload mix?",
            f"{latest_message['modeled_average_message_cost_usd']:.3f} USD",
            f"{latest_message['message_cost_index_2023_100']:.1f} index vs 2023",
            "The trend is workload-weighted: long-context and agentic runs gain share, so average work-unit cost can rise even when low-end token prices improve.",
            "scenario",
            "llm_message_cost_trends.csv",
            10,
        )

    thesis = fixed_task_cost_curves[
        fixed_task_cost_curves["task_profile"].eq("thesis_quality_longform")
        & fixed_task_cost_curves["scenario"].eq("base")
        & fixed_task_cost_curves["horizon_years"].eq(10)
    ]
    if not thesis.empty:
        row = thesis.iloc[0]
        add(
            "Fixed Task Cost",
            "What happens to a thesis-quality fixed writing task?",
            str(row["selected_family"]),
            f"{float(row['forecast_task_cost_usd']):.4f} USD in {int(row['target_year'])}",
            "The fixed-task screen chooses the cheapest adequate family proxy for a stable token budget, so it can fall while frontier workload cost rises.",
            "scenario",
            "fixed_task_cost_curves.csv",
            11,
        )

    labor = jobs.iloc[0]
    add(
        "Labor",
        "Which occupation has the highest near-term pressure index?",
        str(labor["title"]),
        f"{format_number(labor['near_term_disruption_index'])} disruption index",
        "The score combines observed exposure, task structure and bottlenecks; it is pressure for redesign, not a claim that the occupation disappears.",
        "observed",
        "job_exposure_scores.csv",
        12,
    )

    replacement = jobs.sort_values("full_job_automation_feasibility_index", ascending=False).iloc[0]
    add(
        "Labor",
        "Where is whole-job replacement most feasible after gates?",
        str(replacement["title"]),
        f"{format_number(replacement['full_job_automation_feasibility_index'])} feasibility index",
        "The replacement gate keeps physical, trust, regulatory and accountability bottlenecks in the calculation before labeling any role replaceable.",
        "observed",
        "job_replacement_feasibility.csv",
        13,
    )

    domain = business_domain_pressure.sort_values("disruption_index", ascending=False).iloc[0]
    add(
        "Workflows",
        "Which business domain should a reader inspect first?",
        str(domain["business_domain"]).replace("_", " "),
        f"{format_number(domain['disruption_index'])} disruption index",
        "Domain pressure translates occupation evidence into business language while preserving example occupations and human gates.",
        "family_proxy",
        "business_domain_ai_pressure.csv",
        14,
    )

    cadence = release_cadence_family.sort_values("recent_releases_365d", ascending=False).iloc[0]
    add(
        "Execution",
        "Which family shows the most visible recent release velocity?",
        str(cadence["model_family"]),
        f"{int(cadence['recent_releases_365d'])} releases in 365 days",
        "Cadence is a public execution signal and should be read beside quality, price and evidence depth rather than as a standalone rank.",
        "observed",
        "release_cadence_by_family.csv",
        15,
    )

    latest_dates = pd.to_datetime(source_coverage["latest_source_date"], errors="coerce").dropna()
    add(
        "Coverage",
        "How fresh is the visible source layer?",
        latest_dates.max().date().isoformat() if not latest_dates.empty else "n/a",
        f"{int(source_coverage['rows'].sum()):,} source rows tracked",
        "Freshness and missingness are surfaced before the report leans on rankings, which makes stale-source risk easier to spot.",
        "observed",
        "source_coverage_diagnostics.csv",
        16,
    )

    stable_ranks = int(rank_intervals["rank_stability_label"].eq("stable").sum())
    add(
        "Uncertainty",
        "How many family ranks are stable under evidence-scaled stress?",
        f"{stable_ranks} stable rank bands",
        f"{len(rank_intervals)} families stress-tested",
        "Rank bands are sensitivity diagnostics rather than calibrated confidence intervals.",
        "scenario",
        "rank_stability_intervals.csv",
        17,
    )

    high_failures = int(failure_modes["severity"].eq("high").sum())
    add(
        "Risk",
        "What should a reviewer challenge first?",
        f"{high_failures} high-severity assumptions",
        "Named failure modes",
        "The skeptical layer is part of the product: it names how the analysis can break and points to mitigation artifacts.",
        "speculative",
        "claim_failure_modes.csv",
        18,
    )

    return write_table(pd.DataFrame(rows).sort_values("priority_order"), "dashboard_key_findings")


def write_report(
    company_scores: pd.DataFrame,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    analogies: pd.DataFrame,
    claims: pd.DataFrame,
    probabilities: pd.DataFrame,
    gap: pd.DataFrame,
    price_frontier: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    labor_summary: pd.DataFrame,
    replacement: pd.DataFrame,
    findings: pd.DataFrame,
    match_audit: pd.DataFrame,
    direct_price: pd.DataFrame,
    vendor_scores: pd.DataFrame,
    source_coverage: pd.DataFrame,
    family_coverage: pd.DataFrame,
    rank_intervals: pd.DataFrame,
    failure_modes: pd.DataFrame,
    underobserved: pd.DataFrame,
    business_domain_pressure: pd.DataFrame,
    domain_workflows: pd.DataFrame,
    release_cadence_family: pd.DataFrame,
    release_cadence_vendor: pd.DataFrame,
    domain_catalog: pd.DataFrame,
    domain_results: pd.DataFrame,
    domain_frontier: pd.DataFrame,
    domain_velocity: pd.DataFrame,
    domain_forecasts: pd.DataFrame,
    domain_thresholds: pd.DataFrame,
    message_cost_trends: pd.DataFrame,
    message_cost_profile_components: pd.DataFrame,
    fixed_task_cost_candidates: pd.DataFrame,
    fixed_task_cost_curves: pd.DataFrame,
    cost_divergence_scenarios: pd.DataFrame,
    cost_external_evidence: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> None:
    top_company = company_scores.iloc[0]
    top_open = company_scores.sort_values("openness_component", ascending=False).iloc[0]
    top_jobs = jobs[["title", "near_term_disruption_index", "substitution_pressure_index", "augmentation_index", "full_job_automation_feasibility_index", "dominant_outcome", "risk_label"]].head(12)
    top_replacement = replacement[["title", "job_family", "full_job_automation_feasibility_index", "substitution_pressure_index", "human_bottleneck_index", "dominant_outcome"]].head(12)
    base_probs = probabilities[(probabilities["horizon_years"].eq(2)) & (probabilities["scenario"].eq("frontier_quality"))][
        ["model_family", "simulation_win_share", "simulated_score_p10", "simulated_score_p90"]
    ].head(10)
    long_probs = probabilities[(probabilities["horizon_years"].eq(10)) & (probabilities["scenario"].eq("frontier_quality"))][
        ["model_family", "simulation_win_share", "simulated_score_p10", "simulated_score_p90"]
    ].head(10)
    open_long_probs = probabilities[(probabilities["horizon_years"].eq(10)) & (probabilities["scenario"].eq("open_ecosystem_upside"))][
        ["model_family", "simulation_win_share", "simulated_score_p10", "simulated_score_p90"]
    ].head(10)
    efficient_models = price_frontier[price_frontier["price_performance_frontier"]][
        ["canonical_model", "model_family", "access_class", "blended_price_usd_per_1m", "family_best_lmarena", "quality_proxy_level", "price_performance_index"]
    ].head(12)
    direct_models = direct_price[
        [
            "canonical_model",
            "model_family",
            "blended_price_usd_per_1m",
            "direct_evidence_sources",
            "direct_lmarena_rating",
            "direct_lmarena_match_confidence",
            "direct_price_performance_index",
            "quality_proxy_level",
        ]
    ].head(12)
    direct_match_summary = match_audit.groupby("match_confidence").size().rename("rows").reset_index().sort_values("match_confidence")
    vendor_table = vendor_scores[
        ["rank", "vendor", "vendor_frontier_portfolio_score", "flagship_family", "family_count", "portfolio_families", "evidence_count"]
    ].head(10)
    source_table = source_coverage[
        ["table", "rows", "latest_source_date", "release_date_coverage", "price_coverage", "benchmark_value_coverage", "organization_vendor_coverage"]
    ].head(12)
    family_coverage_table = family_coverage[
        ["model_family", "vendor", "coverage_score", "direct_benchmark_match_count", "family_proxy_benchmark_count", "source_gap_count"]
    ].head(12)
    rank_table = rank_intervals[
        ["model_family", "current_rank", "score_p10", "score_p50", "score_p90", "best_rank", "median_rank", "worst_rank", "rank_stability_label"]
    ].head(12)
    weakness_table = failure_modes[["claim_id", "assumption", "failure_mode", "mitigation", "severity"]]
    underobserved_table = underobserved[
        ["model_family", "vendor", "direct_benchmark_match_count", "coverage_score", "underobserved", "underobserved_reasons"]
    ].head(12)
    domain_table = business_domain_pressure[
        ["business_domain", "pressure_label", "disruption_index", "augmentation_index", "replacement_feasibility_index", "human_bottleneck_index", "example_occupations"]
    ]
    workflow_table = domain_workflows[["business_domain", "workflow_example", "likely_ai_role", "human_gate"]]
    cadence_family_table = release_cadence_family[
        ["model_family", "vendor", "total_releases", "recent_releases_365d", "median_days_between_releases", "days_since_latest_release", "cadence_label"]
    ].head(12)
    cadence_vendor_table = release_cadence_vendor[
        ["vendor", "portfolio_families", "total_releases", "recent_releases_365d", "median_days_between_releases", "cadence_label"]
    ].head(12)
    domain_catalog_table = domain_catalog[
        ["domain_label", "normalized_result_rows", "source_count", "benchmark_count", "model_count", "coverage_label", "latest_eval_date", "interpretation"]
    ].head(14)
    domain_velocity_table = domain_velocity[
        ["domain_label", "current_frontier_score", "annual_frontier_point_gain_used", "slope_source", "years_observed", "coverage_label", "forecast_confidence"]
    ].head(14)
    domain_forecast_table = domain_forecasts[(domain_forecasts["scenario"].eq("base")) & (domain_forecasts["horizon_years"].isin([2, 5, 10]))][
        ["domain_label", "target_year", "forecast_frontier_score", "current_frontier_score", "confidence", "caveat"]
    ].head(30)
    domain_threshold_table = domain_thresholds[domain_thresholds["threshold_score"].isin([90, 95])][
        ["domain_label", "threshold_score", "base_years_to_threshold", "estimated_threshold_year", "confidence"]
    ].head(24)
    domain_source_sample = domain_results[
        ["source_name", "domain_label", "benchmark", "task", "model_name", "score_normalized_0_100", "eval_date", "limitations"]
    ].sort_values(["domain_label", "score_normalized_0_100"], ascending=[True, False]).head(18)
    gap_table = gap[["category", "closed_or_api", "open_weight", "open_closed_best_gap", "open_closed_gap_pct_of_closed", "comparison_note"]].head(10)
    cluster_table = cluster_profiles[
        ["labor_cluster_id", "cluster_label", "occupation_count", "full_job_automation_feasibility_index", "augmentation_index", "human_bottleneck_index", "example_occupations"]
    ].head(10)
    labor_weighted = labor_summary[labor_summary["grouping"].eq("dominant_outcome")][
        ["group", "occupation_count", "labor_weight_sum", "weighted_disruption_index", "weighted_replacement_feasibility", "weighted_augmentation_index"]
    ]
    base_forecast = forecasts[(forecasts["scenario"].eq("base")) & (forecasts["metric"].isin(
        [
            "share_of_us_occupation_tasks_materially_touched",
            "frontier_output_price_factor",
            "open_weight_lmarena_gap_remaining",
            "frontier_context_window_multiplier",
        ]
    ))]
    message_cost_table = message_cost_trends[
        [
            "year",
            "released_model_count",
            "low_cost_blended_price_usd_per_1m",
            "median_blended_price_usd_per_1m",
            "frontier_blended_price_usd_per_1m",
            "average_effective_tokens",
            "modeled_average_message_cost_usd",
            "message_cost_index_2023_100",
        ]
    ]
    message_profile_table = message_cost_profile_components[message_cost_profile_components["year"].eq(2026)][
        ["display_name", "mix_share", "input_tokens", "output_tokens", "price_quantile", "profile_cost_usd", "weighted_cost_contribution_usd"]
    ]
    fixed_current_table = fixed_task_cost_curves[fixed_task_cost_curves["scenario"].eq("current")][
        ["display_name", "domain_label", "selected_model", "selected_family", "required_domain_score", "selected_domain_score", "forecast_task_cost_usd", "adequacy_status", "human_gate"]
    ]
    fixed_base_table = fixed_task_cost_curves[
        fixed_task_cost_curves["scenario"].eq("base")
        & fixed_task_cost_curves["horizon_years"].isin([2, 5, 10])
    ][
        ["display_name", "target_year", "selected_family", "selected_domain_score", "forecast_task_cost_usd", "cost_factor_vs_current", "adequacy_status"]
    ].head(24)
    divergence_table = cost_divergence_scenarios[
        [
            "scenario",
            "target_year",
            "modeled_average_message_cost_usd",
            "message_cost_factor_vs_2026",
            "median_fixed_task_cost_usd",
            "fixed_task_cost_factor_vs_2026",
            "frontier_workload_complexity_multiplier",
        ]
    ]
    cost_evidence_table = cost_external_evidence[["source_id", "name", "used_for", "url"]]

    body = f"""# Deep Frontier AI Analysis

Reference date: **{REFERENCE_DATE}**. Generated at: **{CAPTURED_AT}**.

This report is deliberately data-heavy. It uses the local rich frontier-model dataset, the public Anthropic Economic Index release files for occupation exposure, and a new domain benchmark layer covering coding, medicine, terminal agents, finance, legal reasoning, math, science/reasoning, language, vision and search/document work. The goal is not to claim precision about the future; it is to make the assumptions inspectable enough that the forecast can be argued with.

## Dashboard Snapshot

This opening map is the fast path through the analysis. It turns the long report into a set of inspectable questions, each tied to a primary artifact and an evidence-strength label.

{report_table(dashboard[['section', 'question', 'headline', 'metric', 'evidence_level', 'primary_artifact']])}

## How To Read This Report

The report is organized around three questions:

1. **Who has the strongest frontier-family signal right now?** The answer is a composite heuristic, so the report shows both rank and component composition instead of hiding the weighting.
2. **Where are the counterintuitive gaps?** Open-weight systems, low prices, context windows and benchmark ratings move on different axes. The plots keep those axes separate.
3. **Which domains are improving fastest?** The domain panel keeps fields separate: coding and agentic terminal work should not be averaged blindly with medicine, legal reasoning or finance.
4. **What happens when model capability meets labor structure?** Occupation exposure is not the same thing as replacement. The labor section separates task pressure, augmentation, bottlenecks and whole-job feasibility.
5. **Are costs rising or falling?** The economics section separates workload-mix cost per message/task from fixed-task cost curves, because those can move in opposite directions.

Every chart should be read as an audit surface. If a conclusion depends on one metric, the report names that metric and shows the caveat near the visualization. Domain rows with `forecast_enabled=false` are deliberately held flat: no comparable history means no extrapolation. The full remediation log is in `docs/statistical_audit.md`.

Evidence badges used throughout the HTML view: `observed`, `direct_match`, `family_proxy`, `scenario`, `speculative`. They are labels for evidence strength, not decoration.

## Executive Takeaways

1. **Near-term frontier-family leadership is concentrated, but not one-dimensional.** The highest heuristic index in this run is **{top_company['model_family']}** with a frontier momentum heuristic index of **{top_company['frontier_momentum_heuristic_index']:.1f}**. The strongest openness/cost/ecosystem signal is **{top_open['model_family']}**, which is not automatically the same thing as best closed frontier performance.
2. **The next-winner question is a simulation sensitivity exercise.** The table changes component weights thousands of times and injects evidence noise. Its shares are not calibrated probabilities.
3. **Open vs closed is category-specific.** Some LMArena categories show narrow gaps; others preserve a clear closed/API advantage. "Open source caught up" is too crude.
4. **Field-level progress is uneven.** Coding, terminal-agent and language/document signals have denser coverage than legal and finance. The report extrapolates only domains with repeated observations of the same benchmark; other domains are marked `insufficient_history` and held flat.
5. **The job story is not "all jobs disappear."** The highest-risk roles are task bundles where language, analysis, clerical transformation and directive delegation are already exposed. Jobs with physical work, trust, regulation or face-to-face accountability keep meaningful bottlenecks.
6. **The 10-year labor path is a scenario, not an estimate.** The task-contact paths encode explicit adoption assumptions; they are useful for stress testing verification, liability and workflow redesign, not for predicting employment levels.
7. **The cost view is synthetic.** Message/task paths combine assumed workload mixes with current catalog cohorts, while fixed-task paths use explicit quality and price scenarios. Neither is observed invoice history.

## Data Freshness And Coverage

The report now exposes source coverage before leaning on rankings. This section records row counts, captured dates, latest source dates and missingness across release dates, prices, benchmark values, context windows and organization/vendor fields.

{report_table(source_table)}

Family coverage matrix:

{report_table(family_coverage_table)}

![Source coverage dashboard](../figures/deep_analysis/source_coverage_dashboard.png)

![Family signal coverage heatmap](../figures/deep_analysis/family_signal_coverage_heatmap.png)

## Capability Domains

This is the new domain benchmark layer. It pulls together local benchmark sources and additional public sources downloaded during generation: LiveCodeBench, Open Medical-LLM Leaderboard result files, Terminal-Bench, FinanceBench, QFBench and Lexometrica LegalBench RU. Scores are normalized to a 0-100 frontier scale so fields can be compared without pretending that a medical QA percent, a legal composite, an arena rating and an agentic terminal score are the same measurement.

Domain catalog:

{report_table(domain_catalog_table)}

Representative high-scoring source rows:

{report_table(domain_source_sample)}

![Domain benchmark coverage](../figures/deep_analysis/domain_benchmark_coverage.png)

![Domain source matrix](../figures/deep_analysis/domain_source_matrix.png)

## Domain Improvement Velocity

The velocity table estimates how quickly each field is improving in the public benchmark panel. When a domain has enough dated observations, the report uses its observed frontier slope. When the time series is too thin, it falls back to the cross-domain median and labels the slope source explicitly.

{report_table(domain_velocity_table)}

![Domain frontier trends](../figures/deep_analysis/domain_frontier_trends.png)

![Domain current velocity](../figures/deep_analysis/domain_current_velocity.png)

## Domain Capability Forecasts

The domain forecast uses a bounded gap-closure model: a domain starts at its current normalized frontier score, closes a fraction of the remaining gap each year, and is capped below 100. This makes the forecast interpretable: the question is how quickly each field closes the remaining gap, not whether scores can grow without limit.

Base scenario by domain and horizon:

{report_table(domain_forecast_table)}

Threshold timing:

{report_table(domain_threshold_table)}

![Domain forecast base](../figures/deep_analysis/domain_forecast_base.png)

![Domain forecast scenarios](../figures/deep_analysis/domain_forecast_scenarios.png)

![Domain threshold timeline](../figures/deep_analysis/domain_threshold_timeline.png)

## Model Family Frontier Score

The index ranks model families and product lines, not legal companies. It blends benchmark performance, release velocity, API surface, price, research/ecosystem pull and openness. It is not a universal truth; sensitivity outputs show which rankings are weight-sensitive.

{markdown_table(company_scores, ['rank', 'model_family', 'frontier_momentum_heuristic_index', 'sensitivity_label', 'performance_component', 'release_velocity_component', 'ecosystem_component', 'cost_efficiency_component', 'openness_component'], 12)}

![Company frontier scores](../figures/deep_analysis/company_frontier_scores.png)

The headline rank is only the entry point. The stacked component chart below shows why a family ranks where it ranks. That matters because two families can have similar headline indexes for very different reasons: one may be performance-heavy, another may be ecosystem-heavy or cost-efficient.

![Score component stack](../figures/deep_analysis/company_score_component_stack.png)

The evidence-depth scatter is the reviewer sanity check. A family with high score and high evidence count is more defensible than a family with a high score from sparse rows. Bubble size is tied to API catalog breadth, while color shows openness.

![Score evidence scatter](../figures/deep_analysis/company_score_evidence_scatter.png)

## Family Ranking vs Vendor Portfolio Ranking

Reviewers often reason in terms of companies, but model families remain the cleaner technical unit. The vendor view is therefore a companion view: it blends the flagship family with the evidence-weighted portfolio mean and keeps the flagship family visible.

{report_table(vendor_table)}

![Vendor frontier scores](../figures/deep_analysis/vendor_frontier_scores.png)

![Family vs vendor rank shift](../figures/deep_analysis/family_vs_vendor_rank_shift.png)

## Who Builds The Next Best Model?

This table is not a prediction market. It is a Monte Carlo stress test over the scoring components: benchmark performance, release velocity, ecosystem pull, capability surface, cost and openness. Scenario weights are derived from the published baseline weights through documented multipliers (see `LEADERSHIP_SCENARIO_MULTIPLIERS` in the source), so no weight in the simulation is a hand-typed vector. `simulation_win_share` is the share of simulation draws won by each family, not a calibrated real-world probability. The corrected version separates **frontier-quality leadership** from **open-ecosystem upside**. The former asks who is most likely to make the raw best model; the latter asks who benefits if distribution and low cost matter more.

2-year simulated leaders:

{report_table(base_probs)}

10-year simulated leaders, frontier-quality scenario:

{report_table(long_probs)}

10-year simulated leaders, open-ecosystem-upside scenario:

{report_table(open_long_probs)}

![Next frontier probabilities](../figures/deep_analysis/company_next_frontier_probabilities.png)

The scenario matrix compresses the same simulation into a reviewer-friendly view: each cell names the leading family under a scenario/horizon pair and reports its share of simulation draws. This makes it obvious when the answer changes because the question changed.

![Leadership scenario matrix](../figures/deep_analysis/leadership_scenario_matrix.png)

## Open vs Closed: Where Is The Gap?

{report_table(gap_table)}

![Open closed gap by category](../figures/deep_analysis/open_closed_gap_by_category.png)

The gap chart shows differences, but differences alone can hide whether both sides are high-quality. The paired rating chart below shows the actual open and closed best observed ratings by category where both sides exist.

![Open closed category levels](../figures/deep_analysis/open_closed_category_levels.png)

## Price-Performance Frontier

Raw best model and economically deployable model are not the same decision. This frontier is explicitly a family-level proxy: OpenRouter model prices are joined to the best observed LMArena rating for the model family, not to a direct benchmark for every listed model. It should be read as a deployability screen, not model-level proof.

{report_table(efficient_models)}

![Price performance frontier](../figures/deep_analysis/price_performance_frontier.png)

The context-price map keeps three product dimensions visible at once: context window, blended token price and family-level rating proxy. It prevents a common mistake in AI market analysis: treating cheap, long-context and high-quality as one metric.

![Context price rating map](../figures/deep_analysis/price_context_rating_map.png)

## Direct Model Evidence vs Family Proxy

The audit table matches OpenRouter model IDs/names to LMArena, SWE-bench, LiveBench and Open LLM Leaderboard rows. Direct model matches are separated from `family_only` evidence so the deployability screen does not quietly inherit model-level certainty it does not have.

Match confidence audit:

{report_table(direct_match_summary)}

Direct evidence price-performance rows:

{report_table(direct_models)}

![Direct vs proxy price performance](../figures/deep_analysis/direct_vs_proxy_price_performance.png)

## LLM Cost Per Message vs Fixed Task Cost

This section separates two claims that are often blended together. A **modeled average message/task** can become more expensive when users route more work to long-context, tool-heavy or agentic frontier runs. A **fixed task**, such as thesis-quality long-form writing under a stable token budget and quality threshold, can become cheaper when cheaper families become good enough. The tables below do not claim to observe private invoices or usage logs; they expose the assumptions behind the workload mix and the fixed-task thresholds.

Modeled message/task cost by release cohort:

{report_table(message_cost_table)}

2026 workload profile components:

{report_table(message_profile_table)}

![LLM message cost trends](../figures/deep_analysis/llm_message_cost_trends.png)

Current cheapest adequate fixed-task candidates:

{report_table(fixed_current_table)}

Base scenario fixed-task curves:

{report_table(fixed_base_table)}

![Fixed task cost curves](../figures/deep_analysis/fixed_task_cost_curves.png)

The divergence table is the explicit version of the user's hypothesis: frontier work-unit cost can rise because average tasks get harder, while fixed task cost can fall because capability diffuses into cheaper models.

{report_table(divergence_table)}

![Cost task message divergence](../figures/deep_analysis/cost_task_message_divergence.png)

![Fixed task quality cost ladder](../figures/deep_analysis/fixed_task_quality_cost_ladder.png)

Cost evidence notes:

{report_table(cost_evidence_table)}

## Job Exposure And Labor Pressure

The labor table joins Anthropic observed occupation exposure to wage/job companion data, task-level penetration, automation/augmentation mode shares, and keyword-derived task bottlenecks from O*NET text. The output is an occupation-level pressure index, not a prediction that a whole occupation vanishes.

{report_table(top_jobs)}

![Job exposure top](../figures/deep_analysis/job_exposure_top.png)

![Wage scatter](../figures/deep_analysis/job_exposure_wage_scatter.png)

## Whole-Job Replacement Feasibility

The replacement feasibility index gates substitution pressure through physical, trust, regulatory and task-coverage bottlenecks. This is the section that answers the "will AI take all jobs" question more honestly: many occupations are touched; fewer are clean full-job replacement candidates.

{report_table(top_replacement)}

![Replacement feasibility](../figures/deep_analysis/job_replacement_feasibility.png)

## Labor Clusters

{report_table(cluster_table)}

Labor-weighted dominant outcome summary:

{report_table(labor_weighted)}

![Labor clusters](../figures/deep_analysis/labor_cluster_profiles.png)

The labor-weighted outcome mix below is the report's guardrail against overclaiming. It weights modeled outcomes by the best available public labor proxy so a handful of highly automatable occupations do not dominate the narrative.

![Labor outcome mix](../figures/deep_analysis/labor_outcome_mix.png)

## Business Domain Implications

The domain layer translates occupation-level pressure into business language. It does not replace occupation evidence; each domain keeps example occupations and explicit human gates so a portfolio reviewer can see where the abstraction could fail.

{report_table(domain_table)}

Workflow examples:

{report_table(workflow_table)}

![Business domain pressure matrix](../figures/deep_analysis/business_domain_pressure_matrix.png)

## 2, 5 And 10 Year Forecasts

Base scenario subset:

{report_table(base_forecast[['target_year', 'metric', 'value', 'unit', 'method']])}

The dashboard puts four scenario families on one page: context scale, output price, open-weight benchmark gap and task-share contact. The useful reading is not the exact number in 2036; it is which assumptions move together and which do not.

![Forecast scenario dashboard](../figures/deep_analysis/forecast_scenario_dashboard.png)

![Labor task forecast](../figures/deep_analysis/labor_task_forecast.png)

![Cost forecast](../figures/deep_analysis/cost_forecast_scenarios.png)

![Open closed catchup](../figures/deep_analysis/open_closed_catchup.png)

## Uncertainty And Rank Stability

The uncertainty view stress-tests component weights and evidence depth. These intervals are not calibrated confidence intervals; they are a visibility layer for how much the rank can move when public-source signals are perturbed.

{report_table(rank_table)}

![Frontier rank uncertainty](../figures/deep_analysis/frontier_rank_uncertainty.png)

The forecast band chart is a scenario envelope across conservative, base and aggressive cases. It should not be read as a statistical confidence band.

![Forecast uncertainty bands](../figures/deep_analysis/forecast_uncertainty_bands.png)

## Release Velocity And Product Cadence

Release cadence separates visible public execution speed from benchmark quality. The cadence tables combine OpenRouter API catalog entries and Epoch metadata, deduplicated by family, vendor, product line and date.

Family cadence:

{report_table(cadence_family_table)}

Vendor cadence:

{report_table(cadence_vendor_table)}

![Release cadence timeline](../figures/deep_analysis/release_cadence_timeline.png)

![Recent release velocity](../figures/deep_analysis/recent_release_velocity.png)

## Historical Analogy

AI looks less like a single prior wave and more like an uncomfortable hybrid: spreadsheet-style task rebundling, internet-style diffusion, cloud-style API economics, and electricity-style long-run production redesign. Every dimension score in this table is an author-assigned subjective prior (`input_basis=author_assigned_subjective_prior`), published for transparency rather than as evidence; the similarity score inherits that status.

{markdown_table(analogies, ['wave', 'period', 'ai_similarity_score', 'interpretation'], 8)}

![Historical analogy](../figures/deep_analysis/historical_analogy_index.png)

## Forecast Claims

{report_table(claims)}

## Counterintuitive Findings

{report_table(findings)}

## Where This Analysis Is Weak

The skeptical section names assumptions that could break the analysis. It is meant to make the work easier to challenge, not to protect it with broad caveats.

{report_table(weakness_table)}

Under-observed family audit:

{report_table(underobserved_table)}

## Method Notes

- Model-family scoring uses `data/dataset/`: LMArena full leaderboard rows, SWE-bench submissions, Open LLM Leaderboard metrics, OpenRouter prices/context, Epoch model metadata, Hugging Face rollups, GitHub model mentions and OpenAlex paper mentions.
- Domain scoring adds downloaded public benchmark sources under `data/raw/domain_benchmarks/`: LiveCodeBench, Open Medical-LLM, Terminal-Bench, FinanceBench, QFBench and Lexometrica. These are normalized into `domain_benchmark_results.csv`.
- Direct model evidence uses conservative name matching across exact, normalized exact, alias, family-only and unmatched classes. Family-only rows are audit evidence, not direct model proof.
- Vendor scoring maps model families to legal vendors and combines flagship-family signal with evidence-weighted portfolio breadth.
- Rank stability and forecast bands are stress tests and scenario envelopes. They are not calibrated confidence intervals.
- Labor scoring uses Anthropic Economic Index files from Hugging Face, including occupation exposure, task penetration, task automation/augmentation labels, O*NET task mappings/statements, and BLS wage/employment companion data.
- Scenario forecasts are not forecasts from a proprietary model. They are transparent transforms of observed slopes and pressure scores. Every scenario row includes a method field and the input diagnostics include caps/fallback policy.
- Domain forecasts use bounded gap closure from dated public benchmark frontier trends. When a domain lacks enough longitudinal evidence, the forecast uses a cross-domain fallback and marks confidence as low.
- Cost-per-message analysis is a workload-mix model over listed API price cohorts, not observed billing data. It separates low-cost chat, knowledge work, long-context analysis and agentic workflow runs.
- Fixed-task cost curves hold task token budgets and quality thresholds stable, then ask which current or future adequate family proxy is cheapest. They should be read as deployability screens, not direct model guarantees.
- Leadership simulation shares are stochastic sensitivity analyses over explicit score components, not calibrated market probabilities.
- Labor-weighted summaries use the best available public companion weights; where only major-group BLS employment is available, the analysis allocates it across detailed occupations inside that group to avoid treating each detailed occupation as the whole major group.
- BLS web xlsx endpoints returned anti-bot 403 responses in this environment. The analysis therefore uses public BLS-derived companion files already included in Anthropic's release rather than scraping around that restriction.

## Generated Artifacts

- `data/analysis/company_frontier_scores.csv`
- `data/analysis/dashboard_key_findings.csv`
- `data/analysis/domain_benchmark_catalog.csv`
- `data/analysis/domain_benchmark_results.csv`
- `data/analysis/domain_capability_frontier.csv`
- `data/analysis/domain_improvement_velocity.csv`
- `data/analysis/domain_capability_forecasts.csv`
- `data/analysis/domain_forecast_thresholds.csv`
- `data/analysis/company_score_methodology.csv`
- `data/analysis/company_score_sensitivity.csv`
- `data/analysis/model_benchmark_match_audit.csv`
- `data/analysis/direct_model_price_performance.csv`
- `data/analysis/llm_message_cost_trends.csv`
- `data/analysis/llm_message_cost_profile_components.csv`
- `data/analysis/fixed_task_cost_candidates.csv`
- `data/analysis/fixed_task_cost_curves.csv`
- `data/analysis/cost_divergence_scenarios.csv`
- `data/analysis/cost_external_evidence.csv`
- `data/analysis/vendor_frontier_scores.csv`
- `data/analysis/vendor_score_components.csv`
- `data/analysis/source_coverage_diagnostics.csv`
- `data/analysis/family_coverage_matrix.csv`
- `data/analysis/frontier_score_bootstrap.csv`
- `data/analysis/rank_stability_intervals.csv`
- `data/analysis/claim_failure_modes.csv`
- `data/analysis/underobserved_family_audit.csv`
- `data/analysis/business_domain_ai_pressure.csv`
- `data/analysis/domain_workflow_examples.csv`
- `data/analysis/release_cadence_by_family.csv`
- `data/analysis/release_cadence_by_vendor.csv`
- `data/analysis/job_exposure_scores.csv`
- `data/analysis/capability_forecasts.csv`
- `data/analysis/forecast_input_diagnostics.csv`
- `data/analysis/company_next_frontier_probabilities.csv`
- `data/analysis/open_closed_gap_by_category.csv`
- `data/analysis/lmarena_category_leaders.csv`
- `data/analysis/price_performance_frontier.csv`
- `data/analysis/labor_cluster_profiles.csv`
- `data/analysis/labor_market_exposure_summary.csv`
- `data/analysis/job_replacement_feasibility.csv`
- `data/analysis/counterintuitive_findings.csv`
- `data/analysis/historical_analogy_index.csv`
- `data/analysis/forecast_claims.csv`
- `figures/deep_analysis/company_score_component_stack.png`
- `figures/deep_analysis/company_score_evidence_scatter.png`
- `figures/deep_analysis/domain_benchmark_coverage.png`
- `figures/deep_analysis/domain_source_matrix.png`
- `figures/deep_analysis/domain_frontier_trends.png`
- `figures/deep_analysis/domain_current_velocity.png`
- `figures/deep_analysis/domain_forecast_base.png`
- `figures/deep_analysis/domain_forecast_scenarios.png`
- `figures/deep_analysis/domain_threshold_timeline.png`
- `figures/deep_analysis/leadership_scenario_matrix.png`
- `figures/deep_analysis/open_closed_category_levels.png`
- `figures/deep_analysis/price_context_rating_map.png`
- `figures/deep_analysis/labor_outcome_mix.png`
- `figures/deep_analysis/forecast_scenario_dashboard.png`
- `figures/deep_analysis/direct_vs_proxy_price_performance.png`
- `figures/deep_analysis/llm_message_cost_trends.png`
- `figures/deep_analysis/fixed_task_cost_curves.png`
- `figures/deep_analysis/cost_task_message_divergence.png`
- `figures/deep_analysis/fixed_task_quality_cost_ladder.png`
- `figures/deep_analysis/vendor_frontier_scores.png`
- `figures/deep_analysis/family_vs_vendor_rank_shift.png`
- `figures/deep_analysis/source_coverage_dashboard.png`
- `figures/deep_analysis/family_signal_coverage_heatmap.png`
- `figures/deep_analysis/frontier_rank_uncertainty.png`
- `figures/deep_analysis/forecast_uncertainty_bands.png`
- `figures/deep_analysis/business_domain_pressure_matrix.png`
- `figures/deep_analysis/release_cadence_timeline.png`
- `figures/deep_analysis/recent_release_velocity.png`
- `figures/deep_analysis/*.png`
"""
    md_path = REPORT / "deep_frontier_ai_forecast.md"
    md_path.write_text(body, encoding="utf-8")
    write_html_report(body, REPORT / "deep_frontier_ai_forecast.html", dashboard)


def write_html_report(markdown: str, path: Path, dashboard: pd.DataFrame | None = None) -> None:
    write_report_assets()
    lines = markdown.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), "Deep Frontier AI Analysis")
    toc: list[tuple[str, str]] = []
    used_ids: set[str] = set()
    for line in lines:
        if line.startswith("## "):
            label = line[3:].strip()
            section_id = unique_html_id(label, used_ids)
            toc.append((section_id, label))

    out = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<link rel='stylesheet' href='assets/report.css'>",
        "<style>",
        html_report_css(),
        "</style>",
        "</head>",
        "<body>",
        "<div class='report-shell'>",
        "<aside class='report-sidebar' aria-label='Report navigation'>",
        "<a class='sidebar-title' href='#top'>AI Capability Signals</a>",
        "<div class='sidebar-subtitle'>Interactive report</div>",
        "<nav><ol>",
        *[f"<li><a href='#{section_id}'>{html.escape(label)}</a></li>" for section_id, label in toc],
        "</ol></nav>",
        "</aside>",
        "<main id='top' class='report-main'>",
        "<div class='sticky-summary' aria-label='Sticky key-number summary'>"
        "<strong>Dashboard-first view</strong>"
        "<span>Start with the summary cards, then drill into sortable evidence tables and figures.</span>"
        "</div>",
        "<header class='hero'>",
        "<div class='hero-layout'>",
        "<div>",
        "<div class='eyebrow'>Hiring portfolio analysis</div>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='hero-copy'>A dashboard-first, public-source view of frontier model signals, domain benchmark velocity, open/closed gaps, deployability economics and labor exposure. The interface puts the key comparisons up front, then keeps the full audit trail below.</p>",
        "<div class='evidence-badges'>"
        + "".join(f"<span class='evidence-badge evidence-{html.escape(key)}'>{html.escape(key)}</span>" for key in EVIDENCE_BADGES)
        + "</div>",
        "</div>",
        "<div class='hero-aside' aria-label='Report orientation'>",
        "<span>Read this as</span>",
        "<strong>leaderboards + caveats + source audit</strong>",
        "<details class='methodology-block' open><summary>Methodology details</summary><p>Composite indexes are heuristic, direct model evidence is separated from family proxies, and forecast bands are scenario envelopes rather than calibrated confidence intervals.</p></details>",
        "</div>",
        "</div>",
        "<div class='meta-grid'>",
        f"<div><span>Reference date</span><strong>{html.escape(REFERENCE_DATE)}</strong></div>",
        f"<div><span>Generated</span><strong>{html.escape(CAPTURED_AT)}</strong></div>",
        "<div><span>Method</span><strong>Heuristic + sensitivity</strong></div>",
        "</div>",
        "</header>",
    ]
    in_table = False
    in_code = False
    list_type: str | None = None
    table_lines: list[str] = []
    section_open = False
    preamble_open = False
    section_ids = iter([section_id for section_id, _ in toc])
    skipping_dashboard_markdown = False

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    def open_list(tag: str) -> None:
        nonlocal list_type
        if list_type != tag:
            close_list()
            out.append(f"<{tag}>")
            list_type = tag

    for line in lines:
        if skipping_dashboard_markdown and not line.startswith("## "):
            continue
        if skipping_dashboard_markdown and line.startswith("## "):
            skipping_dashboard_markdown = False
        if line.startswith("```"):
            if in_table:
                out.append(pipe_table_to_html(table_lines))
                table_lines = []
                in_table = False
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                close_list()
                out.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("# "):
            continue
        if line.startswith("Reference date:"):
            continue
        if line.startswith("|") and line.endswith("|"):
            close_list()
            table_lines.append(line)
            in_table = True
            continue
        if in_table:
            out.append(pipe_table_to_html(table_lines))
            table_lines = []
            in_table = False
        if not line.strip():
            close_list()
            continue
        if line.startswith("## "):
            close_list()
            if preamble_open:
                out.append("</section>")
                preamble_open = False
            if section_open:
                out.append("</section>")
            section_id = next(section_ids)
            label = line[3:].strip()
            if label == "Dashboard Snapshot" and dashboard is not None:
                out.append(render_dashboard_html(dashboard, section_id))
                section_open = False
                skipping_dashboard_markdown = True
                continue
            section_open = True
            out.append(f"<section id='{section_id}' class='report-section'>")
            out.append("<div class='section-rule'></div>")
            out.append(f"<h2>{html.escape(label)}</h2>")
        elif line.startswith("!["):
            close_list()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                alt = match.group(1)
                src = match.group(2)
                out.append(
                    "<figure class='figure-panel'>"
                    f"<a data-lightbox='figure' href='{html.escape(src)}'><img loading='lazy' alt='{html.escape(alt)}' src='{html.escape(src)}'></a>"
                    f"<figcaption>{html.escape(alt)}</figcaption>"
                    "</figure>"
                )
        elif line.startswith("- "):
            open_list("ul")
            out.append(f"<li>{inline_markdown(line[2:])}</li>")
        elif re.match(r"\d+\. ", line):
            open_list("ol")
            item = re.sub(r"^\d+\.\s+", "", line)
            out.append(f"<li>{inline_markdown(item)}</li>")
        elif line.strip():
            if not section_open and not preamble_open:
                out.append("<section class='report-preamble'>")
                preamble_open = True
            close_list()
            out.append(f"<p>{inline_markdown(line)}</p>")
    if table_lines:
        out.append(pipe_table_to_html(table_lines))
    close_list()
    if in_code:
        out.append("</code></pre>")
    if section_open or preamble_open:
        out.append("</section>")
    out.extend(
        [
            "<footer class='report-footer'>",
            "<a href='#top'>Back to top</a>",
            "<span>Generated from reproducible local analysis artifacts.</span>",
            "</footer>",
            "</main>",
            "</div>",
            "<div id='figure-lightbox' class='lightbox' hidden><button type='button' aria-label='Close figure'>Close</button><img alt='Expanded report figure'></div>",
            "<script src='assets/report.js'></script>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text("\n".join(out), encoding="utf-8")


def render_dashboard_html(dashboard: pd.DataFrame, section_id: str) -> str:
    rows = dashboard.sort_values("priority_order").to_dict("records")
    top_tiles = rows[:6]
    lanes = [
        ("Models", "Leaderboards, vendors and direct evidence", "model-family-frontier-score"),
        ("Domains", "Capability fields, velocity and forecasts", "capability-domains"),
        ("Economics", "Cost, context and deployable price-performance", "price-performance-frontier"),
        ("Labor", "Occupation pressure, domains and replacement gates", "job-exposure-and-labor-pressure"),
        ("Risk", "Coverage, stability and failure modes", "where-this-analysis-is-weak"),
    ]
    tile_html = []
    for row in top_tiles:
        tile_html.append(
            "<article class='dashboard-tile'>"
            f"<span class='tile-kicker'>{html.escape(str(row['section']))}</span>"
            f"<h3>{html.escape(str(row['headline']))}</h3>"
            f"<strong>{html.escape(str(row['metric']))}</strong>"
            f"<p>{inline_markdown(row['question'])}</p>"
            f"<span class='evidence-badge evidence-{html.escape(str(row['evidence_level']))}'>{html.escape(str(row['evidence_level']))}</span>"
            "</article>"
        )
    lane_html = []
    for label, text, href in lanes:
        lane_html.append(
            "<a class='dashboard-lane' href='#{href}'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(text)}</strong>"
            "</a>".format(href=html.escape(href))
        )
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['section']))}</td>"
            f"<td>{html.escape(str(row['headline']))}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{inline_markdown(row['reading'])}</td>"
            f"<td><code>{html.escape(str(row['primary_artifact']))}</code></td>"
            "</tr>"
        )
    return (
        f"<section id='{html.escape(section_id)}' class='analysis-dashboard'>"
        "<div class='section-rule'></div>"
        "<div class='dashboard-heading'>"
        "<div>"
        "<span class='eyebrow'>Highlights</span>"
        "<h2>Dashboard Snapshot</h2>"
        "<p>One-screen entry points into the full analysis. Each card is backed by a generated CSV or figure, so the overview stays auditable.</p>"
        "</div>"
        "<a class='dashboard-download' href='../data/analysis/dashboard_key_findings.csv' download>Download dashboard data</a>"
        "</div>"
        f"<div class='dashboard-tiles'>{''.join(tile_html)}</div>"
        f"<div class='dashboard-lanes'>{''.join(lane_html)}</div>"
        "<div class='dashboard-table'>"
        "<table data-sortable='true'>"
        "<tr><th>section</th><th>headline</th><th>metric</th><th>reading</th><th>artifact</th></tr>"
        + "".join(table_rows)
        + "</table>"
        "</div>"
        "</section>"
    )


def pipe_table_to_html(lines: list[str]) -> str:
    rows = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if i == 1 and all(set(c) <= {"-", ":"} for c in cells):
            continue
        tag = "th" if i == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{inline_markdown(c)}</{tag}>" for c in cells) + "</tr>")
    return (
        "<div class='table-wrap'>"
        "<div class='table-actions'><label class='table-filter'><span>Filter</span><input type='search' data-table-filter placeholder='Filter rows'></label><a href='../data/analysis/analysis_manifest.csv' download>Download table index</a></div>"
        "<table data-sortable='true'>"
        + "".join(rows)
        + "</table></div>"
    )


def unique_html_id(label: str, used: set[str]) -> str:
    base = slug(label)
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def write_report_assets() -> None:
    assets = REPORT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    css = """
.table-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  background: #f7f7f4;
  font-size: 12px;
}
.table-filter {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-weight: 700;
}
.table-filter input {
  width: min(220px, 42vw);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 10px;
  background: #fff;
  color: var(--ink);
}
table[data-sortable='true'] th { cursor: pointer; user-select: none; }
table[data-sortable='true'] th::after { content: ' sort'; color: var(--muted); font-weight: 500; font-size: 10px; }
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 99;
  display: grid;
  place-items: center;
  padding: 32px;
  background: rgba(12, 18, 24, 0.88);
}
.lightbox[hidden] { display: none; }
.lightbox img { max-width: 94vw; max-height: 88vh; background: white; }
.lightbox button {
  position: fixed;
  top: 18px;
  right: 18px;
  border: 1px solid rgba(255,255,255,0.4);
  background: rgba(255,255,255,0.12);
  color: white;
  border-radius: 6px;
  padding: 8px 11px;
}
"""
    js = """
document.querySelectorAll("table[data-sortable='true']").forEach((table) => {
  const headers = Array.from(table.querySelectorAll("th"));
  const tableWrap = table.closest(".table-wrap");
  const filter = tableWrap?.querySelector("[data-table-filter]");
  const bodyRows = () => Array.from(table.querySelectorAll("tr")).slice(1);

  if (filter) {
    filter.addEventListener("input", () => {
      const query = filter.value.trim().toLowerCase();
      bodyRows().forEach((row) => {
        row.hidden = query.length > 0 && !row.textContent.toLowerCase().includes(query);
      });
    });
  }

  headers.forEach((header, index) => {
    header.addEventListener("click", () => {
      const rows = bodyRows();
      const direction = header.dataset.sortDir === "asc" ? "desc" : "asc";
      header.dataset.sortDir = direction;
      rows.sort((a, b) => {
        const av = a.children[index]?.textContent?.trim() || "";
        const bv = b.children[index]?.textContent?.trim() || "";
        const an = Number(av.replace(/[%,$]/g, ""));
        const bn = Number(bv.replace(/[%,$]/g, ""));
        const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
        return direction === "asc" ? cmp : -cmp;
      });
      rows.forEach((row) => table.tBodies[0].appendChild(row));
    });
  });
});

const lightbox = document.getElementById("figure-lightbox");
if (lightbox) {
  const img = lightbox.querySelector("img");
  document.querySelectorAll("a[data-lightbox='figure']").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      img.src = link.href;
      img.alt = link.querySelector("img")?.alt || "Expanded report figure";
      lightbox.hidden = false;
    });
  });
  lightbox.querySelector("button").addEventListener("click", () => {
    lightbox.hidden = true;
    img.removeAttribute("src");
  });
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      lightbox.hidden = true;
      img.removeAttribute("src");
    }
  });
}
"""
    (assets / "report.css").write_text(css, encoding="utf-8")
    (assets / "report.js").write_text(js, encoding="utf-8")


def html_report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f7f4;
  --paper: #ffffff;
  --surface: #eeeeea;
  --surface-strong: #e2e1dc;
  --ink: #111111;
  --muted: #6c6d6a;
  --line: #d8d8d2;
  --line-strong: #bdbdb5;
  --blue: #1e67b1;
  --teal: #17806d;
  --gold: #c58b22;
  --orange: #e86f2a;
  --purple: #7447d8;
  --red: #b2473f;
  --green: #257a4b;
  --code-bg: #ecece7;
  --shadow: 0 16px 36px rgba(17, 17, 17, 0.08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}
a { color: inherit; text-decoration-thickness: 1px; text-underline-offset: 3px; }
.report-shell {
  min-height: 100vh;
}
.report-sidebar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 18px;
  min-height: 76px;
  padding: 20px;
  border-bottom: 1px solid rgba(17, 17, 17, 0.08);
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(14px);
  overflow: hidden;
}
.sidebar-title {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  white-space: nowrap;
  padding: 0 16px;
  border-radius: 999px;
  background: #050505;
  color: #ffffff;
  font-weight: 800;
  font-size: 14px;
  letter-spacing: 0;
  text-decoration: none;
}
.sidebar-subtitle {
  display: none;
}
.report-sidebar nav {
  flex: 1;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  scrollbar-width: none;
}
.report-sidebar nav::-webkit-scrollbar { display: none; }
.report-sidebar ol {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: 8px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  overflow-x: auto;
  scrollbar-width: none;
}
.report-sidebar ol::-webkit-scrollbar { display: none; }
.report-sidebar li { flex: 0 0 auto; }
.report-sidebar a:not(.sidebar-title) {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 13px;
  border-radius: 999px;
  background: var(--surface);
  color: #222222;
  text-decoration: none;
  font-size: 13px;
  line-height: 1;
  white-space: nowrap;
}
.report-sidebar a:not(.sidebar-title):hover {
  background: #deded8;
}
.report-main {
  width: min(1240px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 38px 0 72px;
}
.sticky-summary {
  width: 100%;
  margin: 0 0 18px;
  padding: 10px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  font-size: 13px;
  color: var(--muted);
}
.sticky-summary strong {
  margin-right: 8px;
  color: var(--ink);
}
.hero {
  padding: 48px 0 34px;
  border-bottom: 1px solid var(--line-strong);
}
.hero-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 34px;
  align-items: end;
}
.eyebrow {
  color: var(--purple);
  font-weight: 800;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0;
}
h1 {
  margin: 12px 0 18px;
  max-width: 840px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 76px;
  font-weight: 500;
  line-height: 0.96;
  letter-spacing: 0;
}
.hero-copy {
  max-width: 760px;
  font-size: 21px;
  color: #2d2d2a;
}
.hero-aside {
  border-top: 1px solid var(--line-strong);
  padding-top: 14px;
}
.hero-aside > span,
.meta-grid span,
.tile-kicker,
.dashboard-lane span {
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0;
  font-weight: 800;
}
.hero-aside > strong {
  display: block;
  margin: 7px 0 10px;
  font-size: 22px;
  line-height: 1.12;
}
.evidence-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 22px 0 0;
}
.evidence-badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 3px 8px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #ffffff;
  color: #333333;
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}
.evidence-observed { border-color: #a5c8b1; color: var(--green); }
.evidence-direct_match { border-color: #94b9df; color: var(--blue); }
.evidence-family_proxy { border-color: #d5bd82; color: var(--gold); }
.evidence-scenario { border-color: #b9a2ea; color: var(--purple); }
.evidence-speculative { border-color: #dc9d98; color: var(--red); }
.methodology-block {
  margin: 12px 0 0;
  padding: 0;
  border: 0;
  background: transparent;
}
.methodology-block summary {
  cursor: pointer;
  font-weight: 800;
}
.methodology-block p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 13px;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 32px;
}
.meta-grid div {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px 16px;
  background: var(--paper);
}
.meta-grid strong {
  display: block;
  margin-top: 5px;
  font-size: 15px;
}
.analysis-dashboard,
.report-preamble,
.report-section {
  padding: 44px 0 26px;
  border-bottom: 1px solid var(--line);
}
.section-rule {
  width: 54px;
  height: 6px;
  background: var(--ink);
  margin-bottom: 22px;
}
h2 {
  margin: 0 0 14px;
  font-size: 36px;
  line-height: 1.08;
  letter-spacing: 0;
}
h3 {
  margin: 0;
  font-size: 23px;
  line-height: 1.12;
  letter-spacing: 0;
}
p {
  max-width: 880px;
  margin: 14px 0;
  color: #2e2f2c;
  font-size: 16px;
}
ol, ul {
  max-width: 880px;
  padding-left: 24px;
  color: #2e2f2c;
}
li { margin: 7px 0; }
strong { color: var(--ink); }
code {
  background: var(--code-bg);
  padding: 2px 5px;
  border-radius: 5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
  overflow-wrap: anywhere;
  word-break: break-word;
}
pre {
  max-width: 100%;
  overflow-x: auto;
  padding: 18px;
  border-radius: 8px;
  background: #14202b;
  color: #f8fafc;
}
.dashboard-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 24px;
}
.dashboard-heading p {
  margin-bottom: 0;
}
.dashboard-download {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  color: var(--ink);
  background: #fff;
  text-decoration: none;
  font-size: 13px;
  font-weight: 800;
  white-space: nowrap;
}
.dashboard-tiles {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 24px 0;
}
.dashboard-tile {
  min-height: 220px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  background: var(--paper);
}
.dashboard-tile strong {
  font-size: 29px;
  line-height: 1.05;
}
.dashboard-tile p {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
}
.dashboard-tile .evidence-badge {
  margin-top: auto;
  align-self: flex-start;
}
.dashboard-lanes {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 20px 0 24px;
}
.dashboard-lane {
  display: grid;
  gap: 7px;
  min-height: 94px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  background: var(--surface);
  text-decoration: none;
}
.dashboard-lane strong {
  line-height: 1.2;
}
.dashboard-table {
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
}
.dashboard-table table {
  min-width: 980px;
}
.figure-panel {
  margin: 28px 0 36px;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.figure-panel a {
  display: block;
  background: #fff;
}
.figure-panel img {
  display: block;
  width: 100%;
  height: auto;
}
.figure-panel figcaption {
  padding: 12px 16px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
}
.table-wrap {
  width: 100%;
  margin: 24px 0 32px;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  box-shadow: 0 12px 28px rgba(17, 17, 17, 0.04);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  min-width: 760px;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface);
  color: #222222;
  font-weight: 800;
}
tr:nth-child(even) td { background: #fbfbf8; }
td {
  color: #2d2d2a;
  overflow-wrap: anywhere;
}
.report-footer {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-top: 52px;
  padding-top: 22px;
  border-top: 1px solid var(--line-strong);
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 980px) {
  .report-sidebar {
    align-items: flex-start;
    flex-direction: column;
    gap: 12px;
  }
  .report-main { width: min(100% - 28px, 1240px); }
  .hero-layout { grid-template-columns: 1fr; }
  h1 { font-size: 54px; }
  .dashboard-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dashboard-lanes { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .meta-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .report-main { width: min(100% - 24px, 1240px); padding-top: 24px; }
  h1 { font-size: 42px; }
  h2 { font-size: 29px; }
  .hero-copy { font-size: 18px; }
  .dashboard-heading { display: block; }
  .dashboard-download { margin-top: 14px; }
  .dashboard-tiles,
  .dashboard-lanes { grid-template-columns: 1fr; }
  .table-actions { align-items: stretch; flex-direction: column; }
  .table-filter input { width: 100%; }
}
@media print {
  body { background: white; }
  .report-sidebar { display: none; }
  .report-main { width: 100%; padding: 0; }
  .figure-panel, .table-wrap { box-shadow: none; break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""


def inline_markdown(text: Any) -> str:
    rendered = html.escape(str(text))
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', rendered)
    return rendered


def write_data_dictionary() -> None:
    text = """# Deep Analysis Data Dictionary

Generated by `python -m frontier_ai.deep_analysis`.

## Tables

- `company_frontier_scores`: model-family composite scores using benchmark, API, release, ecosystem, price and openness signals.
- `dashboard_key_findings`: one-screen report entry points with headline metric, evidence label and primary artifact.
- `domain_benchmark_catalog`: domain-level benchmark coverage, source count, model count, latest source date, interpretation and caveat.
- `domain_benchmark_results`: normalized model-benchmark rows across local and downloaded domain sources.
- `domain_capability_frontier`: annual domain frontier history used for improvement-rate estimates.
- `domain_improvement_velocity`: current frontier score, observed or fallback annual point gain, coverage label and forecast confidence by field.
- `domain_capability_forecasts`: 2, 5 and 10-year bounded gap-closure forecasts for each capability field.
- `domain_forecast_thresholds`: estimated base-scenario years to normalized 80/90/95 thresholds by field.
- `company_score_components`: reduced component table for plotting and review.
- `company_score_methodology`: explicit component weights, transforms and rationale.
- `company_score_sensitivity`: rank sensitivity under alternative component weights.
- `model_benchmark_match_audit`: conservative OpenRouter-to-benchmark model matching audit with exact, normalized, alias, family-only and unmatched confidence labels.
- `direct_model_price_performance`: deployability table restricted to rows with direct model-level benchmark evidence.
- `llm_message_cost_trends`: modeled workload-weighted cost per message/task by model release cohort year, with low/median/frontier price cohort statistics.
- `llm_message_cost_profile_components`: profile-level assumptions and cost contributions for simple chat, knowledge work, long-context and agentic workflow runs.
- `fixed_task_cost_candidates`: current task-model candidates for stable fixed task profiles, using OpenRouter prices joined to family-domain benchmark scores.
- `fixed_task_cost_curves`: current and future cheapest adequate model-family proxy for each fixed task profile under conservative/base/aggressive scenarios.
- `cost_divergence_scenarios`: scenario table comparing average message/task cost factors against fixed-task cost factors.
- `cost_external_evidence`: source and caveat registry for external cost, price-performance and agentic-token evidence.
- `vendor_frontier_scores`: vendor portfolio score that combines flagship family and evidence-weighted portfolio components.
- `vendor_score_components`: component-level vendor aggregation audit.
- `source_coverage_diagnostics`: row counts, captured dates, latest source dates and core-field missingness by source table.
- `family_coverage_matrix`: family/vendor coverage matrix for prices, release dates, context, benchmarks and labor signals.
- `frontier_score_bootstrap`: evidence-scaled bootstrap draws for heuristic family scores.
- `rank_stability_intervals`: score intervals, best/median/worst rank and stability labels from bootstrap draws.
- `claim_failure_modes`: skeptical audit of assumptions, failure modes and mitigations.
- `underobserved_family_audit`: families/vendors where evidence coverage is sparse or mostly indirect.
- `business_domain_ai_pressure`: occupation-derived AI pressure mapped to business domains.
- `domain_workflow_examples`: workflow examples, likely AI role and human gate by business domain.
- `release_cadence_by_family`: visible release cadence summary by model family.
- `release_cadence_by_vendor`: visible release cadence summary by vendor portfolio.
- `company_next_frontier_probabilities`: Monte Carlo simulation-win share for each family at 2, 5 and 10-year horizons.
- `leadership_model_audit`: component-level audit explaining why frontier-quality and open-ecosystem scenarios differ.
- `open_closed_gap_by_category`: LMArena category-level open-vs-closed best-model gaps.
- `lmarena_category_leaders`: best observed model/family by LMArena category and access bucket.
- `price_performance_frontier`: OpenRouter price-performance efficient frontier using an explicit family-level rating proxy.
- `job_exposure_scores`: occupation-level AI exposure, substitution pressure, augmentation index, human bottleneck and 2/5/10-year task-share scenarios.
- `job_replacement_feasibility`: whole-job automation feasibility after physical, trust, regulatory and task-coverage gates.
- `labor_cluster_profiles`: unsupervised occupation clusters based on exposure, bottleneck and task-domain features.
- `labor_market_exposure_summary`: labor-weighted summaries by job family, dominant outcome and risk label.
- `task_domain_exposure_heatmap`: SOC-major-group domain and bottleneck means for heatmap plotting.
- `capability_forecasts`: scenario rows for 2, 5 and 10 year horizons.
- `capability_frontier_history`: fitted history series used by the forecast generator.
- `forecast_input_diagnostics`: model-fit slopes, fitted windows and cap/fallback policies for forecast inputs.
- `historical_analogy_index`: structured comparison of AI to prior technology waves.
- `forecast_claims`: human-readable claims with confidence labels and evidence pointers.
- `counterintuitive_findings`: short evidence-backed surprising findings surfaced from the analysis tables.
- `deep_analysis_source_registry`: source registry for the deep analysis layer.

The index and simulation-share columns are normalized analytical constructs. They are intended for comparison, not as exact probabilities.
"""
    (DOCS / "deep_analysis_data_dictionary.md").write_text(text, encoding="utf-8")


def build_deep_analysis(overwrite_sources: bool = False, write_reports_flag: bool = True) -> None:
    ensure_dirs()
    aei = load_aei(overwrite=overwrite_sources)
    company_scores, components = build_company_frontier_scores()
    job_scores, domain = build_job_exposure_scores(aei)
    forecasts, history, claims = build_capability_forecasts(company_scores, job_scores)
    domain_catalog, domain_results, domain_frontier, domain_velocity, domain_forecasts, domain_thresholds = build_domain_benchmark_analysis(overwrite_sources=overwrite_sources)
    analogies = build_historical_analogy_index()
    gap, category_leaders = build_open_closed_gap_by_category()
    price_frontier = build_price_performance_frontier()
    match_audit = build_model_benchmark_match_audit()
    direct_price = build_direct_model_price_performance(match_audit, price_frontier)
    message_cost_trends, message_cost_profile_components, fixed_task_cost_candidates, fixed_task_cost_curves, cost_divergence_scenarios = build_llm_cost_task_analysis(
        forecasts,
        domain_results,
        domain_velocity,
        domain_forecasts,
    )
    cost_external_evidence = build_cost_external_evidence()
    vendor_scores, vendor_components = build_vendor_frontier_scores(company_scores)
    source_coverage, family_coverage = build_coverage_diagnostics(company_scores, match_audit)
    bootstrap, rank_intervals = build_rank_stability(company_scores)
    probabilities = build_company_leadership_simulation(company_scores)
    leadership_audit = build_leadership_model_audit(company_scores, probabilities)
    cluster_profiles, labor_summary, replacement = build_labor_deep_dive(job_scores)
    business_domain_pressure, domain_workflows = build_business_domain_implications(job_scores)
    release_cadence_family, release_cadence_vendor = build_release_cadence()
    failure_modes, underobserved = build_failure_mode_audits(company_scores, match_audit, family_coverage)
    findings = build_counterintuitive_findings(company_scores, probabilities, gap, price_frontier, labor_summary, replacement)
    sources = source_registry()
    dashboard = build_dashboard_key_findings(
        company_scores,
        job_scores,
        probabilities,
        gap,
        direct_price,
        source_coverage,
        family_coverage,
        rank_intervals,
        failure_modes,
        business_domain_pressure,
        release_cadence_family,
        domain_catalog,
        domain_velocity,
        domain_forecasts,
        message_cost_trends,
        fixed_task_cost_curves,
        cost_divergence_scenarios,
    )

    manifest_rows = [
        {"table": path.stem, "rows": len(pd.read_csv(path)), "path": str(path.relative_to(ROOT))}
        for path in sorted(ANALYSIS.glob("*.csv"))
        if path.stem != "analysis_manifest"
    ]
    manifest_rows.append({"table": "analysis_manifest", "rows": len(manifest_rows) + 1, "path": str((ANALYSIS / "analysis_manifest.csv").relative_to(ROOT))})
    manifest = pd.DataFrame(manifest_rows).sort_values("table")
    write_table(manifest, "analysis_manifest")
    write_run_manifest("analysis", ANALYSIS, list(ANALYSIS.glob("*.csv")), upstream_manifest=DATASET / "run_manifest.json")
    build_plots(
        company_scores,
        job_scores,
        forecasts,
        analogies,
        domain,
        gap,
        price_frontier,
        probabilities,
        cluster_profiles,
        labor_summary,
        replacement,
        match_audit,
        direct_price,
        vendor_scores,
        source_coverage,
        family_coverage,
        rank_intervals,
        business_domain_pressure,
        release_cadence_family,
        release_cadence_vendor,
        domain_catalog,
        domain_frontier,
        domain_velocity,
        domain_forecasts,
        domain_thresholds,
        message_cost_trends,
        fixed_task_cost_curves,
        fixed_task_cost_candidates,
        cost_divergence_scenarios,
    )
    if write_reports_flag:
        write_report(
            company_scores,
            job_scores,
            forecasts,
            analogies,
            claims,
            probabilities,
            gap,
            price_frontier,
            cluster_profiles,
            labor_summary,
            replacement,
            findings,
            match_audit,
            direct_price,
            vendor_scores,
            source_coverage,
            family_coverage,
            rank_intervals,
            failure_modes,
            underobserved,
            business_domain_pressure,
            domain_workflows,
            release_cadence_family,
            release_cadence_vendor,
            domain_catalog,
            domain_results,
            domain_frontier,
            domain_velocity,
            domain_forecasts,
            domain_thresholds,
            message_cost_trends,
            message_cost_profile_components,
            fixed_task_cost_candidates,
            fixed_task_cost_curves,
            cost_divergence_scenarios,
            cost_external_evidence,
            dashboard,
        )
        write_run_manifest(
            "deep_report",
            REPORT,
            [
                REPORT / "deep_frontier_ai_forecast.md",
                REPORT / "deep_frontier_ai_forecast.html",
                REPORT / "assets" / "report.css",
                REPORT / "assets" / "report.js",
            ],
            upstream_manifest=ANALYSIS / "run_manifest.json",
        )
    write_data_dictionary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()
    build_deep_analysis(overwrite_sources=args.overwrite_sources, write_reports_flag=not args.skip_reports)


if __name__ == "__main__":
    main()
