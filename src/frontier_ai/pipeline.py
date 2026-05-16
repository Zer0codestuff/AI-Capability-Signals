from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from dateutil import parser as dateparser

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE_PROCESSED = ROOT / "data" / "sample" / "processed"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
SAMPLE_REPORT = ROOT / "data" / "sample" / "report"
APPENDIX = ROOT / "appendix"
SAMPLE_APPENDIX = ROOT / "data" / "sample" / "appendix"
DOCS = ROOT / "docs"

CAPTURED_AT = datetime.now(UTC).replace(microsecond=0).isoformat()
REFERENCE_DATE = "2026-05-15"

EPOCH_URL = "https://epoch.ai/data/all_ai_models.csv"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
LMARENA_TREE_URL = "https://huggingface.co/api/datasets/lmarena-ai/leaderboard-dataset/tree/main?recursive=true"
LMARENA_RESOLVE = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/{path}"
LIVEBENCH_MODEL_JUDGMENT = "https://huggingface.co/datasets/livebench/model_judgment/resolve/main/data/leaderboard-00000-of-00001.parquet"
SWEBENCH_VERIFIED_API = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/verified"

SOURCE_REGISTRY = [
    {
        "source_id": "epoch_ai_models",
        "name": "Epoch AI notable AI models dataset",
        "url": EPOCH_URL,
        "type": "public_dataset",
        "notes": "Historical model metadata including publication dates, parameters, compute estimates and accessibility fields.",
    },
    {
        "source_id": "openrouter_models_api",
        "name": "OpenRouter public models API",
        "url": OPENROUTER_URL,
        "type": "public_api",
        "notes": "Current model catalog with API-visible prices, context windows, modality metadata and created timestamps.",
    },
    {
        "source_id": "lmarena_leaderboard",
        "name": "LMArena leaderboard dataset",
        "url": "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
        "type": "public_dataset",
        "notes": "Public parquet snapshots for text, vision, webdev, image and other arena leaderboard categories.",
    },
    {
        "source_id": "livebench_model_judgment",
        "name": "LiveBench model judgment dataset",
        "url": LIVEBENCH_MODEL_JUDGMENT,
        "type": "public_dataset",
        "notes": "Leaderboard split of LiveBench judgments. Treated as a benchmark source, not a universal capability score.",
    },
    {
        "source_id": "swe_bench_verified",
        "name": "SWE-bench Verified public experiment repository",
        "url": SWEBENCH_VERIFIED_API,
        "type": "public_dataset",
        "notes": "Public submission directories with metadata and resolved instance IDs.",
    },
    {
        "source_id": "openai_gpt55_docs",
        "name": "OpenAI GPT-5.5 model documentation",
        "url": "https://developers.openai.com/api/docs/models/gpt-5.5",
        "type": "official_docs",
        "notes": "Official model page used for GPT-5.5 identifiers, context window and documentation link.",
    },
    {
        "source_id": "anthropic_opus47_release",
        "name": "Anthropic Claude Opus 4.7 announcement",
        "url": "https://www.anthropic.com/news/claude-opus-4-7",
        "type": "official_release",
        "notes": "Official announcement used for release date and price statements.",
    },
    {
        "source_id": "jpl_de421",
        "name": "NASA/JPL DE421 ephemeris via Skyfield",
        "url": "https://ssd.jpl.nasa.gov/planets/eph_export.html",
        "type": "public_scientific_data",
        "notes": "Used for the optional geocentric ecliptic longitude calculations in the release-pattern appendix.",
    },
]

KEY_RELEASE_URLS = {
    "gpt-5.5": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "gpt-5.5-pro": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "claude-opus-4.7": "https://www.anthropic.com/news/claude-opus-4-7",
}

OPEN_WEIGHT_HINTS = (
    "meta-llama",
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "deepseek",
    "yi-",
    "gemma",
    "phi-",
    "command-r",
    "olmo",
    "falcon",
    "baichuan",
    "nous",
    "openchat",
)

CLOSED_VENDOR_HINTS = (
    "openai",
    "anthropic",
    "google",
    "x-ai",
    "perplexity",
    "amazon",
    "ai21",
    "microsoft",
)

ZODIAC = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]

PLANETS = ["Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]


@dataclass
class Downloaded:
    path: Path
    status: str
    url: str


@dataclass(frozen=True)
class EntityClassification:
    vendor: str
    model_family: str
    product_line: str


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, FIGURES, REPORT, APPENDIX, DOCS, ROOT / "notebooks", ROOT / "tests"]:
        path.mkdir(parents=True, exist_ok=True)


@contextmanager
def output_dirs(processed_dir: Path | None = None, report_dir: Path | None = None, appendix_dir: Path | None = None):
    global DATA_PROCESSED, REPORT, APPENDIX
    original_processed = DATA_PROCESSED
    original_report = REPORT
    original_appendix = APPENDIX
    if processed_dir is not None:
        DATA_PROCESSED = processed_dir
    if report_dir is not None:
        REPORT = report_dir
    if appendix_dir is not None:
        APPENDIX = appendix_dir
    try:
        yield
    finally:
        DATA_PROCESSED = original_processed
        REPORT = original_report
        APPENDIX = original_appendix


def slug(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def request_bytes(url: str, timeout: int = 60) -> bytes:
    headers = {"User-Agent": "ai-capability-signals/0.1 reproducible research"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def download(url: str, path: Path, overwrite: bool = False) -> Downloaded:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return Downloaded(path=path, status="cached", url=url)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(request_bytes(url))
    tmp.replace(path)
    return Downloaded(path=path, status="downloaded", url=url)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return clean_text(result.stdout)


def write_run_manifest(layer: str, manifest_dir: Path, outputs: list[Path], upstream_manifest: Path | None = None) -> None:
    existing_outputs = [path for path in outputs if path.exists() and path.is_file()]
    data = {
        "run_id": f"{layer}-{CAPTURED_AT}",
        "layer": layer,
        "generated_at": CAPTURED_AT,
        "reference_date": REFERENCE_DATE,
        "command": " ".join(sys.argv),
        "git_sha": current_git_sha(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "upstream_manifest": str(upstream_manifest.relative_to(ROOT)) if upstream_manifest and upstream_manifest.exists() else None,
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(existing_outputs)
        ],
    }
    write_json(manifest_dir / "run_manifest.json", data)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_default(v) for k, v in value.items()}
    return str(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "nan", "None", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return float(match.group()) if match else None


def token_price_per_million(value: Any) -> float | None:
    parsed = safe_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed * 1_000_000


def clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return fallback
    return text


def clean_label(value: Any, fallback: str = "") -> str:
    if isinstance(value, list):
        parts = [clean_text(v) for v in value if clean_text(v)]
        return ", ".join(parts) if parts else fallback
    return clean_text(value, fallback)


def parse_date(value: Any) -> tuple[str | None, str]:
    if value is None or str(value).strip() == "":
        return None, "missing"
    text = str(value).strip()
    try:
        dt = dateparser.parse(text, default=datetime(1900, 1, 1))
    except Exception:
        return None, "unparsed"
    if not dt:
        return None, "unparsed"
    precision = "day"
    if re.fullmatch(r"\d{4}", text):
        precision = "year"
    elif re.fullmatch(r"\d{4}-\d{2}", text):
        precision = "month"
    elif "T" in text or ":" in text:
        precision = "timestamp"
    return dt.date().isoformat(), precision


def canonical_vendor(organization: str = "", name: str = "") -> str:
    text = f"{organization} {name}".lower()
    rules = [
        ("OpenAI", ["openai"]),
        ("Anthropic", ["anthropic"]),
        ("Google", ["google", "deepmind"]),
        ("Meta", ["meta", "facebook"]),
        ("Mistral", ["mistral"]),
        ("DeepSeek", ["deepseek"]),
        ("Alibaba", ["qwen", "alibaba", "tongyi"]),
        ("xAI", ["x-ai", "xai", "grok"]),
        ("Microsoft", ["microsoft"]),
    ]
    for vendor, needles in rules:
        if any(needle in text for needle in needles):
            return vendor
    cleaned = clean_text(organization)
    if cleaned and "," not in cleaned:
        return cleaned
    return clean_text(name).split(":", 1)[0] if ":" in clean_text(name) else "Unknown"


def classify_entity(name: str, organization: str = "") -> EntityClassification:
    model = clean_text(name)
    org = clean_text(organization)
    text = f" {model} {org} ".lower()
    rules = [
        ("Claude", "Claude", [r"\bclaude\b"]),
        ("Gemini", "Gemini", [r"\bgemini\b", r"\bpalm\b"]),
        ("Gemma", "Gemma", [r"\bgemma\b"]),
        ("GPT", "GPT", [r"\bgpt(?:[-\s.]?\d|[-\s]?oss|[-\s]?latest|$)", r"\bo[1345](?:\b|-)", r"\bcodex\b"]),
        ("Llama", "Llama", [r"\bllama\b", r"\bmeta-llama\b"]),
        ("DeepSeek", "DeepSeek", [r"\bdeepseek\b"]),
        ("Qwen", "Qwen", [r"\bqwen(?:\d|\b)", r"\btongyi\b"]),
        ("Mistral", "Mistral", [r"\bmistral\b", r"\bmixtral\b"]),
        ("Grok", "Grok", [r"\bgrok\b", r"\bx-ai\b"]),
        ("Phi", "Phi", [r"\bphi[-\s]?\d", r"\bphi\b"]),
        ("Command", "Command", [r"\bcommand[-\s]?r", r"\bcommand\b"]),
        ("Yi", "Yi", [r"\byi[-\s]?\d"]),
    ]
    for family, product_line, patterns in rules:
        if any(re.search(pattern, text) for pattern in patterns):
            return EntityClassification(canonical_vendor(org, model), family, product_line)
    return EntityClassification(canonical_vendor(org, model), "Other", "Other")


def classify_family(name: str, organization: str = "") -> str:
    return classify_entity(name, organization).model_family


def classify_access(name: str, organization: str = "", model_accessibility: str = "", open_weights: str = "") -> str:
    text = f"{name} {organization} {model_accessibility} {open_weights}".lower()
    if "not released" in text or "unreleased" in text:
        return "not_released"
    if "open weights" in text or "open-weight" in text or open_weights.lower() == "yes":
        return "open_weight"
    if "gpt-oss" in text:
        return "open_weight"
    if any(h in text for h in OPEN_WEIGHT_HINTS):
        return "likely_open_weight"
    if "api" in text or "closed" in text or any(h in text for h in CLOSED_VENDOR_HINTS):
        return "closed_or_api"
    return "unknown"


def fetch_sources(overwrite: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["epoch"] = download(EPOCH_URL, DATA_RAW / "epoch" / "all_ai_models.csv", overwrite).path

    openrouter_path = download(OPENROUTER_URL, DATA_RAW / "openrouter" / "models.json", overwrite).path
    paths["openrouter"] = openrouter_path

    tree_path = download(LMARENA_TREE_URL, DATA_RAW / "lmarena" / "tree.json", overwrite).path
    paths["lmarena_tree"] = tree_path
    tree = read_json(tree_path)
    latest_paths = [
        item["path"]
        for item in tree
        if item.get("type") == "file" and item.get("path", "").endswith("latest-00000-of-00001.parquet")
    ]
    selected_categories = {"text", "vision", "webdev", "search", "document"}
    for rel_path in latest_paths:
        category = rel_path.split("/", 1)[0]
        if category in selected_categories:
            out = DATA_RAW / "lmarena" / rel_path
            paths[f"lmarena_{category}"] = download(LMARENA_RESOLVE.format(path=rel_path), out, overwrite).path

    paths["livebench"] = download(LIVEBENCH_MODEL_JUDGMENT, DATA_RAW / "livebench" / "model_judgment_leaderboard.parquet", overwrite).path

    swe_index_path = download(SWEBENCH_VERIFIED_API, DATA_RAW / "swebench" / "verified_index.json", overwrite).path
    paths["swebench_index"] = swe_index_path
    return paths


def build_sources_table() -> pd.DataFrame:
    rows = []
    for src in SOURCE_REGISTRY:
        rows.append({**src, "captured_at": CAPTURED_AT})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROCESSED / "sources.csv", index=False)
    return df


def normalize_epoch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        model = row.get("Model")
        release_date, precision = parse_date(row.get("Publication date"))
        org = row.get("Organization") or ""
        entity = classify_entity(str(model), org)
        params = safe_float(row.get("Parameters"))
        active_params = None
        notes = str(row.get("Parameters notes") or "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*B\s*active", notes, flags=re.I)
        if match:
            active_params = float(match.group(1)) * 1e9
        rows.append(
            {
                "model_id": f"epoch::{slug(model)}",
                "canonical_model": model,
                "vendor": entity.vendor,
                "model_family": entity.model_family,
                "product_line": entity.product_line,
                "family": entity.model_family,
                "source_vendor": org,
                "release_date": release_date,
                "release_precision": precision,
                "domain": row.get("Domain"),
                "task": row.get("Task"),
                "country": row.get("Country (of organization)"),
                "parameters": params,
                "active_parameters": active_params,
                "training_compute_flop": safe_float(row.get("Training compute (FLOP)")),
                "training_tokens": safe_float(row.get("Training dataset size (total)")),
                "training_compute_cost_2023_usd": safe_float(row.get("Training compute cost (2023 USD)")),
                "model_accessibility": row.get("Model accessibility"),
                "open_model_weights": row.get("Open model weights?"),
                "access_class": classify_access(
                    str(model),
                    org,
                    str(row.get("Model accessibility") or ""),
                    str(row.get("Open model weights?") or ""),
                ),
                "frontier_model": str(row.get("Frontier model") or "").strip(),
                "reference": row.get("Reference"),
                "source_url": clean_text(row.get("Link"), EPOCH_URL),
                "source_id": "epoch_ai_models",
                "confidence": row.get("Confidence"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "models_epoch_normalized.csv", index=False)
    return out


def normalize_openrouter(path: Path) -> pd.DataFrame:
    payload = read_json(path)
    rows = []
    for item in payload.get("data", []):
        model_id = item.get("id", "")
        source_vendor = model_id.split("/", 1)[0] if "/" in model_id else (item.get("name") or "").split(":", 1)[0]
        entity = classify_entity(str(item.get("name") or model_id), source_vendor)
        created = item.get("created")
        release_date = None
        release_precision = "missing"
        if created:
            release_date = datetime.fromtimestamp(int(created), UTC).date().isoformat()
            release_precision = "timestamp"
        pricing = item.get("pricing") or {}
        prompt_per_1m = token_price_per_million(pricing.get("prompt"))
        completion_per_1m = token_price_per_million(pricing.get("completion"))
        cache_read_per_1m = token_price_per_million(pricing.get("input_cache_read"))
        cache_write_per_1m = token_price_per_million(pricing.get("input_cache_write"))
        rows.append(
            {
                "model_id": f"openrouter::{model_id}",
                "openrouter_id": model_id,
                "canonical_model": item.get("name"),
                "vendor": entity.vendor,
                "source_vendor": source_vendor,
                "model_family": entity.model_family,
                "product_line": entity.product_line,
                "family": entity.model_family,
                "release_date": release_date,
                "release_precision": release_precision,
                "context_window": item.get("context_length"),
                "max_output_tokens": item.get("top_provider", {}).get("max_completion_tokens"),
                "input_usd_per_1m": prompt_per_1m,
                "output_usd_per_1m": completion_per_1m,
                "cache_read_usd_per_1m": cache_read_per_1m,
                "cache_write_usd_per_1m": cache_write_per_1m,
                "modality": (item.get("architecture") or {}).get("modality"),
                "input_modalities": ",".join((item.get("architecture") or {}).get("input_modalities") or []),
                "output_modalities": ",".join((item.get("architecture") or {}).get("output_modalities") or []),
                "access_class": classify_access(str(item.get("name") or model_id), source_vendor),
                "source_url": f"https://openrouter.ai/{model_id}",
                "source_id": "openrouter_models_api",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "models_openrouter_normalized.csv", index=False)
    prices = out[
        [
            "model_id",
            "openrouter_id",
            "canonical_model",
            "vendor",
            "model_family",
            "product_line",
            "family",
            "context_window",
            "input_usd_per_1m",
            "output_usd_per_1m",
            "cache_read_usd_per_1m",
            "cache_write_usd_per_1m",
            "source_url",
        ]
    ].copy()
    prices.to_csv(DATA_PROCESSED / "pricing_openrouter.csv", index=False)
    return out


def create_master_models(epoch: pd.DataFrame, openrouter: pd.DataFrame) -> pd.DataFrame:
    epoch_keep = epoch.assign(source_dataset="epoch")[
        [
            "model_id",
            "canonical_model",
            "vendor",
            "model_family",
            "product_line",
            "family",
            "release_date",
            "release_precision",
            "access_class",
            "parameters",
            "active_parameters",
            "training_compute_flop",
            "training_tokens",
            "training_compute_cost_2023_usd",
            "source_url",
            "source_id",
            "source_dataset",
        ]
    ]
    router_keep = openrouter.assign(
        source_dataset="openrouter",
        parameters=np.nan,
        active_parameters=np.nan,
        training_compute_flop=np.nan,
        training_tokens=np.nan,
        training_compute_cost_2023_usd=np.nan,
        source_id="openrouter_models_api",
    )[
        [
            "model_id",
            "canonical_model",
            "vendor",
            "model_family",
            "product_line",
            "family",
            "release_date",
            "release_precision",
            "access_class",
            "parameters",
            "active_parameters",
            "training_compute_flop",
            "training_tokens",
            "training_compute_cost_2023_usd",
            "source_url",
            "source_id",
            "source_dataset",
        ]
    ]
    master = pd.concat([epoch_keep, router_keep], ignore_index=True)
    master.to_csv(DATA_PROCESSED / "models_master.csv", index=False)
    return master


def parse_lmarena(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for parquet in sorted((raw_dir / "lmarena").glob("*/latest-00000-of-00001.parquet")):
        category = parquet.parent.name
        try:
            frame = pd.read_parquet(parquet)
        except Exception as exc:
            rows.append({"category": category, "error": str(exc)})
            continue
        frame = frame.copy()
        frame["category"] = category
        frame["source_id"] = "lmarena_leaderboard"
        frame["source_url"] = f"https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/tree/main/{category}"
        cols = {c.lower(): c for c in frame.columns}
        model_col = cols.get("model") or cols.get("model_name") or cols.get("name")
        rating_col = cols.get("rating") or cols.get("arena score") or cols.get("score")
        votes_col = cols.get("votes") or cols.get("num_battles")
        license_col = cols.get("license")
        org_col = cols.get("organization") or cols.get("creator") or cols.get("organization_name")
        for _, row in frame.iterrows():
            model = row.get(model_col) if model_col else None
            rating = safe_float(row.get(rating_col)) if rating_col else None
            rows.append(
                {
                    "benchmark": f"LMArena {category}",
                    "category": category,
                    "model": model,
                    "score": rating,
                    "unit": "arena_rating" if rating_col else "unknown",
                    "votes": safe_float(row.get(votes_col)) if votes_col else None,
                    "license": row.get(license_col) if license_col else None,
                    "vendor": row.get(org_col) if org_col else "",
                    "access_class": classify_access(str(model), str(row.get(org_col) if org_col else ""), "", str(row.get(license_col) if license_col else "")),
                    "source_id": "lmarena_leaderboard",
                    "source_url": f"https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/tree/main/{category}",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "benchmarks_lmarena_latest.csv", index=False)
    return out


def parse_livebench(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.to_csv(DATA_PROCESSED / "livebench_model_judgment_raw_preview.csv", index=False)
    cols = {c.lower(): c for c in frame.columns}
    model_col = cols.get("model") or cols.get("model_id") or cols.get("model_name")
    category_col = cols.get("category")
    task_col = cols.get("task")
    score_col = None
    for candidate in ["score", "parsed_score", "judgment_score", "rating"]:
        if candidate in cols:
            score_col = cols[candidate]
            break
    rows = []
    if model_col and score_col:
        grouped = frame.groupby([model_col, category_col] if category_col else [model_col])
        for keys, group in grouped:
            model = keys[0] if isinstance(keys, tuple) else keys
            category = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else "all"
            score = pd.to_numeric(group[score_col], errors="coerce").mean()
            rows.append(
                {
                    "benchmark": "LiveBench",
                    "category": category,
                    "model": model,
                    "score": score,
                    "unit": "mean_judgment_score",
                    "n": len(group),
                    "source_id": "livebench_model_judgment",
                    "source_url": "https://huggingface.co/datasets/livebench/model_judgment",
                }
            )
    else:
        # Keep a schema note rather than inventing a score when the public format changes.
        rows.append(
            {
                "benchmark": "LiveBench",
                "category": "schema",
                "model": "not_extracted",
                "score": np.nan,
                "unit": "not_extracted",
                "n": len(frame),
                "source_id": "livebench_model_judgment",
                "source_url": "https://huggingface.co/datasets/livebench/model_judgment",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "benchmarks_livebench.csv", index=False)
    return out


def fetch_swebench_entries(index_path: Path, overwrite: bool = False, limit: int | None = None) -> pd.DataFrame:
    index = read_json(index_path)
    rows = []
    cutoff = pd.to_datetime(REFERENCE_DATE).date()
    selected = []
    for item in index:
        if item.get("type") != "dir":
            continue
        match = re.match(r"(\d{8})_", item.get("name", ""))
        if match:
            try:
                submission_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                submission_date = None
            if submission_date and submission_date > cutoff:
                continue
        selected.append(item)
    selected = sorted(selected, key=lambda item: item.get("name", ""))
    if limit:
        selected = selected[:limit]
    for item in selected:
        name = item["name"]
        base_api = item["url"]
        raw_base = f"https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified/{name}"
        local_dir = DATA_RAW / "swebench" / name
        local_dir.mkdir(parents=True, exist_ok=True)
        meta_path = local_dir / "metadata.yaml"
        results_path = local_dir / "results.json"
        try:
            download(f"{raw_base}/metadata.yaml", meta_path, overwrite)
            download(f"{raw_base}/results/results.json", results_path, overwrite)
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            results = read_json(results_path)
        except Exception as exc:
            rows.append(
                {
                    "submission": name,
                    "model": None,
                    "system_name": None,
                    "resolved": np.nan,
                    "total": 500,
                    "score": np.nan,
                    "error": str(exc),
                    "source_url": base_api,
                    "source_id": "swe_bench_verified",
                }
            )
            continue
        tags = meta.get("tags") or {}
        info = meta.get("info") or {}
        model_tags = tags.get("model") or []
        if isinstance(model_tags, str):
            model_tags = [model_tags]
        system_name = clean_label(info.get("name"), clean_label(meta.get("name"), name))
        model_label = clean_label(model_tags, system_name)
        org_label = clean_label(tags.get("org"), clean_label(meta.get("org")))
        resolved = len(results.get("resolved") or [])
        total = 500
        rows.append(
            {
                "submission": name,
                "model": model_label,
                "system_name": system_name,
                "org": org_label,
                "os_model": tags.get("os_model"),
                "os_system": tags.get("os_system", meta.get("oss")),
                "resolved": resolved,
                "total": total,
                "score": resolved / total * 100,
                "unit": "percent_resolved",
                "source_url": f"https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/{name}",
                "source_id": "swe_bench_verified",
                "error": "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "benchmarks_swebench_verified.csv", index=False)
    return out


def build_manual_key_models(openrouter: pd.DataFrame) -> pd.DataFrame:
    wanted = ["gpt-5.5", "gpt-5.5-pro", "claude-opus-4.7", "claude-opus-4.7-fast", "gpt-5", "claude-opus-4"]
    rows = []
    for _, row in openrouter.iterrows():
        or_id = str(row.get("openrouter_id", ""))
        if any(w in or_id for w in wanted):
            key = next((w for w in wanted if w in or_id), or_id)
            rows.append(
                {
                    "model": row["canonical_model"],
                    "openrouter_id": or_id,
                    "vendor": row["vendor"],
                    "model_family": row.get("model_family", row.get("family")),
                    "family": row.get("model_family", row.get("family")),
                    "release_date": row["release_date"],
                    "context_window": row["context_window"],
                    "input_usd_per_1m": row["input_usd_per_1m"],
                    "output_usd_per_1m": row["output_usd_per_1m"],
                    "source_url": KEY_RELEASE_URLS.get(key, row["source_url"]),
                    "secondary_source_url": row["source_url"],
                    "notes": "Key current frontier/API model included from public catalog and official release/model pages.",
                }
            )
    out = pd.DataFrame(rows).sort_values(["vendor", "release_date", "model"])
    out.to_csv(DATA_PROCESSED / "key_current_models.csv", index=False)
    return out


def build_transparency_index(epoch: pd.DataFrame, openrouter: pd.DataFrame) -> pd.DataFrame:
    epoch_recent = epoch.copy()
    epoch_recent["year"] = pd.to_datetime(epoch_recent["release_date"], errors="coerce").dt.year
    epoch_recent = epoch_recent[epoch_recent["year"].fillna(0) >= 2020]
    rows = []
    for vendor, group in epoch_recent.groupby("vendor", dropna=True):
        if not vendor or len(group) < 2:
            continue
        fields = {
            "parameters": group["parameters"].notna().mean(),
            "training_compute": group["training_compute_flop"].notna().mean(),
            "training_tokens": group["training_tokens"].notna().mean(),
            "accessibility": group["access_class"].ne("unknown").mean(),
            "source_url": group["source_url"].notna().mean(),
        }
        score = sum(fields.values()) / len(fields) * 100
        rows.append({"vendor": vendor, "models": len(group), "transparency_score": score, **fields})
    if not rows:
        out = pd.DataFrame(columns=["vendor", "models", "transparency_score", "parameters", "training_compute", "training_tokens", "accessibility", "source_url"])
    else:
        out = pd.DataFrame(rows).sort_values("transparency_score", ascending=False)
    out.to_csv(DATA_PROCESSED / "transparency_index.csv", index=False)
    return out


def moon_phase(date: datetime) -> str:
    # Approximation: synodic month phase from a known new moon. Good enough for pattern testing, not astronomy claims.
    known_new_moon = datetime(2000, 1, 6, 18, 14, tzinfo=UTC)
    days = (date - known_new_moon).total_seconds() / 86400
    age = days % 29.53058867
    buckets = [
        (1.84566, "New"),
        (5.53699, "Waxing crescent"),
        (9.22831, "First quarter"),
        (12.91963, "Waxing gibbous"),
        (16.61096, "Full"),
        (20.30228, "Waning gibbous"),
        (23.99361, "Last quarter"),
        (27.68493, "Waning crescent"),
        (29.53059, "New"),
    ]
    for max_age, label in buckets:
        if age < max_age:
            return label
    return "New"


def mercury_retrograde_approx(date: datetime) -> bool:
    # Heuristic windows; exact retrograde calculations require apparent geocentric longitude velocity.
    # Used only as a deliberately weird feature and flagged as approximate.
    year = date.year
    windows = [
        (datetime(year, 1, 1, tzinfo=UTC), datetime(year, 1, 18, tzinfo=UTC)),
        (datetime(year, 4, 1, tzinfo=UTC), datetime(year, 4, 25, tzinfo=UTC)),
        (datetime(year, 8, 1, tzinfo=UTC), datetime(year, 8, 25, tzinfo=UTC)),
        (datetime(year, 11, 20, tzinfo=UTC), datetime(year, 12, 15, tzinfo=UTC)),
    ]
    return any(start <= date <= end for start, end in windows)


def add_planetary_features(events: pd.DataFrame) -> pd.DataFrame:
    try:
        from skyfield.api import Loader
        from skyfield.framelib import ecliptic_frame
    except Exception as exc:
        events["astro_status"] = f"skyfield_unavailable: {exc}"
        return events

    sky_dir = DATA_RAW / "skyfield"
    sky_dir.mkdir(parents=True, exist_ok=True)
    load = Loader(str(sky_dir))
    try:
        eph = load("de421.bsp")
    except Exception as exc:
        events["astro_status"] = f"ephemeris_unavailable: {exc}"
        return events
    ts = load.timescale()
    earth = eph["earth"]
    body_map = {
        "Sun": "sun",
        "Mercury": "mercury",
        "Venus": "venus",
        "Mars": "mars",
        "Jupiter": "jupiter barycenter",
        "Saturn": "saturn barycenter",
        "Uranus": "uranus barycenter",
        "Neptune": "neptune barycenter",
        "Pluto": "pluto barycenter",
    }
    features: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        date_text = row.get("release_date")
        if not date_text or pd.isna(date_text):
            features.append({})
            continue
        dt = dateparser.parse(str(date_text))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        t = ts.from_datetime(dt)
        feat: dict[str, Any] = {"astro_status": "ok"}
        for planet, body_name in body_map.items():
            try:
                apparent = earth.at(t).observe(eph[body_name]).apparent()
                lat, lon, distance = apparent.frame_latlon(ecliptic_frame)
                longitude = lon.degrees % 360
                sign = ZODIAC[int(longitude // 30)]
                feat[f"{planet.lower()}_longitude"] = longitude
                feat[f"{planet.lower()}_sign"] = sign
            except Exception as exc:
                feat[f"{planet.lower()}_sign"] = None
                feat["astro_status"] = f"partial: {exc}"
        features.append(feat)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(features)], axis=1)


def build_oracle_events(master: pd.DataFrame, openrouter: pd.DataFrame) -> pd.DataFrame:
    master_recent = master.copy()
    master_recent["date_dt"] = pd.to_datetime(master_recent["release_date"], errors="coerce", utc=True)
    master_recent = master_recent.dropna(subset=["date_dt"])
    families = {"GPT", "Claude", "Gemini", "Llama", "DeepSeek", "Qwen", "Mistral", "Grok"}
    family_col = "model_family" if "model_family" in master_recent.columns else "family"
    events = master_recent[
        (master_recent[family_col].isin(families))
        & (master_recent["date_dt"].dt.year >= 2018)
        & (master_recent["date_dt"].dt.date <= pd.to_datetime(REFERENCE_DATE).date())
    ].copy()
    # Keep the appendix readable: use one row per family/model/date/source and favor timestamped OpenRouter rows for recent APIs.
    events["event_key"] = events[family_col].fillna("") + "::" + events["canonical_model"].fillna("").map(slug) + "::" + events["release_date"].fillna("")
    events = events.sort_values(["release_precision", "source_dataset"], ascending=[False, False]).drop_duplicates("event_key")
    events = events.sort_values("date_dt")
    events["weekday"] = events["date_dt"].dt.day_name()
    events["month"] = events["date_dt"].dt.month_name()
    events["quarter"] = "Q" + events["date_dt"].dt.quarter.astype(str)
    events["moon_phase"] = events["date_dt"].map(lambda d: moon_phase(d.to_pydatetime()))
    events["mercury_retrograde_approx"] = events["date_dt"].map(lambda d: mercury_retrograde_approx(d.to_pydatetime()))
    enriched = add_planetary_features(events)
    enriched.to_csv(DATA_PROCESSED / "oracle_release_events.csv", index=False)
    return enriched


def permutation_p_value(observed: int, categories: int, draws: int, simulations: int = 10000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(draws, [1 / categories] * categories, size=simulations).max(axis=1)
    return float((np.sum(sims >= observed) + 1) / (simulations + 1))


def date_aware_p_value(events: pd.DataFrame, feature: str, top_count: int, simulations: int = 5000, seed: int = 42) -> float | None:
    if "date_dt" not in events.columns or feature not in {"weekday", "month", "quarter"}:
        return None
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(events["date_dt"], errors="coerce", utc=True).dropna()
    if dates.empty:
        return None
    year_counts = dates.dt.year.value_counts().to_dict()
    maxima = []
    for _ in range(simulations):
        sampled = []
        for year, count in year_counts.items():
            start = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
            end = pd.Timestamp(year=int(year), month=12, day=31, tz="UTC")
            span = (end - start).days + 1
            offsets = rng.integers(0, span, size=int(count))
            sampled.extend(start + pd.to_timedelta(offsets, unit="D"))
        sampled_series = pd.Series(sampled)
        if feature == "weekday":
            buckets = sampled_series.dt.day_name()
        elif feature == "month":
            buckets = sampled_series.dt.month_name()
        else:
            buckets = "Q" + sampled_series.dt.quarter.astype(str)
        maxima.append(int(buckets.value_counts().max()))
    return float((np.sum(np.array(maxima) >= top_count) + 1) / (simulations + 1))


def build_oracle_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if events.empty:
        out = pd.DataFrame(rows)
        out.to_csv(DATA_PROCESSED / "oracle_pattern_tests.csv", index=False)
        return out

    def add_feature(feature: str, categories: int, approx: bool = False) -> None:
        counts = events[feature].dropna().astype(str).value_counts()
        if counts.empty:
            return
        top_label = counts.index[0]
        top_count = int(counts.iloc[0])
        if feature in {"weekday", "month", "quarter"}:
            p_value = date_aware_p_value(events, feature, top_count)
            null_model = "year-preserving random dates"
            verdict = "calendar_cluster_not_causal" if p_value is not None and p_value < 0.01 else "no_robust_calendar_cluster"
        elif feature.endswith("_sign") and feature not in {"sun_sign"}:
            p_value = np.nan
            null_model = "not tested; slow planet signs are dominated by the sample year distribution"
            verdict = "temporal_clustering_expected"
        else:
            p_value = permutation_p_value(top_count, categories, int(counts.sum()))
            null_model = "uniform bucket permutation"
            verdict = "cluster_detected_not_causal" if p_value < 0.01 else "no_robust_cluster"
        rows.append(
            {
                "feature": feature,
                "top_bucket": top_label,
                "top_count": top_count,
                "n": int(counts.sum()),
                "share": top_count / counts.sum(),
                "null_categories": categories,
                "permutation_p_value": p_value,
                "null_model": null_model,
                "approximate_feature": approx,
                "verdict": verdict,
            }
        )

    add_feature("weekday", 7)
    add_feature("month", 12)
    add_feature("moon_phase", 8, approx=True)
    add_feature("quarter", 4)
    for planet in PLANETS:
        col = f"{planet.lower()}_sign"
        if col in events.columns:
            add_feature(col, 12)
    out = pd.DataFrame(rows).sort_values("permutation_p_value", na_position="last")
    out.to_csv(DATA_PROCESSED / "oracle_pattern_tests.csv", index=False)
    return out


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf6",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#e2dfd6",
            "grid.linewidth": 0.8,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_release_timeline(epoch: pd.DataFrame) -> None:
    df = epoch.copy()
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df[(df["year"] >= 2010) & (df["year"] <= 2026)]
    counts = df.pivot_table(index="year", columns="access_class", values="model_id", aggfunc="count", fill_value=0)
    wanted = [c for c in ["closed_or_api", "open_weight", "likely_open_weight", "unknown"] if c in counts.columns]
    counts[wanted].plot(kind="bar", stacked=True, figsize=(11, 5), color=["#32324f", "#d95f02", "#e6ab02", "#999999"])
    plt.title("Notable AI Model Releases by Accessibility Class")
    plt.xlabel("Release year")
    plt.ylabel("Model count in Epoch AI dataset")
    plt.legend(title="Access class")
    save_fig(FIGURES / "model_release_timeline.png")


def plot_compute_trend(epoch: pd.DataFrame) -> None:
    df = epoch.copy()
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df[(df["year"] >= 2010) & df["training_compute_flop"].notna()]
    colors = df["access_class"].map({"open_weight": "#d95f02", "likely_open_weight": "#e6ab02", "closed_or_api": "#32324f"}).fillna("#777777")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["year"], df["training_compute_flop"], c=colors, alpha=0.75, s=38, edgecolors="white", linewidths=0.5)
    plt.yscale("log")
    plt.title("Training Compute Has Grown Superlinearly, But Disclosure Is Uneven")
    plt.xlabel("Publication year")
    plt.ylabel("Training compute (FLOP, log scale)")
    save_fig(FIGURES / "training_compute_trend.png")


def plot_context_price(openrouter: pd.DataFrame) -> None:
    df = openrouter.copy()
    df = df[df["context_window"].notna() & df["output_usd_per_1m"].notna()]
    df = df[df["output_usd_per_1m"] > 0]
    colors = df["access_class"].map({"likely_open_weight": "#e6ab02", "open_weight": "#d95f02", "closed_or_api": "#32324f"}).fillna("#777777")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["output_usd_per_1m"], df["context_window"], c=colors, alpha=0.7, s=35, edgecolors="white", linewidths=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Context Window vs Output Price on Public API Catalogs")
    plt.xlabel("Output price (USD per 1M tokens, log)")
    plt.ylabel("Context window (tokens, log)")
    save_fig(FIGURES / "context_window_price.png")


def plot_transparency(transparency: pd.DataFrame) -> None:
    if transparency.empty:
        return
    top = transparency.sort_values("transparency_score", ascending=True).tail(15)
    plt.figure(figsize=(8, 7))
    plt.barh(top["vendor"].astype(str), top["transparency_score"], color="#2a9d8f")
    plt.title("Transparency Index by Vendor in Epoch AI Dataset")
    plt.xlabel("Disclosure coverage score (0-100)")
    save_fig(FIGURES / "transparency_index.png")


def plot_swebench(swe: pd.DataFrame) -> None:
    df = swe[swe["score"].notna()].sort_values("score", ascending=False).head(15)
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    labels = (
        df["system_name"]
        .fillna(df["model"])
        .fillna(df["submission"])
        .fillna("unknown")
        .astype(str)
        .str.slice(0, 42)
        .tolist()
    )
    scores = df["score"].astype(float).tolist()
    plt.barh(labels[::-1], scores[::-1], color="#264653")
    plt.title("Top SWE-bench Verified Public Submissions")
    plt.xlabel("Resolved instances (%)")
    save_fig(FIGURES / "swebench_top.png")


def plot_lmarena(lma: pd.DataFrame) -> None:
    df = lma[lma["score"].notna()].copy()
    if df.empty:
        return
    df["access_bucket"] = df["access_class"].replace({"closed_or_api": "closed/API", "likely_open_weight": "likely open-weight"})
    cats = ["closed/API", "likely open-weight", "open_weight", "unknown"]
    data = [df[df["access_bucket"] == c]["score"].dropna().values for c in cats if len(df[df["access_bucket"] == c])]
    labels = [c for c in cats if len(df[df["access_bucket"] == c])]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.title("LMArena Latest Ratings: Distribution by Access Class")
    plt.ylabel("Arena rating")
    plt.xticks(rotation=20, ha="right")
    save_fig(FIGURES / "lmarena_access_distribution.png")


def plot_oracle_heatmap(events: pd.DataFrame) -> None:
    sign_cols = [f"{p.lower()}_sign" for p in PLANETS if f"{p.lower()}_sign" in events.columns]
    if not sign_cols:
        return
    matrix = []
    for col in sign_cols:
        counts = events[col].value_counts()
        matrix.append([counts.get(sign, 0) for sign in ZODIAC])
    plt.figure(figsize=(13, 5.5))
    plt.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Release events")
    plt.yticks(range(len(sign_cols)), [c.replace("_sign", "").title() for c in sign_cols])
    plt.xticks(range(len(ZODIAC)), [s[:3] for s in ZODIAC])
    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            if val:
                plt.text(x, y, str(val), ha="center", va="center", fontsize=8, color="#111")
    plt.title("The Frontier Model Oracle: Planetary Sign Occupancy at Model Releases")
    plt.xlabel("Zodiac sign")
    save_fig(FIGURES / "oracle_zodiac_heatmap.png")


def plot_oracle_weekday(events: pd.DataFrame) -> None:
    if events.empty:
        return
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    families = sorted([f for f in events["family"].dropna().unique() if f])
    matrix = []
    for family in families:
        counts = events[events["family"] == family]["weekday"].value_counts()
        matrix.append([counts.get(day, 0) for day in weekdays])
    plt.figure(figsize=(10, max(4, len(families) * 0.45)))
    plt.imshow(matrix, aspect="auto", cmap="PuBuGn")
    plt.colorbar(label="Release events")
    plt.yticks(range(len(families)), families)
    plt.xticks(range(len(weekdays)), [d[:3] for d in weekdays])
    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            if val:
                plt.text(x, y, str(val), ha="center", va="center", fontsize=8, color="#111")
    plt.title("Release Weekday Fingerprints by Model Family")
    save_fig(FIGURES / "oracle_weekday_family_heatmap.png")


def make_plots(epoch: pd.DataFrame, openrouter: pd.DataFrame, transparency: pd.DataFrame, swe: pd.DataFrame, lma: pd.DataFrame, oracle: pd.DataFrame | None = None) -> None:
    apply_plot_style()
    plot_release_timeline(epoch)
    plot_compute_trend(epoch)
    plot_context_price(openrouter)
    plot_transparency(transparency)
    plot_swebench(swe)
    plot_lmarena(lma)
    if oracle is not None:
        plot_oracle_heatmap(oracle)
        plot_oracle_weekday(oracle)


def figure_md(name: str, alt: str) -> str:
    return f"![{alt}](../figures/{name})"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if abs(float(value)) >= 1e15:
        return f"{float(value):.{digits}e}"
    if abs(float(value)) >= 1_000_000:
        return f"{float(value):,.0f}"
    return f"{float(value):,.{digits}f}"


def summarize_findings(epoch: pd.DataFrame, openrouter: pd.DataFrame, swe: pd.DataFrame, lma: pd.DataFrame, oracle_summary: pd.DataFrame | None = None) -> dict[str, Any]:
    epoch_lang = epoch[epoch["domain"].astype(str).str.contains("Language", na=False)]
    epoch_lang["year"] = pd.to_datetime(epoch_lang["release_date"], errors="coerce").dt.year
    max_compute = epoch_lang.loc[epoch_lang["training_compute_flop"].idxmax()] if epoch_lang["training_compute_flop"].notna().any() else None
    gpt55 = openrouter[openrouter["openrouter_id"].astype(str).str.contains("openai/gpt-5.5$", regex=True, na=False)]
    opus47 = openrouter[openrouter["openrouter_id"].astype(str).str.contains("anthropic/claude-opus-4.7$", regex=True, na=False)]
    openrouter_closed = openrouter[openrouter["access_class"] == "closed_or_api"]
    openrouter_open = openrouter[openrouter["access_class"].isin(["open_weight", "likely_open_weight"])]
    median_closed_output = openrouter_closed["output_usd_per_1m"].median()
    median_open_output = openrouter_open["output_usd_per_1m"].median()
    swe_top = swe[swe["score"].notna()].sort_values("score", ascending=False).head(1)
    lma_top = lma[lma["score"].notna()].sort_values("score", ascending=False).head(1)
    oracle_top = oracle_summary.head(1) if oracle_summary is not None and not oracle_summary.empty else pd.DataFrame()
    return {
        "epoch_rows": int(len(epoch)),
        "openrouter_rows": int(len(openrouter)),
        "lmarena_rows": int(len(lma)),
        "swebench_rows": int(len(swe)),
        "max_compute_model": None if max_compute is None else max_compute["canonical_model"],
        "max_compute_flop": None if max_compute is None else max_compute["training_compute_flop"],
        "gpt55_context": None if gpt55.empty else gpt55.iloc[0]["context_window"],
        "gpt55_input": None if gpt55.empty else gpt55.iloc[0]["input_usd_per_1m"],
        "gpt55_output": None if gpt55.empty else gpt55.iloc[0]["output_usd_per_1m"],
        "opus47_context": None if opus47.empty else opus47.iloc[0]["context_window"],
        "opus47_input": None if opus47.empty else opus47.iloc[0]["input_usd_per_1m"],
        "opus47_output": None if opus47.empty else opus47.iloc[0]["output_usd_per_1m"],
        "median_closed_output": median_closed_output,
        "median_open_output": median_open_output,
        "swe_top_system": None if swe_top.empty else swe_top.iloc[0]["system_name"],
        "swe_top_score": None if swe_top.empty else swe_top.iloc[0]["score"],
        "lma_top_model": None if lma_top.empty else lma_top.iloc[0]["model"],
        "lma_top_category": None if lma_top.empty else lma_top.iloc[0]["category"],
        "lma_top_score": None if lma_top.empty else lma_top.iloc[0]["score"],
        "oracle_weirdest": None if oracle_top.empty else oracle_top.iloc[0].to_dict(),
    }


def table_markdown(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    small = df[columns].head(n).copy().fillna("")
    return small.to_markdown(index=False)


def write_reports(
    epoch: pd.DataFrame,
    openrouter: pd.DataFrame,
    transparency: pd.DataFrame,
    swe: pd.DataFrame,
    lma: pd.DataFrame,
    oracle_summary: pd.DataFrame | None = None,
) -> None:
    findings = summarize_findings(epoch, openrouter, swe, lma, oracle_summary)
    write_json(REPORT / "findings.json", findings)
    price_ratio = (
        findings["median_closed_output"] / findings["median_open_output"]
        if findings.get("median_closed_output") and findings.get("median_open_output")
        else None
    )

    key_models = pd.read_csv(DATA_PROCESSED / "key_current_models.csv")
    top_swe = swe[swe["score"].notna()].sort_values("score", ascending=False).head(12)
    top_lma = (
        lma[lma["score"].notna()]
        .sort_values("score", ascending=False)
        .drop_duplicates(["category", "model"])
        .head(12)
    )
    top_context = (
        openrouter[
            openrouter["context_window"].notna()
            & openrouter["input_usd_per_1m"].notna()
            & openrouter["output_usd_per_1m"].notna()
        ]
        .sort_values("context_window", ascending=False)
        .head(12)
    )

    main_report = f"""# AI Capability Signals: Scaling, Prices, Open Weights, and Benchmark Caveats

Reference date: {REFERENCE_DATE}. Data captured at: {CAPTURED_AT}.

This repository is a hiring-portfolio research project focused on reproducible public-data ingestion and defensible caveats. It combines public model metadata, API catalogs and benchmark datasets to analyze frontier AI systems without collapsing unrelated signals into one fake universal score.

## Executive Findings

- The processed project currently includes **{findings['epoch_rows']:,} Epoch AI model rows**, **{findings['openrouter_rows']:,} OpenRouter API catalog rows**, **{findings['lmarena_rows']:,} LMArena leaderboard rows**, and **{findings['swebench_rows']:,} SWE-bench Verified public submissions**.
- GPT-5.5 appears in the public API catalog with a **{fmt_num(findings['gpt55_context'], 0)} token context window**, **${fmt_num(findings['gpt55_input'])}/1M input tokens** and **${fmt_num(findings['gpt55_output'])}/1M output tokens**.
- Claude Opus 4.7 appears with a **{fmt_num(findings['opus47_context'], 0)} token context window**, **${fmt_num(findings['opus47_input'])}/1M input tokens** and **${fmt_num(findings['opus47_output'])}/1M output tokens**.
- Across the current OpenRouter snapshot, the median listed output price is **${fmt_num(findings['median_closed_output'])}/1M tokens** for closed/API-classified models and **${fmt_num(findings['median_open_output'])}/1M tokens** for likely open-weight models. This is not a capability-adjusted claim; it is a public-catalog pricing snapshot.
- The largest disclosed training compute row in the normalized Epoch AI slice is **{findings['max_compute_model']}** at approximately **{fmt_num(findings['max_compute_flop'], 1)} FLOP**.

## Surprising Results

- The public API catalog snapshot shows likely open-weight models with a median output price about **{fmt_num(price_ratio, 1)}x lower** than closed/API-classified models, before any quality adjustment.
- Context length has become decoupled from frontier branding: several catalog models list **2M-token context windows**, while GPT-5.5 and Claude Opus 4.7 sit around the 1M-token tier in this snapshot.
- The highest selected LMArena rows are dominated by closed frontier systems, but the pricing table shows that open-weight/API-hosted alternatives compete on a very different cost curve.

## Key Current Models

{table_markdown(key_models, ['model', 'openrouter_id', 'release_date', 'context_window', 'input_usd_per_1m', 'output_usd_per_1m', 'source_url'], n=12)}

## Scaling and Accessibility

{figure_md('model_release_timeline.png', 'Notable AI model releases by accessibility class')}

{figure_md('training_compute_trend.png', 'Training compute trend')}

The important professional caveat is disclosure. Closed frontier systems often report less about parameters, tokens and compute than open-weight research releases. The transparency index below measures field coverage in the public Epoch AI dataset; it should be interpreted as public disclosure coverage, not model quality.

{figure_md('transparency_index.png', 'Transparency index by vendor')}

## Price, Context and Catalog Reality

{figure_md('context_window_price.png', 'Context window versus output price')}

Large context windows have become a product dimension in their own right. The chart deliberately separates context and price from benchmark claims: a one-million-token context window is useful only when the model can retrieve and reason over that context reliably, which must be tested separately.

Top context windows in the OpenRouter snapshot:

{table_markdown(top_context, ['canonical_model', 'openrouter_id', 'model_family', 'context_window', 'input_usd_per_1m', 'output_usd_per_1m'], n=12)}

## Benchmarks Without Fake Omniscience

The project does not average unrelated benchmarks into a single artificial "AI score." LMArena, SWE-bench and LiveBench measure different things and are stored separately.

{figure_md('swebench_top.png', 'Top SWE-bench Verified submissions')}

{table_markdown(top_swe, ['system_name', 'model', 'org', 'score', 'resolved', 'total', 'source_url'], n=12)}

{figure_md('lmarena_access_distribution.png', 'LMArena rating distribution by access class')}

Top rows from the selected LMArena latest snapshots:

{table_markdown(top_lma, ['benchmark', 'category', 'model', 'score', 'votes', 'access_class'], n=12)}

## Open-Weight vs Closed

The strongest result is not that open-weight models always beat closed models or vice versa. The real story is structural:

- Open-weight systems disclose more internals and make replication/inspection easier.
- Closed systems dominate some API/product leaderboards, but opacity makes scaling analysis harder.
- API price competition is compressing fast enough that the cost frontier can move independently from the capability frontier.
- The "open source" label is too coarse. This repo uses `open_weight`, `likely_open_weight`, `closed_or_api`, and `unknown` rather than treating all public weights as fully open-source systems.

## Limits

- Some current frontier model details are intentionally undisclosed by vendors.
- OpenRouter is an API catalog, not an official benchmark authority.
- Public benchmark formats change; extraction code preserves raw snapshots and avoids fabricating missing scores.
- Entity classification is explicit but still heuristic when source metadata is vague.

## Reproduce

```bash
uv run python -m frontier_ai.pipeline --write-reports
uv run python -m unittest discover -s tests
```
"""
    (REPORT / "frontier_ai_analysis.md").write_text(main_report, encoding="utf-8")

    if oracle_summary is not None:
        oracle_md = f"""# The Frontier Model Oracle

This appendix is the "go wild" part of the project: a serious implementation of a suspicious idea. It asks whether major model launches have release-calendar fingerprints: weekdays, months, moon phases, approximate Mercury retrograde windows and geocentric planetary zodiac signs.

It is not a prediction engine and it is not part of the portfolio headline. It is a demoted exploratory appendix showing how easy it is to manufacture calendar-looking patterns.

## Method

- Input events come from the normalized model table, restricted to GPT, Claude, Gemini, Llama, DeepSeek, Qwen, Mistral and Grok-family releases from 2018 through {REFERENCE_DATE}.
- Planetary positions use NASA/JPL DE421 ephemeris through Skyfield when available.
- Moon phase is an approximation, marked as approximate.
- Calendar features use a year-preserving random-date baseline.
- Slow-planet zodiac features are not assigned causal p-values because their signs are dominated by the years sampled.

## Pattern Tests

{table_markdown(oracle_summary, ['feature', 'top_bucket', 'top_count', 'n', 'share', 'permutation_p_value', 'null_model', 'approximate_feature', 'verdict'], n=25)}

## Heatmaps

{figure_md('oracle_zodiac_heatmap.png', 'Planetary sign occupancy heatmap')}

{figure_md('oracle_weekday_family_heatmap.png', 'Release weekday family heatmap')}

## Interpretation

This is the fun result: release calendars can look patterned even when generated from random dates. The appendix is valuable because it makes that visible. If a striking cluster appears, the next step is not "the planets did it"; it is "check vendor launch schedules, conference calendars, product marketing cycles, earnings windows and data leakage."
"""
        (APPENDIX / "frontier_model_oracle.md").write_text(oracle_md, encoding="utf-8")

    html = main_report.replace("../figures/", "../figures/")
    html = html.replace("\n", "\n")
    html_body = "\n".join(markdown_to_html_blocks(html))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Capability Signals</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #202124; background: #f7f5ef; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 48px 24px 80px; background: #fffdf8; }}
    h1, h2 {{ color: #171717; letter-spacing: 0; }}
    h1 {{ font-size: 42px; line-height: 1.05; }}
    h2 {{ margin-top: 42px; border-top: 1px solid #ddd8ca; padding-top: 26px; }}
    p, li {{ font-size: 16px; line-height: 1.65; }}
    img {{ max-width: 100%; border: 1px solid #ddd8ca; border-radius: 6px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; overflow-wrap: anywhere; }}
    th, td {{ border-bottom: 1px solid #e7e1d2; padding: 8px; text-align: left; vertical-align: top; }}
    pre {{ background: #171717; color: #fafafa; padding: 16px; overflow-x: auto; border-radius: 6px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body><main>{html_body}</main></body>
</html>
"""
    (REPORT / "frontier_ai_analysis.html").write_text(html_doc, encoding="utf-8")


def markdown_to_html_blocks(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    out: list[str] = []
    in_code = False
    in_table = False
    table_lines: list[str] = []
    list_open = False

    def flush_table() -> None:
        nonlocal table_lines, in_table
        if not table_lines:
            return
        rows = [line.strip().strip("|").split("|") for line in table_lines if "|" in line]
        if len(rows) >= 2:
            out.append("<table>")
            header = [c.strip() for c in rows[0]]
            out.append("<thead><tr>" + "".join(f"<th>{escape_html(c)}</th>" for c in header) + "</tr></thead><tbody>")
            for row in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{escape_html(c.strip())}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        table_lines = []
        in_table = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    for line in lines:
        if line.startswith("```"):
            flush_table()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                close_list()
                out.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            out.append(escape_html(line))
            continue
        if line.startswith("|") and "|" in line[1:]:
            close_list()
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            flush_table()
        if not line.strip():
            close_list()
            continue
        if line.startswith("# "):
            close_list()
            out.append(f"<h1>{escape_html(line[2:])}</h1>")
        elif line.startswith("## "):
            close_list()
            out.append(f"<h2>{escape_html(line[3:])}</h2>")
        elif line.startswith("- "):
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{inline_md(line[2:])}</li>")
        elif line.startswith("!["):
            close_list()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                out.append(f'<p><img alt="{escape_html(match.group(1))}" src="{escape_html(match.group(2))}"></p>')
        else:
            close_list()
            out.append(f"<p>{inline_md(line)}</p>")
    flush_table()
    close_list()
    return out


def escape_html(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md(text: str) -> str:
    text = escape_html(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def validate_outputs(require_reports: bool = False) -> None:
    required = [
        DATA_PROCESSED / "models_master.csv",
        DATA_PROCESSED / "models_openrouter_normalized.csv",
        DATA_PROCESSED / "models_epoch_normalized.csv",
        DATA_PROCESSED / "sources.csv",
        DATA_PROCESSED / "key_current_models.csv",
    ]
    if require_reports:
        required.append(REPORT / "frontier_ai_analysis.md")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")
    master = pd.read_csv(DATA_PROCESSED / "models_master.csv")
    if master["source_url"].isna().any():
        raise RuntimeError("Some master model rows are missing source_url")
    key = pd.read_csv(DATA_PROCESSED / "key_current_models.csv")
    ids = " ".join(key["openrouter_id"].astype(str).str.lower())
    if "gpt-5.5" not in ids or "claude-opus-4.7" not in ids:
        raise RuntimeError("Key current models missing GPT-5.5 or Claude Opus 4.7")


def run_sample_pipeline(write_reports_flag: bool = False) -> None:
    with output_dirs(DATA_SAMPLE_PROCESSED, SAMPLE_REPORT, SAMPLE_APPENDIX):
        _run_sample_pipeline_in_active_dirs(write_reports_flag=write_reports_flag)


def _run_sample_pipeline_in_active_dirs(write_reports_flag: bool = False) -> None:
    ensure_dirs()
    epoch = pd.DataFrame(
        [
            {
                "model_id": "sample::gpt-5",
                "canonical_model": "GPT-5",
                "vendor": "OpenAI",
                "model_family": "GPT",
                "product_line": "GPT",
                "family": "GPT",
                "release_date": "2025-08-07",
                "release_precision": "day",
                "domain": "Language",
                "task": "Language modeling",
                "parameters": np.nan,
                "active_parameters": np.nan,
                "training_compute_flop": 1e25,
                "training_tokens": np.nan,
                "training_compute_cost_2023_usd": np.nan,
                "access_class": "closed_or_api",
                "source_url": "sample://epoch/gpt-5",
                "source_id": "sample",
            },
            {
                "model_id": "sample::gemma-3",
                "canonical_model": "Gemma 3 27B",
                "vendor": "Google",
                "model_family": "Gemma",
                "product_line": "Gemma",
                "family": "Gemma",
                "release_date": "2025-03-12",
                "release_precision": "day",
                "domain": "Language",
                "task": "Language modeling",
                "parameters": 27e9,
                "active_parameters": np.nan,
                "training_compute_flop": 5e23,
                "training_tokens": np.nan,
                "training_compute_cost_2023_usd": np.nan,
                "access_class": "likely_open_weight",
                "source_url": "sample://epoch/gemma-3",
                "source_id": "sample",
            },
        ]
    )
    openrouter = pd.DataFrame(
        [
            {
                "model_id": "openrouter::openai/gpt-5.5",
                "openrouter_id": "openai/gpt-5.5",
                "canonical_model": "OpenAI: GPT-5.5",
                "vendor": "OpenAI",
                "source_vendor": "openai",
                "model_family": "GPT",
                "product_line": "GPT",
                "family": "GPT",
                "release_date": "2026-04-24",
                "release_precision": "timestamp",
                "context_window": 1050000,
                "max_output_tokens": 128000,
                "input_usd_per_1m": 5,
                "output_usd_per_1m": 30,
                "cache_read_usd_per_1m": np.nan,
                "cache_write_usd_per_1m": np.nan,
                "modality": "text->text",
                "input_modalities": "text",
                "output_modalities": "text",
                "access_class": "closed_or_api",
                "source_url": "sample://openrouter/gpt-5.5",
                "source_id": "sample",
            },
            {
                "model_id": "openrouter::anthropic/claude-opus-4.7",
                "openrouter_id": "anthropic/claude-opus-4.7",
                "canonical_model": "Anthropic: Claude Opus 4.7",
                "vendor": "Anthropic",
                "source_vendor": "anthropic",
                "model_family": "Claude",
                "product_line": "Claude",
                "family": "Claude",
                "release_date": "2026-04-16",
                "release_precision": "timestamp",
                "context_window": 1000000,
                "max_output_tokens": 128000,
                "input_usd_per_1m": 5,
                "output_usd_per_1m": 25,
                "cache_read_usd_per_1m": np.nan,
                "cache_write_usd_per_1m": np.nan,
                "modality": "text->text",
                "input_modalities": "text",
                "output_modalities": "text",
                "access_class": "closed_or_api",
                "source_url": "sample://openrouter/claude-opus-4.7",
                "source_id": "sample",
            },
        ]
    )
    build_sources_table()
    epoch.to_csv(DATA_PROCESSED / "models_epoch_normalized.csv", index=False)
    openrouter.to_csv(DATA_PROCESSED / "models_openrouter_normalized.csv", index=False)
    openrouter[["model_id", "openrouter_id", "canonical_model", "vendor", "model_family", "product_line", "family", "context_window", "input_usd_per_1m", "output_usd_per_1m", "source_url"]].to_csv(DATA_PROCESSED / "pricing_openrouter.csv", index=False)
    create_master_models(epoch, openrouter)
    build_manual_key_models(openrouter)
    lma = pd.DataFrame(columns=["benchmark", "category", "model", "score", "unit", "votes", "access_class", "source_url"])
    swe = pd.DataFrame(columns=["submission", "model", "system_name", "org", "score", "resolved", "total", "source_url"])
    pd.DataFrame(columns=["benchmark", "category", "model", "score", "unit", "n", "source_id", "source_url"]).to_csv(DATA_PROCESSED / "benchmarks_livebench.csv", index=False)
    lma.to_csv(DATA_PROCESSED / "benchmarks_lmarena_latest.csv", index=False)
    swe.to_csv(DATA_PROCESSED / "benchmarks_swebench_verified.csv", index=False)
    pd.concat([lma, swe.assign(benchmark="SWE-bench Verified", category="software_engineering")], ignore_index=True, sort=False).to_csv(DATA_PROCESSED / "benchmarks_all.csv", index=False)
    transparency = build_transparency_index(epoch, openrouter)
    oracle_summary = pd.DataFrame(columns=["feature", "top_bucket", "top_count", "n", "share", "permutation_p_value", "null_model", "approximate_feature", "verdict"])
    pd.DataFrame().to_csv(DATA_PROCESSED / "oracle_release_events.csv", index=False)
    oracle_summary.to_csv(DATA_PROCESSED / "oracle_pattern_tests.csv", index=False)
    if write_reports_flag:
        write_reports(epoch, openrouter, transparency, swe, lma, oracle_summary)
    validate_outputs(require_reports=write_reports_flag)
    write_run_manifest("processed_sample", DATA_PROCESSED, list(DATA_PROCESSED.glob("*.csv")))
    if write_reports_flag:
        write_run_manifest("report_sample", REPORT, list(REPORT.glob("*.*")), upstream_manifest=DATA_PROCESSED / "run_manifest.json")


def run_pipeline(overwrite: bool = False, skip_plots: bool = False, swe_limit: int | None = None, write_reports_flag: bool = False, sample: bool = False) -> None:
    if sample:
        run_sample_pipeline(write_reports_flag=write_reports_flag)
        print("Done. Sample outputs are in data/sample/processed/.", flush=True)
        return
    ensure_dirs()
    print("Fetching public sources...", flush=True)
    paths = fetch_sources(overwrite=overwrite)
    build_sources_table()
    print("Normalizing Epoch AI and OpenRouter...", flush=True)
    epoch = normalize_epoch(paths["epoch"])
    openrouter = normalize_openrouter(paths["openrouter"])
    create_master_models(epoch, openrouter)
    build_manual_key_models(openrouter)
    print("Parsing benchmark datasets...", flush=True)
    lma = parse_lmarena(DATA_RAW)
    live = parse_livebench(paths["livebench"])
    swe = fetch_swebench_entries(paths["swebench_index"], overwrite=overwrite, limit=swe_limit)
    benchmark_all = pd.concat([lma, live, swe.assign(benchmark="SWE-bench Verified", category="software_engineering")], ignore_index=True, sort=False)
    benchmark_all.to_csv(DATA_PROCESSED / "benchmarks_all.csv", index=False)
    print("Computing transparency and release-pattern features...", flush=True)
    transparency = build_transparency_index(epoch, openrouter)
    master = pd.read_csv(DATA_PROCESSED / "models_master.csv")
    oracle = build_oracle_events(master, openrouter)
    oracle_summary = build_oracle_summary(oracle)
    if not skip_plots:
        print("Rendering charts...", flush=True)
        make_plots(epoch, openrouter, transparency, swe, lma, oracle)
    if write_reports_flag:
        print("Writing generated reports...", flush=True)
        write_reports(epoch, openrouter, transparency, swe, lma, oracle_summary)
    validate_outputs(require_reports=write_reports_flag)
    write_run_manifest("processed", DATA_PROCESSED, list(DATA_PROCESSED.glob("*.csv")))
    if write_reports_flag:
        write_run_manifest("report", REPORT, list(REPORT.glob("*.*")), upstream_manifest=DATA_PROCESSED / "run_manifest.json")
    print("Done. Outputs are in data/processed/ plus optional report/ and figures/.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AI Capability Signals research repo.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download public source snapshots.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip chart rendering.")
    parser.add_argument("--swe-limit", type=int, default=None, help="Limit SWE-bench submissions for debugging.")
    parser.add_argument("--sample", action="store_true", help="Run a tiny no-network sample pipeline.")
    parser.add_argument("--write-reports", action="store_true", help="Write generated Markdown/HTML reports.")
    args = parser.parse_args()
    run_pipeline(overwrite=args.overwrite, skip_plots=args.skip_plots, swe_limit=args.swe_limit, write_reports_flag=args.write_reports, sample=args.sample)


if __name__ == "__main__":
    main()
