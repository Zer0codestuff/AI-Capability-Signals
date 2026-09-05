"""Refresh public evidence. Fail on schema drift; never silently invent missing values."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from pipeline.analysis import build_trends
from pipeline.policy import METR_RELIABLE_MINUTES, METR_VERSION

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "metr": {
        "name": "METR",
        "url": "https://metr.org/assets/benchmark_results_1_1.yaml",
        "page": "https://metr.org/time-horizons/",
        "description": "Time Horizon 1.1. Software, ML and cybersecurity tasks.",
    },
    "openrouter": {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/models",
        "page": "https://openrouter.ai/docs/guides/overview/models",
        "description": "Current API list prices and embedded Artificial Analysis scores.",
    },
    "epoch": {
        "name": "Epoch AI",
        "url": "https://epoch.ai/data/all_ai_models.csv",
        "page": "https://epoch.ai/data/ai-models",
        "description": "Curated model records, including researcher estimates and missing values.",
    },
}

NAMES = {
    "gpt_4": "GPT-4",
    "gpt_4_1106_inspect": "GPT-4 (Nov 2023)",
    "claude_mythos_preview_early_inspect": "Claude Mythos Preview (early)",
    "claude_3_5_sonnet_20240620_inspect": "Claude 3.5 Sonnet (Jun 2024)",
    "claude_3_5_sonnet_20241022_inspect": "Claude 3.5 Sonnet (Oct 2024)",
    "gpt_5_2025_08_07_inspect": "GPT-5",
    "gpt_4o_inspect": "GPT-4o",
}


def model_name(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    text = key.removesuffix("_inspect").replace("_", " ")
    text = re.sub(r"(\d) (\d)", r"\1.\2", text)
    text = text.title().replace("Gpt ", "GPT-").replace("Gpt", "GPT")
    return re.sub(r"^O(\d)", r"o\1", text)


def positive(value, *, zero=False):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and (number >= 0 if zero else number > 0) else None


def parse_metr(body: bytes) -> tuple[list[dict], dict]:
    data = yaml.safe_load(body)
    if data.get("benchmark_name") != METR_VERSION:
        raise ValueError("METR benchmark version changed. Review before publication.")
    models, excluded_versions = [], {}
    for key, row in data["results"].items():
        if row.get("benchmark_name") != data["benchmark_name"]:
            version = row.get("benchmark_name", "missing")
            excluded_versions[version] = excluded_versions.get(version, 0) + 1
            continue
        model = {
            "id": key, "name": model_name(key), "release_date": str(row["release_date"]),
            "scaffolds": row.get("scaffolds", []),
        }
        date.fromisoformat(model["release_date"])
        for threshold in ("50", "80"):
            metric = row["metrics"][f"p{threshold}_horizon_length"]
            values = {k: positive(metric[k]) for k in ("estimate", "ci_low", "ci_high")}
            if any(v is None for v in values.values()):
                raise ValueError(f"Invalid METR interval for {key}.")
            if not values["ci_low"] <= values["estimate"] <= values["ci_high"]:
                raise ValueError(f"Unordered METR interval for {key}.")
            model[f"p{threshold}"] = values
        if model["p80"]["estimate"] > model["p50"]["estimate"]:
            raise ValueError("A higher reliability threshold cannot imply a longer horizon.")
        models.append(model)
    if len(models) < 12:
        raise ValueError("Unexpected METR coverage drop.")
    return sorted(models, key=lambda m: (m["release_date"], m["id"])), {
        "version": data["benchmark_name"],
        "long_tasks_version": data["long_tasks_version"],
        "swaa_version": data["swaa_version"],
        "published_doubling_times": data["doubling_time_in_days"],
        "reliable_range_minutes": METR_RELIABLE_MINUTES,
        "excluded_versions": excluded_versions,
    }


def parse_catalogue(body: bytes) -> tuple[list[dict], dict]:
    rows = json.loads(body)["data"]
    if not isinstance(rows, list) or len(rows) < 50:
        raise ValueError("OpenRouter catalogue is unexpectedly small.")
    seen, models, excluded = set(), [], {}

    def exclude(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate model ID: {row['id']}")
        seen.add(row["id"])
        # Do not compare free quotas, routers, or image/audio APIs as text requests.
        if ":" in row["id"] or row["id"].startswith("openrouter/"):
            exclude("routing_or_special_variant")
            continue
        if row["architecture"].get("output_modalities") != ["text"]:
            exclude("not_text_only_output")
            continue
        price = row["pricing"]
        prompt, completion = positive(price.get("prompt")), positive(price.get("completion"))
        if prompt is None or completion is None:
            exclude("missing_or_nonpositive_price")
            continue
        if positive(price.get("request"), zero=True) not in (None, 0):
            exclude("additional_request_fee")
            continue
        benchmarks = (row.get("benchmarks") or {}).get("artificial_analysis") or {}
        scores = {
            key: positive(benchmarks.get(f"{key}_index"), zero=True)
            for key in ("intelligence", "coding", "agentic")
        }
        if any(v is not None and v > 100 for v in scores.values()):
            raise ValueError("Benchmark scale changed.")
        if all(v is None for v in scores.values()):
            exclude("no_embedded_benchmark")
            continue
        tiers = []
        for tier in price.get("overrides", []):
            minimum = positive(tier.get("min_prompt_tokens"), zero=True)
            if minimum is None:
                raise ValueError("Unknown price tier schema.")
            tiers.append({
                "minimum": minimum,
                "input": positive(tier.get("prompt"), zero=True),
                "output": positive(tier.get("completion"), zero=True),
            })
        models.append({
            "id": row["id"], "name": row["name"].split(": ", 1)[-1],
            "provider": row["name"].split(": ", 1)[0],
            "input": prompt, "output": completion, "tiers": tiers,
            "context": int(row["context_length"]),
            "max_output": (row.get("top_provider") or {}).get("max_completion_tokens"),
            "scores": scores,
            "weights_link": (
                "https://huggingface.co/" + row["hugging_face_id"]
                if row.get("hugging_face_id") else None
            ),
            "url": "https://openrouter.ai/" + row["id"],
        })
    if len(models) < 20:
        raise ValueError("Too few priced, benchmarked models. Review upstream schema.")
    return sorted(models, key=lambda m: m["id"]), {
        "catalogue_rows": len(rows), "included": len(models), "excluded": excluded,
        "benchmark_version": None,
        "benchmark_note": (
            "Artificial Analysis scores as supplied by OpenRouter in this snapshot. "
            "The API does not expose the evaluation version or reasoning configuration. "
            "Indicative catalogue comparison only; not a historical series or universal IQ."
        ),
    }


def parse_sizes(body: bytes) -> list[dict]:
    selected = {
        "Llama 3.1-405B": ("Llama 3.1 405B", None),
        "DeepSeek-V3": ("DeepSeek V3", 37),
        "Qwen3-235B-A22B": ("Qwen3 235B", 22),
        "GPT-4.1": ("GPT-4.1", None),
    }
    rows = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    required = {"Model", "Parameters", "Parameters notes", "Link", "Confidence"}
    if not required.issubset(rows.fieldnames or []):
        raise ValueError("Epoch schema changed.")
    results = {}
    for row in rows:
        if row["Model"] not in selected:
            continue
        name, active = selected[row["Model"]]
        total = positive(row["Parameters"])
        # These two exact published notes were reviewed. Fail rather than retaining
        # a hardcoded active count after upstream evidence changes.
        if active and not re.search(
            rf"\b{active}(?:B| billion)\s+(?:activated|active)\b",
            row["Parameters notes"], re.I | re.S,
        ):
            raise ValueError(f"Review active parameter evidence for {name}.")
        results[row["Model"]] = {
            "name": name, "epoch_name": row["Model"], "release_date": row["Publication date"],
            "total_billions": total / 1e9 if total else None,
            "active_billions": active, "notes": row["Parameters notes"],
            "confidence": row["Confidence"], "url": row["Link"].split()[0],
            "status": "source_record" if total else "not_in_source",
        }
    if set(results) != set(selected):
        raise ValueError("A selected Epoch record is missing.")
    return [results[key] for key in selected]


def validate_bundle(bundle: dict):
    if bundle["schema_version"] != 1:
        raise ValueError("Unknown story schema.")
    if len({r["id"] for r in bundle["horizons"]}) != len(bundle["horizons"]):
        raise ValueError("Duplicate horizon ID.")
    if any(m["release_date"] > bundle["as_of"] for m in bundle["horizons"]):
        raise ValueError("Future-dated observation.")
    json.dumps(bundle, allow_nan=False)


def refresh(offline=False):
    cache = ROOT / "data/snapshots"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_file = cache / "latest.json"
    previous = json.loads(manifest_file.read_text()) if offline else {}
    evidence, source_records = {}, []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for key, config in SOURCES.items():
        if offline:
            saved = previous[key]
            body = gzip.decompress((cache / f"{saved['sha256']}.gz").read_bytes())
            retrieved = saved["retrieved_at"]
            if hashlib.sha256(body).hexdigest() != saved["sha256"]:
                raise ValueError("Cached source hash mismatch.")
        else:
            request = urllib.request.Request(
                config["url"], headers={"User-Agent": "AI-Capability-Signals/3.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
            retrieved = now
        digest = hashlib.sha256(body).hexdigest()
        snapshot = cache / f"{digest}.gz"
        if not snapshot.exists():
            snapshot.write_bytes(gzip.compress(body, mtime=0))
        evidence[key] = body
        source_records.append({
            "id": key, **config, "sha256": digest, "retrieved_at": retrieved,
        })
    horizons, benchmark = parse_metr(evidence["metr"])
    prices, coverage = parse_catalogue(evidence["openrouter"])
    sizes = parse_sizes(evidence["epoch"])
    as_of = max(s["retrieved_at"][:10] for s in source_records)
    bundle = {
        "schema_version": 1, "as_of": as_of,
        "sources": source_records, "horizons": horizons, "benchmark": benchmark,
        "prices": prices, "price_coverage": coverage, "sizes": sizes,
        "trends": build_trends(horizons),
    }
    validate_bundle(bundle)
    output = ROOT / "public/data/story.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    manifest_file.write_text(json.dumps({s["id"]: s for s in source_records}, indent=2) + "\n")
    print(
        f"Published {len(horizons)} METR measurements, {len(prices)} priced models, "
        f"{len(sizes)} size examples. Retrieved {as_of}. "
        f"Latest METR model release: {horizons[-1]['release_date']}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline", action="store_true", help="Rebuild from hash-verified raw snapshots"
    )
    refresh(parser.parse_args().offline)
