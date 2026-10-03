"""Source registry, download and snapshot cache.

Every published number traces back to one of these files. Each download is hashed
(SHA-256) and cached gzip-compressed under ``data/snapshots`` so a run can be repeated
offline and a reviewer can check that two runs saw identical inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "snapshots"
USER_AGENT = "AI-Capability-Signals/4.0 (+https://github.com/Zer0codestuff/AI-Capability-Signals)"


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    publisher: str
    url: str
    page: str
    license: str
    description: str


SOURCES: dict[str, Source] = {
    s.id: s
    for s in [
        Source(
            "epoch_models",
            "AI Models database",
            "Epoch AI",
            "https://epoch.ai/data/all_ai_models.csv",
            "https://epoch.ai/data/ai-models",
            "CC BY 4.0",
            "Release date, parameters, training compute, training cost and openness "
            "for thousands of AI models. Many values are researcher estimates.",
        ),
        Source(
            "epoch_benchmarks",
            "Benchmarking Hub and Epoch Capabilities Index",
            "Epoch AI",
            "https://epoch.ai/data/benchmark_data.zip",
            "https://epoch.ai/benchmarks",
            "CC BY 4.0",
            "Benchmark results and the Epoch Capabilities Index (ECI), which combines "
            "dozens of benchmarks into one scale.",
        ),
        Source(
            "epoch_hardware",
            "Machine Learning Hardware database",
            "Epoch AI",
            "https://epoch.ai/data/ml_hardware.csv",
            "https://epoch.ai/data/machine-learning-hardware",
            "CC BY 4.0",
            "Specifications and launch prices of AI accelerators (GPUs, TPUs).",
        ),
        Source(
            "epoch_clusters",
            "AI Supercomputers database",
            "Epoch AI",
            "https://epoch.ai/data/gpu_clusters.csv",
            "https://epoch.ai/data/ai-supercomputers",
            "CC BY 4.0",
            "Size, owner and power of the largest AI computing clusters.",
        ),
        Source(
            "metr",
            "Time Horizon 1.1 results",
            "METR",
            "https://metr.org/assets/benchmark_results_1_1.yaml",
            "https://metr.org/time-horizons/",
            "Published research data, attribution required",
            "How long a task (measured in skilled human time) AI agents can complete "
            "with 50% and 80% success, on software, ML and cybersecurity tasks.",
        ),
        Source(
            "epoch_prices",
            "LLM inference price trends (analysis inputs)",
            "Epoch AI",
            "https://raw.githubusercontent.com/epoch-research/llm-benchmark-efficiency/main/data/{file}",
            "https://epoch.ai/data-insights/llm-inference-price-trends",
            "CC BY 4.0",
            "List prices of language models as observed between 2021 and early 2025, "
            "blended 3:1 between input and output tokens.",
        ),
        Source(
            "llm_prices",
            "LLM price history",
            "llm-prices.com (Simon Willison)",
            "https://www.llm-prices.com/historical-v1.json",
            "https://github.com/simonw/llm-prices",
            "No license declared, factual list prices, attributed",
            "Vendor list prices kept by hand, including the price a model had before a later price change.",
        ),
        Source(
            "models_dev",
            "Model catalogue",
            "models.dev",
            "https://models.dev/api.json",
            "https://models.dev",
            "MIT",
            "Release dates and current list prices of each provider's own API per provider.",
        ),
        Source(
            "openrouter",
            "Model catalogue API",
            "OpenRouter",
            "https://openrouter.ai/api/v1/models",
            "https://openrouter.ai/models",
            "Public API, provider terms apply",
            "Current list prices of models offered through OpenRouter.",
        ),
    ]
}


def _cache_path(digest: str) -> Path:
    return SNAPSHOTS / f"{digest}.gz"


def fetch(url: str, *, retries: int = 3, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retried and re-raised below
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Download failed: {url}: {last}")


class Snapshot:
    """Downloads (or replays) every source once per run and records provenance."""

    def __init__(self, offline: bool = False):
        self.offline = offline
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        self.manifest_file = SNAPSHOTS / "latest.json"
        self.previous = json.loads(self.manifest_file.read_text()) if self.manifest_file.exists() else {}
        self.records: dict[str, dict] = {}

    def get(self, key: str, url: str, source_id: str) -> bytes:
        if self.offline:
            saved = self.previous.get(key)
            if not saved:
                raise RuntimeError(f"No cached snapshot for {key}. Run an online refresh first.")
            body = gzip.decompress(_cache_path(saved["sha256"]).read_bytes())
            if hashlib.sha256(body).hexdigest() != saved["sha256"]:
                raise RuntimeError(f"Cached snapshot hash mismatch for {key}.")
            retrieved = saved["retrieved_at"]
        else:
            body = fetch(url)
            retrieved = datetime.now(UTC).isoformat(timespec="seconds")
        digest = hashlib.sha256(body).hexdigest()
        path = _cache_path(digest)
        if not path.exists():
            path.write_bytes(gzip.compress(body, mtime=0))
        self.records[key] = {
            "key": key,
            "source": source_id,
            "url": url,
            "sha256": digest,
            "bytes": len(body),
            "retrieved_at": retrieved,
        }
        return body

    def save_manifest(self) -> None:
        self.manifest_file.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")


def unzip(body: bytes) -> dict[str, bytes]:
    """Return every file in a zip archive keyed by its base name."""
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        return {
            name.rsplit("/", 1)[-1]: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
