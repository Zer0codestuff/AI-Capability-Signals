"""SWE-bench Verified public submissions.

A fixed 500-instance benchmark with dated, voluntary submissions. It is the project's only source
of task-completion rates on a stable instance set, which makes it valuable and also easy to misuse:

* A submission measures **an agent scaffold plus a model**, not a model. Two submissions using the
  same model can differ by tens of points. Scores are therefore attributed to a system, and the
  model tag is carried as metadata rather than treated as the unit of analysis.
* Submission is voluntary, so the frontier is a lower bound on what exists.
* The directory can go long periods without new entries. Staleness is measured at ingestion and
  reported, so no analysis can present a stale leaderboard as a current signal.
"""

from __future__ import annotations

import re

import pandas as pd

from ..netcache import Fetcher
from ..schema import Contract
from ..taxonomy import classify_family, classify_vendor

INDEX_URL = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/verified"
RAW_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"

#: SWE-bench Verified is a fixed set of 500 human-validated instances.
TOTAL_INSTANCES = 500

CONTRACT = Contract(
    name="swebench_verified",
    required=(
        "submission",
        "submission_date",
        "system_label",
        "model_tag",
        "vendor",
        "family",
        "resolved",
        "total",
        "resolve_rate_pct",
        "uses_open_weights_model",
    ),
    numeric=("resolved", "total", "resolve_rate_pct"),
    non_null=("submission", "resolved"),
    min_rows=20,
    unique_key=("submission",),
    notes="One row per public submission directory.",
)

_DIR_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<label>.+)$")


def load(fetcher: Fetcher, limit: int | None = None) -> pd.DataFrame:
    index_artifact = fetcher.fetch("swebench_verified", INDEX_URL, "swebench/verified_index.json")
    index = index_artifact.read_json()
    if not isinstance(index, list):
        raise ValueError(
            "swebench_verified: GitHub contents API did not return a list. "
            "This usually means a rate limit response; retry later or supply a token."
        )

    directories = sorted(
        (entry["name"] for entry in index if isinstance(entry, dict) and entry.get("type") == "dir"),
    )
    if limit is not None:
        directories = directories[-limit:]

    records = []
    for name in directories:
        parsed = _DIR_PATTERN.match(name)
        if not parsed:
            # Directory naming is the only date source. An unparseable name is reported rather than
            # given a guessed date.
            records.append(_unparsed_record(name))
            continue
        results_artifact = fetcher.fetch(
            "swebench_verified",
            f"{RAW_BASE}/{name}/results/results.json",
            f"swebench/{name}/results.json",
        )
        results = results_artifact.read_json()
        resolved = results.get("resolved")
        if not isinstance(resolved, list):
            raise ValueError(
                f"swebench_verified[{name}]: results.json has no 'resolved' list; upstream format changed."
            )
        label = parsed.group("label")
        records.append(
            {
                "submission": name,
                "submission_date": pd.to_datetime(parsed.group("date"), format="%Y%m%d"),
                "system_label": label,
                "model_tag": _model_tag(label),
                "resolved": len(resolved),
                "total": TOTAL_INSTANCES,
                "parse_status": "ok",
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["resolve_rate_pct"] = frame["resolved"] / frame["total"] * 100.0
    frame["vendor"] = [classify_vendor(tag) for tag in frame["model_tag"]]
    frame["family"] = [classify_family(tag) for tag in frame["model_tag"]]
    frame["uses_open_weights_model"] = frame["system_label"].map(_looks_open_weights)
    frame["source_id"] = "swebench_verified"
    frame["source_url"] = (
        "https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/" + frame["submission"]
    )
    return CONTRACT.validate(frame.sort_values("submission").reset_index(drop=True))


def _unparsed_record(name: str) -> dict[str, object]:
    return {
        "submission": name,
        "submission_date": pd.NaT,
        "system_label": name,
        "model_tag": "",
        "resolved": float("nan"),
        "total": TOTAL_INSTANCES,
        "parse_status": "unparsed_directory_name",
    }


def _model_tag(label: str) -> str:
    """Best-effort model identifier from a submission directory name.

    Directory names are author-chosen (``sweagent_claude3opus``, ``openhands_claude-opus-4-5``), so
    this returns the raw remainder after known scaffold prefixes and leaves interpretation to the
    reader. It is used for grouping in descriptive tables only, never to join a benchmark score to
    a price.
    """
    text = label
    for scaffold in (
        "sweagent",
        "swe-agent",
        "openhands",
        "livesweagent",
        "rag",
        "agentless",
        "moatless",
        "autocoderover",
        "sonar-foundation-agent",
        "epam-ai-run",
        "nfactorial",
        "gru",
        "isoform",
        "blackbox",
        "codestory",
        "amazon-q",
        "honeycomb",
        "lingma",
        "marscode",
        "aider",
        "devlo",
        "bytedance",
        "globant",
        "emergent",
        "refact",
        "solver",
        "tools",
        "composio",
    ):
        text = re.sub(rf"^{re.escape(scaffold)}[-_]?", "", text, flags=re.IGNORECASE)
    return text.strip("-_")


def _looks_open_weights(label: str) -> bool | None:
    """Flag submissions whose label names a known open-weight model family.

    Returns ``None`` when the label gives no signal. This is a weak, name-based signal and is used
    only for a descriptive split with the uncertainty stated; it never feeds an estimate that is
    reported with an interval.
    """
    text = label.lower()
    open_markers = ("llama", "qwen", "deepseek", "mistral", "glm", "kimi", "devstral", "olmo", "frog")
    closed_markers = ("claude", "gpt", "gemini", "o1", "o3", "o4", "grok", "sonnet", "opus")
    if any(marker in text for marker in open_markers):
        return True
    if any(marker in text for marker in closed_markers):
        return False
    return None


def staleness_days(frame: pd.DataFrame, reference_date: str) -> int:
    """Days between the most recent submission and the run's reference date."""
    latest = pd.to_datetime(frame["submission_date"], errors="coerce").max()
    if pd.isna(latest):
        return -1
    return int((pd.Timestamp(reference_date) - latest).days)
