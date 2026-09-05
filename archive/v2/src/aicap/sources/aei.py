"""Anthropic Economic Index: observed usage composition by occupation.

This source replaces the previous version's labor-market analysis. That analysis built
"substitution pressure" and "replacement feasibility" indices out of keyword hits on O\\*NET task
text and hand-chosen coefficients, then ranked real occupations by how replaceable they were.

The Economic Index publishes, per occupation, the metrics that analysis was trying to invent:
the share of usage classified as automation versus augmentation, and a mean AI-autonomy score.
Those are vendor-computed measurements with a documented methodology, so this module reads them and
stops. No index is constructed on top.

Three limits are load-bearing and are attached to every published row:

* It measures the composition of **one vendor's observed usage**, not a sample of work or workers.
* Usage share is **not** employment impact. An occupation with high usage share may be growing.
* The two surfaces (consumer product and first-party API) differ so much for the same occupations
  that any single "share of work automated" number is an artifact of which surface was measured.
  That contrast is reported rather than averaged away.
"""

from __future__ import annotations

import io

import pandas as pd

from ..netcache import Fetcher
from ..schema import Contract

DATASET_TREE = "https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main"
RELEASE_TREE = "https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main/{release}?recursive=true"
RESOLVE_URL = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/{path}"

#: Surfaces published by the index, mapped to the filename fragment that identifies each.
SURFACES = {
    "claude_ai": "aei_claude_ai_",
    "first_party_api": "aei_1p_api_",
}

#: Metrics read from the release. Restricted to those with a documented unit and a direct
#: interpretation. Artifact-type percentages are excluded: there are 32 of them and mining them for
#: the most quotable one is exactly the multiple-comparisons trap this project tests for elsewhere.
METRICS = (
    "pct",
    "collaboration_bucket_automation_pct",
    "collaboration_bucket_augmentation_pct",
    "ai_autonomy_mean",
    "human_only_ability_pct",
    "multitasking_pct",
    "human_education_years_mean",
    "human_only_time_mean",
    "human_with_ai_time_mean",
)

CONTRACT = Contract(
    name="aei_occupations",
    required=(
        "surface",
        "release",
        "period_start",
        "period_end",
        "soc_code",
        "occupation",
        "hierarchy_level",
        "usage_share_pct",
        "automation_share_pct",
        "augmentation_share_pct",
        "ai_autonomy_mean",
    ),
    numeric=("usage_share_pct", "automation_share_pct", "augmentation_share_pct", "ai_autonomy_mean"),
    non_null=("surface", "soc_code", "occupation"),
    min_rows=100,
    unique_key=("surface", "period_start", "soc_code", "hierarchy_level"),
    notes="One row per occupation, surface and monthly period, from the latest published release.",
)

_CHUNK_ROWS = 400_000


def latest_release(fetcher: Fetcher) -> str:
    """Return the most recent ``release_YYYY_MM_DD`` directory in the dataset."""
    artifact = fetcher.fetch("anthropic_economic_index", DATASET_TREE, "aei/tree.json")
    tree = artifact.read_json()
    if not isinstance(tree, list):
        raise ValueError("anthropic_economic_index: dataset tree response is not a list.")
    releases = sorted(
        entry["path"]
        for entry in tree
        if isinstance(entry, dict)
        and entry.get("type") == "directory"
        and str(entry.get("path", "")).startswith("release_")
    )
    if not releases:
        raise ValueError("anthropic_economic_index: no release_* directories found.")
    return releases[-1]


def load(fetcher: Fetcher, release: str | None = None) -> pd.DataFrame:
    release = release or latest_release(fetcher)
    listing_artifact = fetcher.fetch(
        "anthropic_economic_index",
        RELEASE_TREE.format(release=release),
        f"aei/{release}_tree.json",
    )
    listing = listing_artifact.read_json()
    paths = [
        entry["path"]
        for entry in listing
        if isinstance(entry, dict) and entry.get("type") == "file" and str(entry["path"]).endswith(".csv")
    ]

    frames: list[pd.DataFrame] = []
    for surface, fragment in SURFACES.items():
        match = next((path for path in paths if fragment in path), None)
        if match is None:
            continue
        artifact = fetcher.fetch(
            "anthropic_economic_index",
            RESOLVE_URL.format(path=match),
            f"aei/{release}/{surface}.csv",
        )
        frames.append(_read_surface(artifact.path.read_bytes(), surface, release))

    if not frames:
        raise ValueError(
            f"anthropic_economic_index: no surface CSVs found in {release}. Layout has changed."
        )
    combined = pd.concat(frames, ignore_index=True)
    return CONTRACT.validate(combined)


def _read_surface(payload: bytes, surface: str, release: str) -> pd.DataFrame:
    """Stream one release CSV, keeping only global occupation rows for the declared metrics.

    The published files are hundreds of megabytes because they cover every geography and category.
    Filtering during a chunked read keeps memory flat and makes the selection explicit: global
    geography, the ``soc_occupation`` category, and the metric whitelist.
    """
    required = {
        "date_start",
        "date_end",
        "geo_level",
        "category_name",
        "hierarchy_level",
        "metric_id",
        "value",
        "node_name",
        "node_external_id",
    }
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(io.BytesIO(payload), chunksize=_CHUNK_ROWS):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"aei[{surface}]: missing column(s) {sorted(missing)}.")
        selected = chunk[
            (chunk["geo_level"] == "global")
            & (chunk["category_name"] == "soc_occupation")
            & (chunk["metric_id"].isin(METRICS))
        ]
        if not selected.empty:
            kept.append(selected)

    if not kept:
        raise ValueError(f"aei[{surface}]: no global soc_occupation rows matched the metric whitelist.")

    long = pd.concat(kept, ignore_index=True)
    wide = long.pivot_table(
        index=["date_start", "date_end", "hierarchy_level", "node_name", "node_external_id"],
        columns="metric_id",
        values="value",
        aggfunc="first",
    ).reset_index()

    for metric in METRICS:
        if metric not in wide.columns:
            wide[metric] = float("nan")

    frame = pd.DataFrame(
        {
            "surface": surface,
            "release": release,
            "period_start": pd.to_datetime(wide["date_start"], errors="coerce"),
            "period_end": pd.to_datetime(wide["date_end"], errors="coerce"),
            "hierarchy_level": wide["hierarchy_level"].astype(int),
            "soc_code": wide["node_external_id"].astype(str),
            "occupation": wide["node_name"].astype(str),
            "usage_share_pct": pd.to_numeric(wide["pct"], errors="coerce"),
            "automation_share_pct": pd.to_numeric(
                wide["collaboration_bucket_automation_pct"], errors="coerce"
            ),
            "augmentation_share_pct": pd.to_numeric(
                wide["collaboration_bucket_augmentation_pct"], errors="coerce"
            ),
            "ai_autonomy_mean": pd.to_numeric(wide["ai_autonomy_mean"], errors="coerce"),
            "human_only_ability_pct": pd.to_numeric(wide["human_only_ability_pct"], errors="coerce"),
            "multitasking_pct": pd.to_numeric(wide["multitasking_pct"], errors="coerce"),
            "human_education_years_mean": pd.to_numeric(
                wide["human_education_years_mean"], errors="coerce"
            ),
            "human_only_time_mean": pd.to_numeric(wide["human_only_time_mean"], errors="coerce"),
            "human_with_ai_time_mean": pd.to_numeric(wide["human_with_ai_time_mean"], errors="coerce"),
            "source_id": "anthropic_economic_index",
        }
    )
    frame["soc_major_group"] = frame["soc_code"].str.slice(0, 2)
    return frame
