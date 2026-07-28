"""LMArena leaderboard dataset.

This is the only source in the project with a genuine longitudinal panel: 200+ dated leaderboard
publications, each carrying a Bradley-Terry rating with published confidence bounds and a variance
per model. That published uncertainty is used directly, rather than manufactured by perturbing a
derived score.

Two upstream properties are handled explicitly at ingestion.

**Duplicate leaderboards.** For some ``(model, category, publish_date)`` keys the upstream file
contains several rows with different ratings and variances. Taking a maximum over them — as the
previous version did — biases affected models upward relative to unaffected ones. Here duplicates
are counted, reported to the data-quality module, and combined by inverse-variance weighting with
the between-replicate spread folded into the standard error.

**Methodology breaks.** The dataset card documents a change from Elo to Bradley-Terry (2024-01-09),
style control becoming the default (2025-05-16) and frequency re-weighting (2025-07-23). Each row
is tagged with the regime it belongs to so that any trend crossing a break can be either restricted
to one regime or flagged.
"""

from __future__ import annotations

import io

import pandas as pd

from ..netcache import Fetcher
from ..schema import Contract
from ..stats import inverse_variance_combine
from ..taxonomy import classify_family, classify_vendor, lmarena_weights_class

TREE_URL = "https://huggingface.co/api/datasets/lmarena-ai/leaderboard-dataset/tree/main?recursive=true"
RESOLVE_URL = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/{path}"

#: Arenas ingested. Restricted to Bradley-Terry text/vision/code arenas, whose schema and scale are
#: documented and mutually comparable in kind. The Agent arenas use IPS scores on a different scale
#: and are excluded rather than silently pooled.
ARENAS = ("text", "text_style_control", "vision", "webdev", "search")

#: Methodology regimes from the dataset card, as ``(start_date, label)`` in ascending order.
METHODOLOGY_REGIMES: tuple[tuple[str, str], ...] = (
    ("2023-01-01", "elo"),
    ("2024-01-09", "bradley_terry"),
    ("2025-05-16", "bradley_terry_style_control_default"),
    ("2025-07-23", "bradley_terry_frequency_reweighted"),
)

CONTRACT = Contract(
    name="lmarena_ratings",
    required=(
        "arena",
        "category",
        "model_name",
        "organization",
        "licence",
        "publish_date",
        "rating",
        "rating_se",
        "vote_count",
        "replicate_count",
        "weights_class",
        "vendor",
        "family",
        "methodology_regime",
    ),
    numeric=("rating", "rating_se", "vote_count", "replicate_count"),
    non_null=("arena", "category", "model_name", "publish_date", "rating"),
    min_rows=1000,
    unique_key=("arena", "category", "model_name", "publish_date"),
    notes="One row per model, category, arena and leaderboard publication date.",
)


def load(fetcher: Fetcher, split: str = "full", arenas: tuple[str, ...] = ARENAS) -> pd.DataFrame:
    """Load arena ratings.

    ``split="full"`` gives the whole publication history (the panel used for trend estimates);
    ``split="latest"`` gives only the most recent publication.
    """
    tree_artifact = fetcher.fetch("lmarena_leaderboard", TREE_URL, "lmarena/tree.json")
    tree = tree_artifact.read_json()
    if not isinstance(tree, list):
        raise ValueError("lmarena_leaderboard: dataset tree response is not a list.")

    available = {
        entry["path"] for entry in tree if isinstance(entry, dict) and entry.get("type") == "file"
    }

    frames: list[pd.DataFrame] = []
    for arena in arenas:
        path = f"{arena}/{split}-00000-of-00001.parquet"
        if path not in available:
            continue
        artifact = fetcher.fetch(
            "lmarena_leaderboard",
            RESOLVE_URL.format(path=path),
            f"lmarena/{arena}_{split}.parquet",
        )
        raw = pd.read_parquet(io.BytesIO(artifact.read_bytes()))
        frames.append(_normalise_arena(raw, arena))

    if not frames:
        raise ValueError(
            f"lmarena_leaderboard: no arena parquet files matched split={split!r} for {arenas}. "
            "The dataset layout has changed."
        )

    combined = pd.concat(frames, ignore_index=True)
    return CONTRACT.validate(combined)


def _normalise_arena(raw: pd.DataFrame, arena: str) -> pd.DataFrame:
    required = {"model_name", "rating", "category", "leaderboard_publish_date"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"lmarena_leaderboard[{arena}]: missing column(s) {sorted(missing)}.")

    frame = pd.DataFrame(
        {
            "arena": arena,
            "category": raw["category"].astype(str),
            "model_name": raw["model_name"].astype(str),
            "organization": raw.get("organization", pd.Series(index=raw.index, dtype=object)).astype(str),
            "licence": raw.get("license", pd.Series(index=raw.index, dtype=object)).astype(str),
            "publish_date": pd.to_datetime(raw["leaderboard_publish_date"], errors="coerce"),
            "rating": pd.to_numeric(raw["rating"], errors="coerce"),
            "rating_lower": pd.to_numeric(raw.get("rating_lower", pd.Series(index=raw.index)), errors="coerce"),
            "rating_upper": pd.to_numeric(raw.get("rating_upper", pd.Series(index=raw.index)), errors="coerce"),
            "variance": pd.to_numeric(raw.get("variance", pd.Series(index=raw.index)), errors="coerce"),
            "vote_count": pd.to_numeric(raw.get("vote_count", pd.Series(index=raw.index)), errors="coerce"),
        }
    ).dropna(subset=["model_name", "rating", "publish_date"])

    collapsed = _collapse_replicates(frame)
    collapsed["weights_class"] = [lmarena_weights_class(licence) for licence in collapsed["licence"]]
    collapsed["vendor"] = [
        classify_vendor(model, organization)
        for model, organization in zip(collapsed["model_name"], collapsed["organization"], strict=True)
    ]
    collapsed["family"] = [
        classify_family(model, organization)
        for model, organization in zip(collapsed["model_name"], collapsed["organization"], strict=True)
    ]
    collapsed["methodology_regime"] = collapsed["publish_date"].map(methodology_regime)
    collapsed["source_id"] = "lmarena_leaderboard"
    return collapsed


def _collapse_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce duplicate ``(arena, category, model, date)`` rows to one row with a combined estimate.

    The single-row case is the overwhelming majority and takes a fast path. Duplicated keys are
    combined with :func:`aicap.stats.inverse_variance_combine`, which weights by precision and adds
    the between-replicate spread to the standard error. ``replicate_count`` is retained so the
    data-quality module can report how much of the panel was affected.
    """
    key = ["arena", "category", "model_name", "publish_date"]
    counts = frame.groupby(key, dropna=False).size().rename("replicate_count")
    frame = frame.merge(counts, left_on=key, right_index=True, how="left")

    singles = frame[frame["replicate_count"] == 1].copy()
    singles["rating_se"] = singles["variance"].pow(0.5)

    duplicates = frame[frame["replicate_count"] > 1]
    collapsed_rows: list[pd.DataFrame] = [singles]
    if not duplicates.empty:
        records = []
        for keys, group in duplicates.groupby(key, dropna=False):
            point, se = inverse_variance_combine(group["rating"].tolist(), group["variance"].tolist())
            first = group.iloc[0]
            records.append(
                {
                    "arena": keys[0],
                    "category": keys[1],
                    "model_name": keys[2],
                    "publish_date": keys[3],
                    "organization": first["organization"],
                    "licence": first["licence"],
                    "rating": point,
                    "rating_lower": float(group["rating_lower"].min()),
                    "rating_upper": float(group["rating_upper"].max()),
                    "variance": float(group["variance"].mean()),
                    "vote_count": float(group["vote_count"].max()),
                    "replicate_count": int(len(group)),
                    "rating_se": se,
                }
            )
        collapsed_rows.append(pd.DataFrame.from_records(records))

    result = pd.concat(collapsed_rows, ignore_index=True)
    # Fall back to half the published interval width when a variance is absent, so that rows with
    # bounds but no variance still carry usable uncertainty instead of being dropped.
    fallback = (result["rating_upper"] - result["rating_lower"]) / (2 * 1.959963985)
    result["rating_se"] = result["rating_se"].fillna(fallback)
    return result.reset_index(drop=True)


def methodology_regime(date: object) -> str:
    """Label a publication date with the rating methodology in force at the time."""
    stamp = pd.to_datetime(date, errors="coerce")
    if pd.isna(stamp):
        return "unknown"
    label = "pre_elo"
    for start, name in METHODOLOGY_REGIMES:
        if stamp >= pd.Timestamp(start):
            label = name
    return label


def duplicate_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much of the panel arrived as duplicate leaderboard rows."""
    grouped = frame.groupby("arena", as_index=False).agg(
        rows=("replicate_count", "size"),
        rows_from_duplicate_keys=("replicate_count", lambda s: int((s > 1).sum())),
        max_replicates=("replicate_count", "max"),
    )
    grouped["share_from_duplicate_keys"] = (
        grouped["rows_from_duplicate_keys"] / grouped["rows"]
    ).round(4)
    return grouped
