"""Epoch AI notable-models dataset.

What this source is good for: publication dates, training compute, parameter counts, an
authoritative accessibility class, and a curated frontier flag.

What it is not good for: counting recent releases. The dataset is curated with a lag, so the most
recent year or two are right-censored and a naive count-by-year shows a spurious decline. The
censoring boundary is estimated here, at ingestion, and carried downstream as
:func:`censoring_boundary` so no analysis can forget it.
"""

from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd

from ..netcache import Fetcher
from ..schema import Contract
from ..taxonomy import classify_family, classify_vendor, epoch_weights_class

URL = "https://epoch.ai/data/all_ai_models.csv"

#: Fields whose presence defines "disclosure" for this project, chosen before looking at the
#: results and fixed here so the disclosure measure cannot be tuned after the fact.
DISCLOSURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("parameters", "Parameter count"),
    ("training_compute_flop", "Training compute (FLOP)"),
    ("training_tokens", "Training dataset size"),
    ("training_hardware", "Training hardware"),
    ("accessibility_raw", "Accessibility statement"),
)

CONTRACT = Contract(
    name="epoch_models",
    required=(
        "model",
        "organization",
        "publication_date",
        "domain",
        "parameters",
        "training_compute_flop",
        "training_tokens",
        "training_hardware",
        "accessibility_raw",
        "open_weights_raw",
        "frontier_flag",
        "weights_class",
        "vendor",
        "family",
        "country",
    ),
    numeric=("parameters", "training_compute_flop", "training_tokens"),
    non_null=("model", "weights_class"),
    min_rows=1000,
    notes="One row per notable model as curated by Epoch AI.",
)

_COLUMN_MAP = {
    "Model": "model",
    "Organization": "organization",
    "Publication date": "publication_date",
    "Domain": "domain",
    "Task": "task",
    "Parameters": "parameters",
    "Parameters notes": "parameters_notes",
    "Training compute (FLOP)": "training_compute_flop",
    "Training dataset size (total)": "training_tokens",
    "Training hardware": "training_hardware",
    "Training compute cost (2023 USD)": "training_cost_2023_usd",
    "Model accessibility": "accessibility_raw",
    "Open model weights?": "open_weights_raw",
    "Frontier model": "frontier_flag_raw",
    "Country (of organization)": "country",
    "Reference": "reference",
    "Link": "source_link",
    "Notability criteria": "notability",
    "Confidence": "confidence",
}


def load(fetcher: Fetcher) -> pd.DataFrame:
    artifact = fetcher.fetch("epoch_models", URL, "epoch/all_ai_models.csv")
    raw = pd.read_csv(io.BytesIO(artifact.read_bytes()), low_memory=False)

    missing = [column for column in _COLUMN_MAP if column not in raw.columns]
    if missing:
        raise ValueError(
            f"epoch_models: upstream CSV is missing expected column(s) {missing}. "
            f"Available: {sorted(raw.columns)[:40]}..."
        )

    frame = raw[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP).copy()

    for column in ("parameters", "training_compute_flop", "training_tokens", "training_cost_2023_usd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["publication_date"] = pd.to_datetime(frame["publication_date"], errors="coerce", format="mixed")
    frame["publication_year"] = frame["publication_date"].dt.year
    # Epoch mixes "2026", "2026-07" and "2026-07-24" in one column. Precision is preserved so
    # month-level analyses can exclude year-only rows rather than treating them as 1 January.
    frame["date_precision"] = raw["Publication date"].map(_date_precision)

    frame["frontier_flag"] = frame["frontier_flag_raw"].astype(str).str.strip().str.lower().eq("true")
    frame["weights_class"] = [
        epoch_weights_class(accessibility, flag)
        for accessibility, flag in zip(frame["accessibility_raw"], frame["open_weights_raw"], strict=True)
    ]
    frame["vendor"] = [
        classify_vendor(model, organization)
        for model, organization in zip(frame["model"], frame["organization"], strict=True)
    ]
    frame["family"] = [
        classify_family(model, organization)
        for model, organization in zip(frame["model"], frame["organization"], strict=True)
    ]
    frame["active_parameters"] = frame["parameters_notes"].map(_active_parameters)

    for field_name, _label in DISCLOSURE_FIELDS:
        frame[f"discloses_{field_name}"] = frame[field_name].notna() & (
            frame[field_name].astype(str).str.strip() != ""
        )

    frame["source_id"] = "epoch_models"
    return CONTRACT.validate(frame.reset_index(drop=True))


def _date_precision(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return "missing"
    if re.fullmatch(r"\d{4}", text):
        return "year"
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return "month"
    return "day"


def _active_parameters(notes: Any) -> float:
    """Extract an active-parameter count from Epoch's free-text parameter notes.

    Mixture-of-experts models report total and active parameters differently across rows, so this
    reads the explicit "N B active" phrasing only, and returns NaN otherwise rather than guessing.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*B\s*active", str(notes or ""), flags=re.IGNORECASE)
    return float(match.group(1)) * 1e9 if match else np.nan


def censoring_boundary(frame: pd.DataFrame) -> int:
    """Return the earliest year that may be depressed by curation lag.

    Rule: walk back from the most recent year while each year holds fewer models than the year
    before it. The first year that breaks the descending run is treated as complete, and everything
    after it is treated as provisional.

    The rule deliberately cannot distinguish curation lag from a genuine slowdown, and that is why
    it is used to *suppress* claims rather than to make one: analyses cut at this boundary and the
    pipeline records a refusal explaining that the boundary years are unusable either way. Guessing
    which explanation applies would be the same mistake as publishing the raw counts as a trend.
    """
    counts = frame.dropna(subset=["publication_year"]).groupby("publication_year").size().sort_index()
    if counts.empty:
        return 9999
    years = [int(year) for year in counts.index]
    censored_from = years[-1] + 1
    for index in range(len(years) - 1, 0, -1):
        if counts.iloc[index] < counts.iloc[index - 1]:
            censored_from = years[index]
        else:
            break
    return censored_from
