"""Paths, source registry and the snapshot-date policy.

Snapshot policy
---------------
The previous version of this project hardcoded ``REFERENCE_DATE = "2026-05-15"``. Two months
later the published report still described superseded models as current, because nothing in the
code could notice that the constant had gone stale.

Here the reference date is *derived from the data*: it is the freshest observation horizon over
the ingested sources (see :func:`aicap.provenance.RunContext.reference_date`). Sources that lag
behind are reported in the freshness table and trigger refusals in the analyses that depend on
them when they are too stale for present-tense claims. A run can still pin a date explicitly with
``--reference-date`` for reproducing an older snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
ANALYSIS = ROOT / "data" / "analysis"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
DOCS = ROOT / "docs"

#: Directories written by a normal run, created on demand.
OUTPUT_DIRS = (RAW, INTERIM, ANALYSIS, FIGURES, REPORT)

#: Random seed for every stochastic procedure in the package. Bootstrap and permutation
#: routines take an explicit ``seed`` argument that defaults to this value, so re-running the
#: pipeline reproduces intervals exactly.
SEED = 20260728

#: Nominal coverage for every interval the project publishes.
CONFIDENCE_LEVEL = 0.95

#: Bootstrap resamples. 10_000 keeps the Monte Carlo error on a 95% percentile interval
#: endpoint small relative to the reported precision (2 significant figures).
BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class Source:
    """A public data source, its access URL and the terms under which it is redistributed."""

    source_id: str
    name: str
    url: str
    kind: str
    licence: str
    #: What this project uses the source for. Deliberately narrow: if a source is only
    #: trustworthy for one purpose, the registry says so.
    used_for: str
    #: Known limitations that constrain interpretation, carried into the published source table.
    caveat: str


SOURCES: tuple[Source, ...] = (
    Source(
        source_id="epoch_models",
        name="Epoch AI notable AI models",
        url="https://epoch.ai/data/all_ai_models.csv",
        kind="curated_dataset",
        licence="CC-BY (see THIRD_PARTY_DATA.md)",
        used_for=(
            "Model publication dates, training compute, parameters, accessibility class, "
            "frontier flag and disclosure completeness."
        ),
        caveat=(
            "Curated with a lag, so recent years are right-censored. Training compute is "
            "disclosed for a minority of models, and disclosure is not random."
        ),
    ),
    Source(
        source_id="openrouter_models",
        name="OpenRouter public models API",
        url="https://openrouter.ai/api/v1/models",
        kind="commercial_api_catalogue",
        licence="Public API response; terms at openrouter.ai",
        used_for=(
            "Current listed prices including prompt-length tiers, context windows, modality, "
            "Hugging Face weight identifiers, and vendor-published benchmark indices."
        ),
        caveat=(
            "A catalogue of currently listed models. It is a cross-section, not a price "
            "history: withdrawn models are absent and listed prices are not invoices."
        ),
    ),
    Source(
        source_id="lmarena_leaderboard",
        name="LMArena leaderboard dataset",
        url="https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
        kind="public_dataset",
        licence="See dataset card",
        used_for=(
            "Bradley-Terry ratings with published confidence bounds and variances, per "
            "category, across the full history of published leaderboards."
        ),
        caveat=(
            "Human preference, not task success. The rating system changed in Jan 2024 "
            "(Elo to Bradley-Terry), May 2025 (style control default) and Jul 2025 "
            "(frequency re-weighting), so long spans cross methodology breaks."
        ),
    ),
    Source(
        source_id="swebench_verified",
        name="SWE-bench Verified public submissions",
        url="https://api.github.com/repos/swe-bench/experiments/contents/evaluation/verified",
        kind="public_leaderboard",
        licence="MIT (repository)",
        used_for="Dated, agent-level resolve rates on a fixed 500-instance benchmark.",
        caveat=(
            "Scores describe an agent scaffold plus a model, not a model alone. Submission "
            "is voluntary and the directory has not received new entries recently."
        ),
    ),
    Source(
        source_id="anthropic_economic_index",
        name="Anthropic Economic Index",
        url="https://huggingface.co/datasets/Anthropic/EconomicIndex",
        kind="public_dataset",
        licence="CC-BY",
        used_for=(
            "Vendor-published metrics on observed usage composition by occupation: "
            "automation/augmentation split and mean AI autonomy."
        ),
        caveat=(
            "Measures the composition of one vendor's observed usage. It is not a sample of "
            "the workforce and carries no information about employment outcomes."
        ),
    ),
)

SOURCES_BY_ID = {source.source_id: source for source in SOURCES}


def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
