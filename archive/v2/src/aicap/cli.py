"""Orchestrate ingestion, analysis and reporting.

The pipeline is deliberately linear and loud. Each stage either succeeds and writes its tables, or
raises. Silent per-source skips — the previous version's habit — are how a run could "succeed" while
missing a source that every downstream claim depended on.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis import (
    TableWriter,
    benchmark_agreement,
    calendar_control,
    compute_scaling,
    data_quality,
    disclosure,
    open_weights_lag,
    price_structure,
    swebench_progress,
    usage_composition,
)
from .config import ANALYSIS, FIGURES, INTERIM, RAW, REPORT, SOURCES, ensure_output_dirs
from .netcache import Fetcher
from .provenance import RunContext, write_manifest
from .report import figures as figure_module
from .report import render
from .sources import aei, epoch, lmarena, openrouter, swebench


@dataclass
class Bundles:
    """Ingested frames plus the run context that carries freshness and refusals."""

    epoch: pd.DataFrame
    openrouter: pd.DataFrame
    lmarena: pd.DataFrame
    swebench: pd.DataFrame
    aei: pd.DataFrame
    context: RunContext


def ingest(fetcher: Fetcher, context: RunContext, swe_limit: int | None = None) -> Bundles:
    ensure_output_dirs()
    INTERIM.mkdir(parents=True, exist_ok=True)

    print("Ingesting Epoch AI models...", flush=True)
    epoch_frame = epoch.load(fetcher)
    print("Ingesting OpenRouter catalogue...", flush=True)
    openrouter_frame = openrouter.load(fetcher)
    print("Ingesting LMArena leaderboards (this is the largest download)...", flush=True)
    lmarena_frame = lmarena.load(fetcher, split="full")
    print("Ingesting SWE-bench Verified submissions...", flush=True)
    swebench_frame = swebench.load(fetcher, limit=swe_limit)
    print("Ingesting Anthropic Economic Index...", flush=True)
    aei_frame = aei.load(fetcher)

    # Persist interim frames so a rerun of analysis alone does not re-fetch.
    epoch_frame.to_parquet(INTERIM / "epoch.parquet", index=False)
    openrouter_frame.to_parquet(INTERIM / "openrouter.parquet", index=False)
    lmarena_frame.to_parquet(INTERIM / "lmarena.parquet", index=False)
    swebench_frame.to_parquet(INTERIM / "swebench.parquet", index=False)
    aei_frame.to_parquet(INTERIM / "aei.parquet", index=False)

    context.record_freshness(
        "epoch_models",
        epoch_frame["publication_date"].max(),
        "model publication date (curated; recent years right-censored)",
        len(epoch_frame),
    )
    context.record_freshness(
        "openrouter_models",
        openrouter_frame["created_date"].max(),
        "catalogue listing created timestamp (cross-section, not a price history)",
        len(openrouter_frame),
    )
    context.record_freshness(
        "lmarena_leaderboard",
        lmarena_frame["publish_date"].max(),
        "leaderboard publication date",
        len(lmarena_frame),
    )
    context.record_freshness(
        "swebench_verified",
        swebench_frame["submission_date"].max(),
        "submission directory date",
        len(swebench_frame),
    )
    context.record_freshness(
        "anthropic_economic_index",
        aei_frame["period_end"].max(),
        "usage period end date",
        len(aei_frame),
    )
    context.input_records = list(fetcher.provenance_records())
    return Bundles(epoch_frame, openrouter_frame, lmarena_frame, swebench_frame, aei_frame, context)


def load_interim(context: RunContext) -> Bundles:
    """Reload previously ingested frames without touching the network."""
    required = {
        "epoch": INTERIM / "epoch.parquet",
        "openrouter": INTERIM / "openrouter.parquet",
        "lmarena": INTERIM / "lmarena.parquet",
        "swebench": INTERIM / "swebench.parquet",
        "aei": INTERIM / "aei.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Interim frames missing: {missing}. Run without --from-interim first."
        )
    epoch_frame = pd.read_parquet(required["epoch"])
    openrouter_frame = pd.read_parquet(required["openrouter"])
    lmarena_frame = pd.read_parquet(required["lmarena"])
    swebench_frame = pd.read_parquet(required["swebench"])
    aei_frame = pd.read_parquet(required["aei"])

    context.record_freshness(
        "epoch_models", epoch_frame["publication_date"].max(),
        "model publication date (curated; recent years right-censored)", len(epoch_frame),
    )
    context.record_freshness(
        "openrouter_models", openrouter_frame["created_date"].max(),
        "catalogue listing created timestamp (cross-section, not a price history)", len(openrouter_frame),
    )
    context.record_freshness(
        "lmarena_leaderboard", lmarena_frame["publish_date"].max(),
        "leaderboard publication date", len(lmarena_frame),
    )
    context.record_freshness(
        "swebench_verified", swebench_frame["submission_date"].max(),
        "submission directory date", len(swebench_frame),
    )
    context.record_freshness(
        "anthropic_economic_index", aei_frame["period_end"].max(),
        "usage period end date", len(aei_frame),
    )
    return Bundles(epoch_frame, openrouter_frame, lmarena_frame, swebench_frame, aei_frame, context)


def analyse(bundles: Bundles, writer: TableWriter) -> None:
    context = bundles.context
    print(f"Reference date (derived): {context.reference_date}", flush=True)

    print("Running data-quality diagnostics...", flush=True)
    for name, frame in data_quality.build(
        bundles.epoch, bundles.openrouter, bundles.lmarena, bundles.swebench, bundles.aei, context
    ).items():
        writer.write(name, frame, "Data-quality finding or diagnostic.")

    print("Analysing disclosure...", flush=True)
    for name, frame in disclosure.build(bundles.epoch, context).items():
        writer.write(name, frame, "Disclosure completeness from Epoch AI fields.")

    print("Analysing compute scaling...", flush=True)
    for name, frame in compute_scaling.build(bundles.epoch, context).items():
        writer.write(name, frame, "Disclosed training-compute frontier and backtested forecast.")

    print("Analysing price structure...", flush=True)
    for name, frame in price_structure.build(bundles.openrouter, context).items():
        writer.write(name, frame, "Catalogue price structure and price-quality relationship.")

    print("Measuring benchmark agreement...", flush=True)
    for name, frame in benchmark_agreement.build(
        bundles.openrouter, bundles.lmarena, bundles.swebench, context
    ).items():
        writer.write(name, frame, "Cross-benchmark rank agreement.")

    print("Analysing open-weight lag...", flush=True)
    for name, frame in open_weights_lag.build(bundles.lmarena, context).items():
        writer.write(name, frame, "Open-versus-closed arena frontier and catch-up lag.")

    print("Analysing usage composition...", flush=True)
    for name, frame in usage_composition.build(bundles.aei, context).items():
        writer.write(name, frame, "Observed Claude usage composition by occupation.")

    print("Running calendar negative-control tests...", flush=True)
    for name, frame in calendar_control.build(bundles.epoch, context).items():
        writer.write(name, frame, "Calendar clustering tests with multiple-comparison correction.")

    print("Summarising SWE-bench Verified history...", flush=True)
    for name, frame in swebench_progress.build(bundles.swebench, context).items():
        writer.write(name, frame, "SWE-bench Verified historical frontier.")

    writer.write(
        "source_freshness",
        context.freshness_frame(),
        "Observation horizon of each ingested source; the reference date is the minimum.",
    )
    writer.write(
        "refusals",
        context.refusals_frame(),
        "Claims the pipeline declines to make, with the reason and what would unblock them.",
    )
    writer.write(
        "sources",
        pd.DataFrame(
            [
                {
                    "source_id": source.source_id,
                    "name": source.name,
                    "url": source.url,
                    "kind": source.kind,
                    "licence": source.licence,
                    "used_for": source.used_for,
                    "caveat": source.caveat,
                }
                for source in SOURCES
            ]
        ),
        "Source registry: what each source is used for, and its known limitations.",
    )
    writer.write("analysis_manifest", writer.manifest(), "Inventory of analysis tables written this run.")


def run(
    offline: bool = False,
    refresh: bool = False,
    from_interim: bool = False,
    skip_report: bool = False,
    skip_figures: bool = False,
    swe_limit: int | None = None,
    reference_date: str | None = None,
    command: str | None = None,
) -> RunContext:
    ensure_output_dirs()
    context = RunContext(reference_date_override=reference_date)
    fetcher = Fetcher(offline=offline, refresh=refresh)

    if from_interim:
        bundles = load_interim(context)
    else:
        bundles = ingest(fetcher, context, swe_limit=swe_limit)

    writer = TableWriter(directory=ANALYSIS)
    analyse(bundles, writer)

    if not skip_figures:
        print("Rendering figures...", flush=True)
        figure_module.render_all(writer, FIGURES)

    if not skip_report:
        print("Writing report...", flush=True)
        render.write_report(writer, context, REPORT)

    write_manifest(
        REPORT / "run_manifest.json",
        context,
        outputs=list(ANALYSIS.glob("*.csv")) + list(FIGURES.glob("*.png")) + list(REPORT.glob("*")),
        command=command or " ".join(sys.argv),
    )
    print(
        f"Done. Reference date={context.reference_date}; "
        f"{len(context.refusals)} refusals; "
        f"{len(writer.written)} analysis tables.",
        flush=True,
    )
    return context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aicap",
        description="Reproducible analysis of public frontier AI capability, price and disclosure signals.",
    )
    parser.add_argument("--version", action="version", version=f"aicap {__version__}")
    parser.add_argument("--offline", action="store_true", help="Use only cached raw artifacts.")
    parser.add_argument("--refresh", action="store_true", help="Re-download every source.")
    parser.add_argument(
        "--from-interim",
        action="store_true",
        help="Skip ingestion and analyse previously written interim parquet frames.",
    )
    parser.add_argument("--skip-report", action="store_true", help="Skip Markdown/HTML report writing.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure rendering.")
    parser.add_argument("--swe-limit", type=int, default=None, help="Limit SWE-bench submissions.")
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Pin the analytical cutoff (YYYY-MM-DD). Default: derived from the least fresh source.",
    )
    args = parser.parse_args(argv)
    run(
        offline=args.offline,
        refresh=args.refresh,
        from_interim=args.from_interim,
        skip_report=args.skip_report,
        skip_figures=args.skip_figures,
        swe_limit=args.swe_limit,
        reference_date=args.reference_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
