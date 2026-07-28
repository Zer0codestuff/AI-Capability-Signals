"""Run provenance, the derived reference date, and the refusals ledger.

Three ideas live here.

**Derived reference date.** A hardcoded reference date goes stale silently. Instead each source
reports the date of its most recent observation, and the run's reference date is the *minimum* of
those. A report can then never claim more freshness than its least fresh input supports, and the
gap between sources becomes a visible number rather than a footnote.

**Run manifest.** Outputs are hashed and recorded with the code version and the input digests, so
two runs can be compared byte for byte.

**Refusals ledger.** When an analysis cannot support a claim, it appends a
:class:`Refusal` instead of emitting a plausible number. The ledger is published as
``data/analysis/refusals.csv``. This is the structural fix for the previous version's habit of
capping an absurd extrapolation and publishing the cap.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT
from .netcache import sha256_file


@dataclass(frozen=True)
class Refusal:
    """A claim the pipeline declines to make, and why."""

    claim: str
    reason: str
    #: What would have to change in the data for the claim to become supportable.
    unblocked_by: str
    analysis: str


@dataclass(frozen=True)
class SourceFreshness:
    """The observation horizon of one source."""

    source_id: str
    latest_observation: str
    #: What the date means for this source. A publication date and a leaderboard publish date are
    #: not the same kind of horizon, and conflating them is how the previous version turned an
    #: ingestion timestamp into an evaluation date.
    date_semantics: str
    rows: int


@dataclass
class RunContext:
    """Accumulates everything needed to describe a run."""

    started_at: str = field(default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat())
    refusals: list[Refusal] = field(default_factory=list)
    freshness: list[SourceFreshness] = field(default_factory=list)
    input_records: list[dict[str, Any]] = field(default_factory=list)
    reference_date_override: str | None = None

    def refuse(self, claim: str, reason: str, unblocked_by: str, analysis: str) -> None:
        self.refusals.append(Refusal(claim=claim, reason=reason, unblocked_by=unblocked_by, analysis=analysis))

    def record_freshness(
        self, source_id: str, latest_observation: object, date_semantics: str, rows: int
    ) -> None:
        stamp = pd.to_datetime(latest_observation, errors="coerce")
        self.freshness.append(
            SourceFreshness(
                source_id=source_id,
                latest_observation="" if pd.isna(stamp) else stamp.date().isoformat(),
                date_semantics=date_semantics,
                rows=int(rows),
            )
        )

    @property
    def reference_date(self) -> str:
        """The run's analytical cutoff.

        Defaults to the least fresh source horizon so that cross-source comparisons are made on a
        window every source actually covers.
        """
        if self.reference_date_override:
            return self.reference_date_override
        dates = [f.latest_observation for f in self.freshness if f.latest_observation]
        if not dates:
            return datetime.now(UTC).date().isoformat()
        return min(dates)

    @property
    def max_source_date(self) -> str:
        dates = [f.latest_observation for f in self.freshness if f.latest_observation]
        return max(dates) if dates else ""

    def freshness_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([asdict(f) for f in self.freshness])
        if frame.empty:
            return pd.DataFrame(columns=["source_id", "latest_observation", "date_semantics", "rows"])
        reference = pd.Timestamp(self.reference_date)
        frame["days_behind_freshest"] = (
            pd.Timestamp(self.max_source_date) - pd.to_datetime(frame["latest_observation"])
        ).dt.days
        frame["is_binding_constraint"] = frame["latest_observation"] == reference.date().isoformat()
        return frame.sort_values("latest_observation").reset_index(drop=True)

    def refusals_frame(self) -> pd.DataFrame:
        if not self.refusals:
            return pd.DataFrame(columns=["analysis", "claim", "reason", "unblocked_by"])
        frame = pd.DataFrame([asdict(r) for r in self.refusals])
        return frame[["analysis", "claim", "reason", "unblocked_by"]].sort_values(
            ["analysis", "claim"]
        ).reset_index(drop=True)


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    path: Path,
    context: RunContext,
    outputs: Iterable[Path],
    command: str,
) -> dict[str, Any]:
    """Write a run manifest describing inputs, outputs and environment."""
    output_records = [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(outputs)
        if p.exists() and p.is_file()
    ]
    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "started_at": context.started_at,
        "command": command,
        "reference_date": context.reference_date,
        "reference_date_is_derived": context.reference_date_override is None,
        "freshest_source_date": context.max_source_date,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "refusal_count": len(context.refusals),
        "inputs": context.input_records,
        "outputs": output_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
