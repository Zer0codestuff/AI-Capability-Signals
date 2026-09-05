"""Estimators, one module per question.

Rules every module in this package follows
-----------------------------------------
1. A module answers one question and says so in its docstring, including what would falsify its
   answer.
2. Numbers are either direct measurements from a source field, or the output of a named estimator
   in :mod:`aicap.stats` accompanied by an interval.
3. When the data cannot support the question, the module calls
   :meth:`aicap.provenance.RunContext.refuse` and emits no number. Capping an implausible
   extrapolation and publishing the cap is not an option.
4. No module builds a composite score across incommensurable metrics. Whether such a score would
   even be coherent is itself measured, in :mod:`aicap.analysis.benchmark_agreement`.
5. Modules receive ingested frames and a :class:`aicap.provenance.RunContext`; they never fetch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import ANALYSIS


@dataclass
class TableWriter:
    """Collects analysis tables, writes them as CSV, and records a manifest.

    Float formatting is centralised so that published tables do not carry 15 significant digits of
    spurious precision. Values are rounded at write time only; the in-memory frames keep full
    precision for downstream use.
    """

    directory: Path = ANALYSIS
    written: dict[str, Path] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    def write(self, name: str, frame: pd.DataFrame, description: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}.csv"
        rounded = frame.copy()
        for column in rounded.columns:
            if pd.api.types.is_float_dtype(rounded[column]):
                rounded[column] = rounded[column].round(6)
        rounded.to_csv(path, index=False)
        self.written[name] = path
        self.descriptions[name] = description
        self.row_counts[name] = int(len(frame))
        return path

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "table": name,
                    "rows": self.row_counts[name],
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "description": self.descriptions[name],
                }
                for name, path in sorted(self.written.items())
            ]
        )

    def read(self, name: str) -> pd.DataFrame:
        """Read a previously written table. Used by the report layer, which must not recompute."""
        return pd.read_csv(self.directory / f"{name}.csv")
