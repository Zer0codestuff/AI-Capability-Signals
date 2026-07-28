"""Schema contracts for ingested data.

The previous version parsed every source with chained ``.get()`` calls and silent defaults, so a
renamed upstream column produced an empty column and a green run. Each source module here
declares a :class:`Contract` and calls :meth:`Contract.validate` immediately after parsing, which
turns upstream drift into a loud, specific failure at ingestion time rather than a quiet
distortion three layers downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SchemaError(RuntimeError):
    """Raised when ingested data does not satisfy its declared contract."""


@dataclass(frozen=True)
class Contract:
    """A minimal, checkable description of a parsed table.

    Parameters
    ----------
    name:
        Table name, used in error messages.
    required:
        Columns that must exist. Missing columns raise.
    numeric:
        Columns that must be numeric dtype after parsing.
    non_null:
        Columns that must be fully populated. Use for keys, not for measurements.
    min_rows:
        Lower bound on row count. Guards against an upstream endpoint that starts returning
        an empty payload with a valid schema.
    unique_key:
        Columns that must jointly identify a row. Violations raise, because a silent duplicate
        key is how snapshot pseudo-replication enters an analysis.
    """

    name: str
    required: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    non_null: tuple[str, ...] = ()
    min_rows: int = 1
    unique_key: tuple[str, ...] = ()
    notes: str = ""

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.required if column not in frame.columns]
        if missing:
            raise SchemaError(
                f"{self.name}: upstream schema drift, missing column(s) {missing}. "
                f"Present columns: {sorted(frame.columns)}"
            )
        if len(frame) < self.min_rows:
            raise SchemaError(
                f"{self.name}: expected at least {self.min_rows} rows, parsed {len(frame)}. "
                "The upstream payload is probably empty or the parser matched nothing."
            )
        for column in self.numeric:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise SchemaError(
                    f"{self.name}: column {column!r} must be numeric, got {frame[column].dtype}."
                )
        for column in self.non_null:
            null_count = int(frame[column].isna().sum())
            if null_count:
                raise SchemaError(
                    f"{self.name}: column {column!r} must be fully populated, found "
                    f"{null_count} null value(s)."
                )
        if self.unique_key:
            duplicated = int(frame.duplicated(subset=list(self.unique_key)).sum())
            if duplicated:
                raise SchemaError(
                    f"{self.name}: {duplicated} row(s) duplicate the key {self.unique_key}. "
                    "Resolve duplicates explicitly before validating, so the resolution rule is "
                    "visible in the code rather than implicit in a later groupby."
                )
        return frame
