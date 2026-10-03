"""Guards that keep doubtful source values out of records, trends and headlines.

The sources are large hand-built databases. A value can be transcribed correctly and
still not mean what a chart assumes: a size a system could handle rather than a model
that was trained, a rumour, a number typed in the wrong unit. Three rules apply before
any value can set a record, enter a trend or be quoted in the text:

1. Vetted only. In the Epoch AI models database a value counts only if the model is in
   the curated "notable" set and the row is rated Confident or Likely. Speculative and
   unrated rows are drawn as background dots and nothing else.
2. No unit slips. A parameter count whose own note quotes a size a thousand or a
   million times larger or smaller is treated as a typing error and set aside.
3. No unreviewed leaps. A recent value that beats the previous record more than tenfold
   is held back until a person adds it to ``reviewed.json``.

Everything set aside is listed in the published data file.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

RATED = {"Confident", "Likely"}
MAX_LEAP = 10.0
RECENT_DAYS = 3 * 365
REVIEWED = {
    (item["chart"], item["name"])
    for item in json.loads((Path(__file__).parent / "reviewed.json").read_text())
}
UNITS = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}
SIZE = re.compile(r"(\d[\d,]*\.?\d*)\s?(trillion|billion|million|bn|T|B|M)\b")


class Audit:
    def __init__(self) -> None:
        self.flags: list[dict] = []

    def flag(self, kind: str, chart: str, name: str, detail: str) -> None:
        entry = {"kind": kind, "chart": chart, "name": name, "detail": detail}
        if entry not in self.flags:
            self.flags.append(entry)


def unit_slip(value: float, note: str) -> float | None:
    """Size quoted in the note when the value looks like the same number in the wrong unit."""
    quoted = [float(digits.replace(",", "")) * UNITS[unit.lower()] for digits, unit in SIZE.findall(note)]
    if not quoted or any(0.67 <= size / value <= 1.5 for size in quoted):
        return None
    for size in quoted:
        ratio = size / value
        if any(abs(ratio / scale - 1) < 0.02 for scale in (1e3, 1e-3, 1e6, 1e-6)):
            return size
    return None


def status(row: dict, field: str) -> str | None:
    """Why a database row may not be used for `field`, or None when it may."""
    if field == "params" and row.get("slip"):
        return "unit"
    if not row["notable"]:
        return "unvetted"
    if row["confidence"] == "Speculative":
        return "speculative"
    if row["confidence"] not in RATED:
        return "unrated"
    return None


def records(
    points: list[tuple[date, float, object]],
    *,
    chart: str,
    today: date,
    audit: Audit,
    lowest: bool = False,
) -> tuple[list[tuple[date, float, object]], set[object]]:
    """Running records, holding back recent unreviewed leaps. Returns records and held names."""
    best: float | None = None
    kept = []
    held: set[object] = set()
    recent = today - timedelta(days=RECENT_DAYS)
    for when, value, name in sorted(points, key=lambda p: (p[0], p[1] if lowest else -p[1])):
        if best is not None and not (value < best if lowest else value > best):
            continue
        leap = (best / value if lowest else value / best) if best else 1.0
        if leap > MAX_LEAP and when >= recent and (chart, str(name)) not in REVIEWED:
            held.add(name)
            audit.flag(
                "held",
                chart,
                str(name),
                f"beats the previous record {leap:.0f} times over, so it is held back until reviewed",
            )
            continue
        best = value
        kept.append((when, value, name))
    return kept, held
