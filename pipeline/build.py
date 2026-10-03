"""Build ``public/data/signals.json``, the only data file the website reads.

Run ``python -m pipeline.build`` to download fresh sources, or add ``--offline`` to
replay the last cached snapshot. The output is replaced only after every chapter has
been computed and checked.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from . import audit, prices, stats
from .audit import Audit
from .sources import ROOT, SOURCES, Snapshot, unzip

OUTPUT = ROOT / "public" / "data" / "signals.json"
CORRECTIONS = json.loads((Path(__file__).parent / "corrections.json").read_text())
PROJECTION_YEARS = 2
# GPT-4 is the first model in the index that was state of the art when it came out.
ECI_START = date(2023, 3, 1)
# (file, whether each row carries the day its price was recorded)
EPOCH_PRICE_FILES = [
    ("epoch_ai_price_data_not_in_aa_with_benchmarks.csv", True),
    ("aa_data_with_math5.csv", False),
]
METR_NAMES = {
    "gpt2": "GPT-2",
    "davinci_002": "GPT-3 (davinci-002)",
    "gpt_3_5_turbo_instruct": "GPT-3.5 Turbo Instruct",
    "gpt_4": "GPT-4",
    "gpt_4_1106": "GPT-4 Turbo (Nov 2023)",
    "gpt_4_turbo": "GPT-4 Turbo",
    "gpt_4o": "GPT-4o",
    "o1_preview": "o1-preview",
    "o1": "o1",
    "o3": "o3",
    "gpt_5_2025_08_07": "GPT-5",
    "claude_3_5_sonnet_20240620": "Claude 3.5 Sonnet (Jun 2024)",
    "claude_3_5_sonnet_20241022": "Claude 3.5 Sonnet (Oct 2024)",
    "claude_4_opus": "Claude Opus 4",
    "claude_4_1_opus": "Claude Opus 4.1",
    "claude_mythos_preview_early": "Claude Mythos Preview",
}
MONTHS = {
    name: index + 1
    for index, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
NAME_MONTH = re.compile(r"\((\w{3})\w*\.? (\d{4})\)")
# A projection is drawn only when the fit has enough points and, in the backtest, did
# better than assuming no further change.
MIN_PROJECTION_POINTS = 10


# ---------------------------------------------------------------- small helpers


def rows(body: bytes) -> list[dict[str, str]]:
    csv.field_size_limit(10**9)
    return list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))


def number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def day(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def sig(value: float, digits: int = 4) -> float:
    if value == 0 or not math.isfinite(value):
        return 0.0
    return float(f"{value:.{digits}g}")


def short(value: float) -> str:
    """A large number in words a reader can check: 174T, 366M, 2.1B."""
    for size, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= size:
            return f"{value / size:.3g}{suffix}"
    return f"{value:.3g}"


WHY = {
    "unvetted": "comes from outside the curated set",
    "speculative": "is a speculative estimate",
    "unrated": "has no confidence rating",
    "unit": "disagrees with its own note",
    "held": "awaits review",
}


def months_between(start: date, end: date) -> float:
    return (end - start).days / 30.4375


def add_years(start: date, years: float) -> date:
    return start + timedelta(days=round(365.25 * years))


def point(when: date, value: float, name: str, org: str = "", group: str = "", **extra) -> dict:
    result: dict = {"d": when.isoformat(), "v": sig(value), "n": name}
    if org:
        result["o"] = org
    if group:
        result["g"] = group
    result.update({key: item for key, item in extra.items() if item is not None})
    return result


def country_group(raw: str) -> str:
    first = raw.split(",")[0].strip()
    return {"United States of America": "us", "China": "china"}.get(first, "other")


# ---------------------------------------------------------------- trend output


def interval(values: tuple[float, float, float]) -> dict:
    return {"v": sig(values[0]), "lo": sig(values[1]), "hi": sig(values[2])}


def describe_line(line: stats.Line, log: bool) -> dict:
    low, high = line.slope_interval()
    slope = line.fit.slope
    result = {
        "n": line.n,
        "rate": interval((stats.rate(slope, log), stats.rate(low, log), stats.rate(high, log))),
        "r2": round(line.fit.r2, 3),
    }
    if log:
        centre = stats.doubling_months(slope)
        if centre is not None and low * high > 0:
            ends = sorted([stats.doubling_months(low), stats.doubling_months(high)])
            result["doubling"] = interval((centre, ends[0], ends[1]))
        elif centre is not None:
            result["doubling"] = {"v": sig(centre)}
    return result


def trend_json(
    points: list[tuple[date, float]],
    *,
    log: bool,
    today: date,
    seed: int,
    levels: list[tuple[str, str, float]] | None = None,
) -> dict | None:
    """Fit, projection band, milestones and backtest for one series."""
    ordered = sorted(points)
    if len(ordered) < 4:
        return None
    try:
        trend = stats.Trend(ordered, log=log, seed=seed)
    except ValueError:
        return None
    last = ordered[-1][0]
    if trend.basis == "recent" and trend.shape.split is not None:
        start = next(d for d, _ in ordered if stats.year_fraction(d) - stats.ORIGIN > trend.shape.split)
    else:
        start = ordered[0][0]
    result = describe_line(trend.line, log)
    result.update(
        {
            "log": log,
            "window": {"from": ordered[0][0].isoformat(), "to": last.isoformat(), "n": len(ordered)},
            "basis": trend.basis,
            "whole": describe_line(trend.full, log),
            "line": [
                {"d": start.isoformat(), "v": sig(trend.predict(start)[0])},
                {"d": last.isoformat(), "v": sig(trend.predict(last)[0])},
            ],
        }
    )
    shape = trend.shape
    result["shape"] = {"verdict": shape.verdict}
    if shape.early and shape.late and shape.split is not None:
        result["shape"].update(
            {
                "split": stats.from_year_fraction(shape.split + stats.ORIGIN).isoformat(),
                "early": describe_line(shape.early, log),
                "late": describe_line(shape.late, log),
            }
        )
    check = stats.backtest(ordered, log=log, seed=seed + 100)
    result["backtest"] = (
        {key: (round(item, 4) if isinstance(item, float) else item) for key, item in check.items()}
        if check
        else None
    )
    band = []
    milestones = []
    reliable = bool(check and check["typical_error"] < check["naive_error"])
    result["projectable"] = reliable and trend.line.n >= MIN_PROJECTION_POINTS
    if result["projectable"]:
        end = add_years(today, PROJECTION_YEARS)
        cursor = last
        while cursor <= end:
            mid, low, high = trend.predict(cursor)
            band.append({"d": cursor.isoformat(), "v": sig(mid), "lo": sig(low), "hi": sig(high)})
            cursor += timedelta(days=30)
        for identifier, label, level in levels or []:
            crossing = trend.crossing(level)
            if crossing and last < crossing[0] <= add_years(today, 8):
                milestones.append(
                    {
                        "id": identifier,
                        "label": label,
                        "v": level,
                        "d": crossing[0].isoformat(),
                        "lo": min(crossing[1], crossing[2]).isoformat(),
                        "hi": max(crossing[1], crossing[2]).isoformat(),
                    }
                )
    result["band"] = band
    result["milestones"] = milestones
    return result


def step_series(identifier: str, label: str, records: list[tuple[date, float, object]]) -> dict:
    return {
        "id": identifier,
        "label": label,
        "kind": "step",
        "points": [{"d": d.isoformat(), "v": sig(v), "n": str(name)} for d, v, name in records],
    }


# ---------------------------------------------------------------- loading


def load(snapshot: Snapshot) -> dict:
    def get(key: str, source: str, url: str | None = None) -> bytes:
        return snapshot.get(key, url or SOURCES[source].url, source)

    benchmarks = unzip(get("epoch_benchmarks", "epoch_benchmarks"))
    price_url = SOURCES["epoch_prices"].url
    return {
        "models": rows(get("epoch_models", "epoch_models")),
        "eci": rows(benchmarks["eci_scores.csv"]),
        "hardware": rows(get("epoch_hardware", "epoch_hardware")),
        "clusters": rows(get("epoch_clusters", "epoch_clusters")),
        "metr": yaml.safe_load(get("metr", "metr")),
        "price_files": [
            (get(f"epoch_prices/{name}", "epoch_prices", price_url.format(file=name)), dated)
            for name, dated in EPOCH_PRICE_FILES
        ],
        "llm_prices": json.loads(get("llm_prices", "llm_prices")),
        "models_dev": json.loads(get("models_dev", "models_dev")),
        "openrouter": json.loads(get("openrouter", "openrouter")),
    }


def apply_corrections(raw: dict) -> list[dict]:
    """Apply the documented fixes of source errors, each only while the error is still there."""
    tables = {"epoch_models": raw["models"], "epoch_eci": raw["eci"]}
    report = []
    for correction in CORRECTIONS:
        applied = False
        for row in tables[correction["source"]]:
            if row.get(correction["key"]) != correction["model"]:
                continue
            current = row.get(correction["field"], "")
            wrong = correction["from"]
            same = number(current) == wrong if isinstance(wrong, (int, float)) else current[:10] == wrong
            if same:
                row[correction["field"]] = str(correction["to"])
                applied = True
        report.append({**correction, "applied": applied})
    return report


def eci_models(raw: list[dict[str, str]], today: date) -> list[dict]:
    result = []
    for row in raw:
        when, value = day(row.get("date")), number(row.get("eci"))
        if when is None or value is None or when > today:
            continue
        access = row.get("Accessibility group", "")
        result.append(
            {
                "name": row["Model"].strip(),
                "org": prices.organisation_of(row.get("Organization", "")) or "Independent",
                "country": country_group(row.get("Country (of organization)", "")),
                "access": {"Open weights": "open", "Closed weights": "closed"}.get(access, "other"),
                "d": when,
                "v": value,
                "lo": number(row.get("eci_ci_low")),
                "hi": number(row.get("eci_ci_high")),
            }
        )
    return sorted(result, key=lambda model: (model["d"], model["name"]))


def model_records(models: list[dict], *, lowest: bool = False) -> list[tuple[date, float, object]]:
    return stats.running_records([(m["d"], m["v"], m["name"]) for m in models], lowest=lowest)


# ---------------------------------------------------------------- chapters


def intelligence(models: list[dict], today: date) -> dict:
    current = [m for m in models if m["d"] >= ECI_START]
    records = model_records(current)
    record_names = {name for _, _, name in records}
    by_name = {m["name"]: m for m in models}
    references = [
        {"v": by_name[name]["v"], "label": label}
        for name, label in [
            ("GPT-4 (Mar 2023)", "GPT-4"),
            ("o1", "o1"),
            ("GPT-5", "GPT-5"),
        ]
        if name in by_name
    ]
    frontier = step_series("frontier", "Best model so far", records)
    frontier["trend"] = trend_json([(d, v) for d, v, _ in records], log=False, today=today, seed=10)
    first, last = records[0], records[-1]
    return {
        "charts": {
            "eci": {
                "id": "eci",
                "unit": "eci",
                "scale": "linear",
                "points": [
                    point(
                        m["d"],
                        m["v"],
                        m["name"],
                        m["org"],
                        "frontier" if m["name"] in record_names else "other",
                        lo=m["lo"],
                        hi=m["hi"],
                    )
                    for m in models
                ],
                "groups": [
                    {"id": "frontier", "label": "Set a new record"},
                    {"id": "other", "label": "Other models"},
                ],
                "series": [frontier],
                "refs": references,
            }
        },
        "facts": {
            "models": len(models),
            "first": {"n": first[2], "v": sig(first[1]), "d": first[0].isoformat()},
            "last": {"n": last[2], "v": sig(last[1]), "d": last[0].isoformat()},
            "records": len(records),
        },
    }


def metr_name(key: str) -> str:
    base = key.removesuffix("_inspect")
    if base in METR_NAMES:
        return METR_NAMES[base]
    words = []
    for word in base.split("_"):
        if word.isdigit() and len(word) == 8:
            continue
        words.append({"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}.get(word, word))
    text = " ".join(words)
    # "claude 3 5 sonnet" and "gpt 5 2" carry versions as separate digits.
    parts = text.split(" ")
    merged: list[str] = []
    for part in parts:
        if part.isdigit() and merged and merged[-1].replace(".", "").isdigit():
            merged[-1] = f"{merged[-1]}.{part}"
        else:
            merged.append(part)
    text = " ".join(merged)
    if text.startswith("GPT "):
        text = "GPT-" + text[4:]
    return " ".join(
        word if word[:1].isupper() or word[:1].isdigit() else word.capitalize() for word in text.split(" ")
    )


def tasks(metr: dict, today: date) -> dict:
    measured = []
    for key, entry in (metr.get("results") or {}).items():
        when = day(entry.get("release_date"))
        metrics = entry.get("metrics") or {}
        half = metrics.get("p50_horizon_length") or {}
        value = number(half.get("estimate"))
        if when is None or value is None or when > today:
            continue
        version = str(entry.get("benchmark_name", "")).rsplit("-", 1)[-1]
        measured.append(
            {
                "name": metr_name(key),
                "d": when,
                "v": value,
                "lo": number(half.get("ci_low")),
                "hi": number(half.get("ci_high")),
                "p80": number((metrics.get("p80_horizon_length") or {}).get("estimate")),
                "sota": bool(metrics.get("is_sota")),
                "current": version == "v1.1",
            }
        )
    measured.sort(key=lambda m: m["d"])
    fitted = [m for m in measured if m["current"] and m["sota"] and m["d"] >= date(2023, 1, 1)]
    series = step_series("sota", "Longest task so far", [(m["d"], m["v"], m["name"]) for m in fitted])
    series["trend"] = trend_json(
        [(m["d"], m["v"]) for m in fitted],
        log=True,
        today=today,
        seed=20,
        levels=[
            ("week", "a full work week (40 hours)", 2400.0),
            ("month", "a full work month (167 hours)", 10020.0),
            ("year", "a full work year (2,000 hours)", 120000.0),
        ],
    )
    published = (metr.get("doubling_time_in_days") or {}).get("from_2023_on") or {}
    last = fitted[-1]
    return {
        "charts": {
            "horizon": {
                "id": "horizon",
                "unit": "minutes",
                "scale": "log",
                "points": [
                    point(
                        m["d"],
                        m["v"],
                        m["name"],
                        "",
                        "frontier" if m in fitted else "older" if not m["current"] else "other",
                        lo=m["lo"],
                        hi=m["hi"],
                        p80=sig(m["p80"]) if m["p80"] else None,
                    )
                    for m in measured
                ],
                "groups": [
                    {"id": "frontier", "label": "Set a new record"},
                    {"id": "other", "label": "Other models"},
                    {"id": "older", "label": "Older test version"},
                ],
                "series": [series],
                "refs": [
                    {"v": 480, "label": "1 work day"},
                    {"v": 2400, "label": "1 work week"},
                ],
            }
        },
        "facts": {
            "measured": len(measured),
            "first": {"n": fitted[0]["name"], "v": sig(fitted[0]["v"]), "d": fitted[0]["d"].isoformat()},
            "last": {
                "n": last["name"],
                "v": sig(last["v"]),
                "d": last["d"].isoformat(),
                "p80": sig(last["p80"]) if last["p80"] else None,
            },
            "published_doubling_days": number(published.get("point_estimate")),
            "reliable_limit_minutes": 960,
        },
    }


def language_models(models: list[dict[str, str]], today: date, checks: Audit) -> list[dict]:
    result = []
    for row in models:
        when = day(row.get("Publication date"))
        if when is None or when > today:
            continue
        entry = {
            "name": row["Model"].strip(),
            "org": prices.organisation_of(row.get("Organization", "")),
            "d": when,
            "language": "Language" in row.get("Domain", ""),
            "notable": bool(row.get("Notability criteria", "").strip()),
            "params": number(row.get("Parameters")),
            "compute": number(row.get("Training compute (FLOP)")),
            "cost": number(row.get("Training compute cost (2023 USD)")),
            "confidence": row.get("Confidence", "").strip(),
            "slip": False,
        }
        if entry["params"]:
            quoted = audit.unit_slip(entry["params"], row.get("Parameters notes", ""))
            if quoted:
                entry["slip"] = True
                if entry["notable"]:
                    checks.flag(
                        "unit",
                        "params",
                        entry["name"],
                        f"value {short(entry['params'])} but its note quotes {short(quoted)}, "
                        "so it is not used",
                    )
        result.append(entry)
    return result


def scale_chart(
    identifier: str,
    unit: str,
    rows_: list[dict],
    field: str,
    *,
    fit_from: date,
    show_from: date,
    today: date,
    seed: int,
    checks: Audit,
    record_label: str,
    fitted_label: str,
) -> tuple[dict, dict]:
    """Chart of one database quantity. Only vetted rows set records or enter the trend."""
    usable = [m for m in rows_ if audit.status(m, field) is None]
    points = [(m["d"], m[field], m["name"]) for m in usable]
    records, held = audit.records(points, chart=identifier, today=today, audit=checks)
    usable = [m for m in usable if m["name"] not in held]
    top = {(name, d) for d, _, name in stats.top_at_release([(m["d"], m[field], m["name"]) for m in usable])}
    fitted = [m for m in usable if (m["name"], m["d"]) in top and m["d"] >= fit_from]
    fitted_keys = {(m["name"], m["d"]) for m in fitted}
    usable_keys = {(m["name"], m["d"]) for m in usable}
    shown_usable = [m for m in usable if m["d"] >= show_from]
    low, high = min(m[field] for m in shown_usable), max(m[field] for m in shown_usable)
    shown = []
    beyond = []
    for m in rows_:
        if m["d"] < show_from:
            continue
        if (m["name"], m["d"]) in usable_keys:
            shown.append((m, None))
        elif low <= m[field] <= high:
            shown.append((m, "held" if m["name"] in held else audit.status(m, field)))
        else:
            beyond.append(m)
    for m in sorted(beyond, key=lambda m: -m[field])[:5]:
        if m[field] > high:
            checks.flag(
                "beyond",
                identifier,
                m["name"],
                f"{short(m[field])} is above every vetted value and {WHY[audit.status(m, field) or 'held']}, "
                "so it is not drawn",
            )
    series = step_series("records", record_label, [r for r in records if r[0] >= show_from])
    series["trend"] = trend_json([(m["d"], m[field]) for m in fitted], log=True, today=today, seed=seed)
    chart = {
        "id": identifier,
        "unit": unit,
        "scale": "log",
        "points": [
            point(
                m["d"],
                m[field],
                m["name"],
                m["org"],
                "frontier" if (m["name"], m["d"]) in fitted_keys else "other",
                q=reason,
                c=m["confidence"] or None,
            )
            for m, reason in shown
        ],
        "groups": [
            {"id": "frontier", "label": fitted_label},
            {"id": "other", "label": "Other models"},
        ],
        "series": [series],
        "refs": [],
    }
    confident = [m for m in shown_usable if m["confidence"] == "Confident"]
    largest = max(shown_usable, key=lambda m: m[field])
    largest_confident = max(confident, key=lambda m: m[field])

    def named(m: dict) -> dict:
        return {"n": m["name"], "v": sig(m[field]), "d": m["d"].isoformat(), "c": m["confidence"]}

    facts = {
        "models": len(shown),
        "vetted": len(shown_usable),
        "set_aside": sum(1 for _, reason in shown if reason) + len(beyond),
        "largest": named(largest),
        "largest_confident": named(largest_confident),
    }
    return chart, facts


def size(models: list[dict], today: date, checks: Audit) -> dict:
    language = [m for m in models if m["language"] and m["params"] and m["d"] >= date(2012, 1, 1)]
    chart, facts = scale_chart(
        "params",
        "params",
        language,
        "params",
        fit_from=date(2017, 1, 1),
        show_from=date(2017, 1, 1),
        today=today,
        seed=30,
        checks=checks,
        record_label="Largest vetted model",
        fitted_label="Among the 10 largest when released",
    )
    disclosure = []
    for year in range(2017, today.year + 1):
        cohort = [m for m in models if m["language"] and m["notable"] and m["d"].year == year]
        if cohort:
            disclosure.append(
                {
                    "label": str(year),
                    "n": len(cohort),
                    "params": round(sum(1 for m in cohort if m["params"]) / len(cohort), 3),
                    "compute": round(sum(1 for m in cohort if m["compute"]) / len(cohort), 3),
                }
            )
    return {"charts": {"params": chart}, "bars": {"disclosure": disclosure}, "facts": facts}


def compute(models: list[dict], today: date, checks: Audit) -> dict:
    chart, facts = scale_chart(
        "compute",
        "flop",
        [m for m in models if m["compute"]],
        "compute",
        fit_from=date(2010, 1, 1),
        show_from=date(2010, 1, 1),
        today=today,
        seed=40,
        checks=checks,
        record_label="Largest training run",
        fitted_label="Among the 10 largest when released",
    )
    return {"charts": {"compute": chart}, "facts": facts}


def cost(models: list[dict], today: date, checks: Audit) -> dict:
    priced = [m for m in models if m["cost"] and m["d"] >= date(2012, 1, 1)]
    chart, facts = scale_chart(
        "cost",
        "usd",
        priced,
        "cost",
        fit_from=date(2016, 1, 1),
        show_from=date(2012, 1, 1),
        today=today,
        seed=50,
        checks=checks,
        record_label="Most expensive training run",
        fitted_label="Among the 10 most expensive when released",
    )
    recent = [m for m in priced if audit.status(m, "cost") is None and m["d"] >= add_years(today, -2)]
    facts["estimates_last_two_years"] = len(recent)
    return {"charts": {"cost": chart}, "facts": facts}


def price(models: list[dict], attached: dict[str, prices.Price], today: date, checks: Audit) -> dict:
    priced = []
    for m in models:
        found = attached.get(m["name"])
        if found is None:
            continue
        # A hosting price of an open model counts from the day it was recorded.
        seen = day(found.seen) if found.seen else None
        when = max(m["d"], seen) if seen and m["access"] != "closed" else m["d"]
        priced.append(dict(m, d=when, released=m["d"], usd=found.usd, kind=found.kind, seen=found.seen))
    priced.sort(key=lambda m: (m["d"], m["name"]))
    by_name = {m["name"]: m for m in models}
    series = []
    levels = []
    for identifier, label, reference in [
        ("gpt4", "GPT-4 level", "GPT-4 (Mar 2023)"),
        ("o1", "o1 level", "o1"),
        ("gpt5", "GPT-5 level", "GPT-5"),
    ]:
        anchor = by_name.get(reference)
        if anchor is None:
            continue
        eligible = [m for m in priced if m["v"] >= anchor["v"] and m["d"] >= anchor["d"]]
        records, _ = audit.records(
            [(m["d"], m["usd"], m["name"]) for m in eligible],
            chart="price",
            today=today,
            audit=checks,
            lowest=True,
        )
        if len(records) < 2:
            continue
        entry = step_series(identifier, label, records)
        entry["trend"] = trend_json(
            [(d, v) for d, v, _ in records], log=True, today=today, seed=60 + len(series)
        )
        series.append(entry)
        first, last = records[0], records[-1]
        levels.append(
            {
                "id": identifier,
                "label": label,
                "reference": reference,
                "eci": sig(anchor["v"]),
                "first": {"n": first[2], "v": sig(first[1]), "d": first[0].isoformat()},
                "last": {"n": last[2], "v": sig(last[1]), "d": last[0].isoformat()},
                "fold": sig(first[1] / last[1]),
                "records": len(records),
            }
        )
    record_names = {name for _, _, name in model_records([m for m in models if m["d"] >= ECI_START])}
    best = [m for m in priced if m["name"] in record_names]
    series.append(
        {
            "id": "best",
            "label": "Best model at the time",
            "kind": "points",
            "points": [{"d": m["d"].isoformat(), "v": sig(m["usd"]), "n": m["name"]} for m in best],
        }
    )
    kinds = defaultdict(int)
    for m in priced:
        kinds[m["kind"]] += 1
    return {
        "charts": {
            "price": {
                "id": "price",
                "unit": "usd_mtok",
                "scale": "log",
                "points": [
                    point(
                        m["d"],
                        m["usd"],
                        m["name"],
                        m["org"],
                        "other",
                        e=sig(m["v"]),
                        k=m["kind"],
                        s=m["seen"],
                    )
                    for m in priced
                ],
                "groups": [{"id": "other", "label": "Models with a known price"}],
                "series": series,
                "refs": [],
            }
        },
        "facts": {
            "priced": len(priced),
            "indexed": len(models),
            "levels": levels,
            "kinds": dict(kinds),
            "best_first": {"n": best[0]["name"], "v": sig(best[0]["usd"]), "d": best[0]["d"].isoformat()}
            if best
            else None,
            "best_last": {"n": best[-1]["name"], "v": sig(best[-1]["usd"]), "d": best[-1]["d"].isoformat()}
            if best
            else None,
        },
    }


def lag(
    leader: list[tuple[date, float, object]],
    follower: list[tuple[date, float, object]],
    today: date,
) -> list[dict]:
    """Months the follower needed to match each level the leader had already reached."""
    result = []
    for when, level, name in [*follower, (today, follower[-1][1], "Today")]:
        match = next(((d, n) for d, v, n in leader if v >= level), None)
        if match is None or match[0] > when:
            continue
        result.append(
            {
                "d": when.isoformat(),
                "v": round(months_between(match[0], when), 1),
                "n": str(name),
                "m": str(match[1]),
            }
        )
    return result


def gap_chart(
    identifier: str,
    models: list[dict],
    key: str,
    groups: list[tuple[str, str]],
    today: date,
    seed: int,
) -> tuple[dict, dict[str, list]]:
    series = []
    records_by_group = {}
    for offset, (group, label) in enumerate(groups):
        members = [m for m in models if m[key] == group and (m["d"] >= ECI_START or group != groups[0][0])]
        records = model_records(members)
        records_by_group[group] = records
        entry = step_series(group, label, records)
        entry["trend"] = trend_json(
            [(d, v) for d, v, _ in records], log=False, today=today, seed=seed + offset
        )
        series.append(entry)
    known = {group for group, _ in groups}
    chart = {
        "id": identifier,
        "unit": "eci",
        "scale": "linear",
        "points": [
            point(m["d"], m["v"], m["name"], m["org"], m[key] if m[key] in known else "other") for m in models
        ],
        "groups": [{"id": group, "label": label} for group, label in groups]
        + [{"id": "other", "label": "Other"}],
        "series": series,
        "refs": [],
    }
    return chart, records_by_group


def lag_facts(series: list[dict], today: date) -> dict:
    recent = [p["v"] for p in series[:-1] if day(p["d"]) >= add_years(today, -2)]
    earlier = [p["v"] for p in series[:-1] if day(p["d"]) < add_years(today, -2)]
    return {
        "now": series[-1]["v"],
        "matched": series[-1]["m"],
        "recent_average": round(sum(recent) / len(recent), 1) if recent else None,
        "earlier_average": round(sum(earlier) / len(earlier), 1) if earlier else None,
        "latest_record": series[-2] if len(series) > 1 else None,
    }


def openness(models: list[dict], today: date) -> dict:
    chart, records = gap_chart(
        "access",
        models,
        "access",
        [("closed", "Best closed model"), ("open", "Best open model")],
        today,
        70,
    )
    series = lag(records["closed"], records["open"], today)
    best_closed, best_open = records["closed"][-1], records["open"][-1]
    return {
        "charts": {
            "access": chart,
            "access_lag": {
                "id": "access_lag",
                "unit": "months",
                "scale": "linear",
                "points": [],
                "groups": [],
                "series": [{"id": "lag", "label": "Months behind", "kind": "line", "points": series}],
                "refs": [],
            },
        },
        "facts": {
            "lag": lag_facts(series, today),
            "closed": {"n": best_closed[2], "v": sig(best_closed[1]), "d": best_closed[0].isoformat()},
            "open": {"n": best_open[2], "v": sig(best_open[1]), "d": best_open[0].isoformat()},
            "counts": {
                group: sum(1 for m in models if m["access"] == group) for group in ("open", "closed", "other")
            },
        },
    }


def race(models: list[dict], today: date) -> dict:
    chart, records = gap_chart(
        "country",
        models,
        "country",
        [("us", "Best US model"), ("china", "Best Chinese model")],
        today,
        80,
    )
    series = lag(records["us"], records["china"], today)
    overall = model_records([m for m in models if m["d"] >= ECI_START])
    by_name = {m["name"]: m for m in models}
    days_on_top: dict[str, int] = defaultdict(int)
    record_count: dict[str, int] = defaultdict(int)
    for index, (when, _, name) in enumerate(overall):
        until = overall[index + 1][0] if index + 1 < len(overall) else today
        organisation = by_name[str(name)]["org"]
        days_on_top[organisation] += (until - when).days
        record_count[organisation] += 1
    best: dict[str, dict] = {}
    for model in models:
        if model["org"] not in best or model["v"] > best[model["org"]]["v"]:
            best[model["org"]] = model
    board = sorted(best.values(), key=lambda m: -m["v"])[:12]
    return {
        "charts": {
            "country": chart,
            "country_lag": {
                "id": "country_lag",
                "unit": "months",
                "scale": "linear",
                "points": [],
                "groups": [],
                "series": [{"id": "lag", "label": "Months behind", "kind": "line", "points": series}],
                "refs": [],
            },
        },
        "board": [
            {
                "org": m["org"],
                "country": m["country"],
                "n": m["name"],
                "v": sig(m["v"]),
                "d": m["d"].isoformat(),
                "access": m["access"],
                "days_on_top": days_on_top.get(m["org"], 0),
                "records": record_count.get(m["org"], 0),
            }
            for m in board
        ],
        "facts": {
            "lag": lag_facts(series, today),
            "us": {"n": records["us"][-1][2], "v": sig(records["us"][-1][1])},
            "china": {"n": records["china"][-1][2], "v": sig(records["china"][-1][1])},
            "days_tracked": (today - overall[0][0]).days,
            "counts": {
                group: sum(1 for m in models if m["country"] == group) for group in ("us", "china", "other")
            },
        },
    }


TRAINING_FORMATS = [
    "FP32 (single precision) performance (FLOP/s)",
    "FP16 (half precision) performance (FLOP/s)",
    "Tensor-FP16/BF16 performance (FLOP/s)",
]


def hardware(chips: list[dict[str, str]], clusters: list[dict[str, str]], today: date, checks: Audit) -> dict:
    priced = []
    for row in chips:
        when = day(row.get("Release date"))
        # One yardstick for every chip: speed at the 32 or 16 bit formats used for training.
        # The 8 and 4 bit formats newer chips add for running models would inflate the trend.
        speed = max((number(row.get(column)) or 0.0) for column in TRAINING_FORMATS)
        usd = number(row.get("Release price (USD)"))
        if when and speed and usd and date(2008, 1, 1) <= when <= today:
            priced.append(
                {
                    "name": row["Hardware name"].strip(),
                    "org": row.get("Manufacturer", ""),
                    "d": when,
                    "v": speed / usd,
                }
            )
    fitted = [m for m in priced if m["d"] >= date(2012, 1, 1)]
    chip_records, _ = audit.records(
        [(m["d"], m["v"], m["name"]) for m in priced], chart="chips", today=today, audit=checks
    )
    chip_series = step_series("records", "Best value so far", chip_records)
    chip_series["trend"] = trend_json([(m["d"], m["v"]) for m in fitted], log=True, today=today, seed=90)

    sites = []
    for row in clusters:
        when = day(row.get("First Operational Date"))
        size_ = number(row.get("H100 equivalents"))
        if when and size_ and when <= today and row.get("Status") == "Existing":
            sites.append(
                {
                    "name": row["Name"].strip(),
                    "org": row.get("Owner", "") or row.get("Sector", ""),
                    "d": when,
                    "v": size_,
                    "country": country_group(row.get("Country", "")),
                }
            )
    top = {(name, d) for d, _, name in stats.top_at_release([(m["d"], m["v"], m["name"]) for m in sites])}
    shown = [m for m in sites if m["d"] >= date(2017, 1, 1)]
    top_sites = [m for m in shown if (m["name"], m["d"]) in top and m["d"] >= date(2019, 1, 1)]
    top_names = {(m["name"], m["d"]) for m in top_sites}
    cluster_records, _ = audit.records(
        [(m["d"], m["v"], m["name"]) for m in shown], chart="clusters", today=today, audit=checks
    )
    cluster_series = step_series("records", "Largest cluster so far", cluster_records)
    cluster_series["trend"] = trend_json(
        [(m["d"], m["v"]) for m in top_sites], log=True, today=today, seed=95
    )
    largest = next(m for m in shown if m["name"] == cluster_records[-1][2])
    return {
        "charts": {
            "chips": {
                "id": "chips",
                "unit": "ops_per_usd",
                "scale": "log",
                "points": [
                    point(m["d"], m["v"], m["name"], m["org"], "frontier" if m in fitted else "other")
                    for m in priced
                ],
                "groups": [
                    {"id": "frontier", "label": "AI chips with a known launch price"},
                    {"id": "other", "label": "Before 2012"},
                ],
                "series": [chip_series],
                "refs": [],
            },
            "clusters": {
                "id": "clusters",
                "unit": "h100e",
                "scale": "log",
                "points": [
                    point(
                        m["d"],
                        m["v"],
                        m["name"],
                        m["org"],
                        "frontier" if (m["name"], m["d"]) in top_names else "other",
                    )
                    for m in shown
                ],
                "groups": [
                    {"id": "frontier", "label": "Among the 10 largest when switched on"},
                    {"id": "other", "label": "Other clusters"},
                ],
                "series": [cluster_series],
                "refs": [],
            },
        },
        "facts": {
            "chips": len(priced),
            "clusters": len(shown),
            "largest": {"n": largest["name"], "v": sig(largest["v"]), "d": largest["d"].isoformat()},
            "clusters_through": max(m["d"] for m in shown).isoformat(),
        },
    }


def explorer(models: list[dict], attached: dict[str, prices.Price], database: list[dict]) -> list[dict]:
    details = {m["name"]: m for m in database}
    table = []
    for model in sorted(models, key=lambda m: -m["v"]):
        extra = details.get(model["name"], {})
        priced = attached.get(model["name"])
        table.append(
            {
                "n": model["name"],
                "o": model["org"],
                "c": model["country"],
                "a": model["access"],
                "d": model["d"].isoformat(),
                "e": sig(model["v"]),
                "p": sig(priced.usd) if priced else None,
                "pk": priced.kind if priced else None,
                # Unvetted or speculative values stay out of the table, like everywhere else.
                "params": sig(extra["params"])
                if extra.get("params") and audit.status(extra, "params") is None
                else None,
                "compute": sig(extra["compute"])
                if extra.get("compute") and audit.status(extra, "compute") is None
                else None,
            }
        )
    return table


# ---------------------------------------------------------------- assembly


def build(snapshot: Snapshot) -> dict:
    raw = load(snapshot)
    retrieved = max(record["retrieved_at"] for record in snapshot.records.values())
    today = date.fromisoformat(retrieved[:10])
    checks = Audit()
    corrections = apply_corrections(raw)
    database = language_models(raw["models"], today, checks)
    models = eci_models(raw["eci"], today)
    for model in models:
        named = NAME_MONTH.search(model["name"])
        if named and named.group(1).lower() in MONTHS:
            stated = int(named.group(2)) * 12 + MONTHS[named.group(1).lower()]
            if abs(stated - (model["d"].year * 12 + model["d"].month)) > 1:
                checks.flag("date", "eci", model["name"], f"name and date {model['d']} disagree")
    attached = prices.attach(
        models,
        epoch_files=raw["price_files"],
        history=raw["llm_prices"],
        models_dev=raw["models_dev"],
        openrouter=raw["openrouter"],
    )
    chapters = {
        "intelligence": intelligence(models, today),
        "tasks": tasks(raw["metr"], today),
        "size": size(database, today, checks),
        "compute": compute(database, today, checks),
        "cost": cost(database, today, checks),
        "price": price(models, attached, today, checks),
        "openness": openness(models, today),
        "race": race(models, today),
        "hardware": hardware(raw["hardware"], raw["clusters"], today, checks),
    }
    used = {record["source"] for record in snapshot.records.values()}
    sources = []
    for identifier, source in SOURCES.items():
        if identifier not in used:
            continue
        files = [r for r in snapshot.records.values() if r["source"] == identifier]
        sources.append(
            {
                "id": identifier,
                "name": source.name,
                "publisher": source.publisher,
                "page": source.page,
                "license": source.license,
                "description": source.description,
                "retrieved_at": max(r["retrieved_at"] for r in files),
                "files": [{"url": r["url"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in files],
            }
        )
    story = {
        "version": 4,
        "generated_on": today.isoformat(),
        "projection_until": add_years(today, PROJECTION_YEARS).isoformat(),
        "data_through": max(m["d"] for m in models).isoformat(),
        "chapters": chapters,
        "explorer": explorer(models, attached, database),
        "sources": sources,
        "quality": {
            "corrections": corrections,
            "launch_prices": [entry for entry in prices.LAUNCH if entry["model"] in attached],
            "flags": checks.flags,
            "database_models": len(database),
            "indexed_models": len(models),
            "priced_models": len(attached),
        },
    }
    check(story)
    return story


def check(story: dict) -> None:
    """Refuse to publish a bundle that is obviously broken."""
    chapters = story["chapters"]
    problems = []
    if story["quality"]["indexed_models"] < 150:
        problems.append("fewer than 150 models in the capability index")
    for name, chapter in chapters.items():
        for identifier, chart in chapter["charts"].items():
            if not chart["series"] or not chart["series"][0]["points"]:
                problems.append(f"{name}/{identifier}: no series")
            for series in chart["series"]:
                for entry in series["points"]:
                    if not math.isfinite(entry["v"]):
                        problems.append(f"{name}/{identifier}: non-finite value")
    for name, identifier in [
        ("intelligence", "eci"),
        ("tasks", "horizon"),
        ("size", "params"),
        ("compute", "compute"),
        ("cost", "cost"),
        ("price", "price"),
        ("hardware", "chips"),
        ("hardware", "clusters"),
    ]:
        if not chapters[name]["charts"][identifier]["series"][0].get("trend"):
            problems.append(f"{name}/{identifier}: no trend could be fitted")
    if problems:
        raise RuntimeError("Refusing to publish: " + "; ".join(sorted(set(problems))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="replay the last cached snapshot")
    arguments = parser.parse_args()
    snapshot = Snapshot(offline=arguments.offline)
    story = build(snapshot)
    snapshot.save_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(story, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} kB)")
    for name, chapter in story["chapters"].items():
        for identifier, chart in chapter["charts"].items():
            trend = chart["series"][0].get("trend")
            if trend:
                print(
                    f"  {name}/{identifier}: {trend['rate']['v']:g} per year "
                    f"[{trend['rate']['lo']:g}, {trend['rate']['hi']:g}], n={trend['n']}, "
                    f"{trend['shape']['verdict']}, basis={trend['basis']}"
                )


if __name__ == "__main__":
    main()
