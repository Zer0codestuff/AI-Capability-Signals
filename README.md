# AI Capability Signals

Reproducible, uncertainty-quantified analysis of public frontier AI capability, price and disclosure signals.

Published report: [report/frontier_signals.html](report/frontier_signals.html)

## What changed in 2.0

Version 1 produced 49 analytical tables, multi-year forecasts, leadership "probabilities" and labour-replacement rankings. An audit of that version — kept at [`docs/audit_of_previous_version.md`](docs/audit_of_previous_version.md) — found that many of those outputs were not statistically defensible: composite indices of incomparable metrics, Monte Carlo procedures that resampled neither data nor models, a current price catalogue treated as a historical series, seven CSV tables that were hardcoded prose, and tests that validated schemas rather than estimators.

Version 2 answers fewer questions, and only ones the public data can support. Every published number is either a direct measurement from a named source field, or the output of a named estimator with an uncertainty interval whose construction is stated. Claims the data cannot support are recorded as refusals rather than estimated.

## Questions this project answers

1. **Disclosure.** Do open-weight releases disclose more than closed-weight releases? (difference of proportions, BH-corrected)
2. **Compute scaling.** How fast does disclosed training compute grow, and does a fitted trend forecast better than assuming no change? (HC3 OLS + Theil–Sen, rolling-origin backtest)
3. **Price structure.** What does a given measured quality level cost in the current catalogue, and how wide is the price spread at fixed quality? (Pareto frontier, matched-quality bands; quality and price from the same record)
4. **Benchmark agreement.** Do public benchmarks agree about model ordering strongly enough to justify a composite score? (Kendall τ with model-level bootstrap; answer: no)
5. **Open-weight lag.** How far behind is the open-weight arena frontier, and how long does catch-up take? (within-regime gap trend; Kaplan–Meier lag with right censoring)
6. **Usage composition.** What does observed Claude usage look like by occupation, and do the consumer and API surfaces agree? (paired contrast; not employment impact)
7. **Calendar control.** Do release dates cluster on weekday/month/quarter after multiple-comparison correction? (year-preserving permutation null + BH)

## Questions this project refuses

The refusals ledger (`data/analysis/refusals.csv`) is a first-class output. Typical refusals:

- a single composite capability score (benchmarks do not agree enough)
- a historical price trend (the catalogue is a cross-section)
- a date when open weights will match the closed frontier (trend ≠ crossing forecast)
- current SWE-bench Verified SOTA (the public directory is stale)
- job replacement / disruption rankings (usage composition is not employment impact)
- release counts for recent years (Epoch curation is right-censored)

## Reproduce

```bash
uv sync
uv run aicap                     # full refresh from public sources
uv run aicap --from-interim      # re-analyse cached frames
uv run python -m unittest discover -s tests -v
```

Offline / CI:

```bash
uv run aicap --offline --from-interim --skip-report
```

## Layout

```
src/aicap/
  sources/     validated ingestion with schema contracts
  analysis/    one module per question; refusals when unsupported
  report/      rendering only — no computation
  stats.py     estimators covered by tests/test_stats.py
  identity.py  strict model matcher; no family-best fallback
data/
  raw/         content-addressed source snapshots (gitignored)
  interim/     parsed frames (gitignored)
  analysis/    published tables (versioned)
report/        Markdown + HTML report
docs/          methodology, data dictionary, audit of v1
```

## Method in one paragraph

Reference date = minimum of each source's latest observation. Intervals are Wilson (proportions), HC3 (OLS), Theil–Sen (robust slopes), and percentile bootstraps that resample the unit of analysis. Multiple comparisons inside a pre-registered family are Benjamini–Hochberg corrected. Forecasts require beating a last-value baseline out of sample; published interval widths come from measured backtest errors. See [`docs/methodology.md`](docs/methodology.md).

## Licence

Code is MIT. Third-party source data keeps its original terms — see [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).
