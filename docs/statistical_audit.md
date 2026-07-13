# Statistical Audit and Remediation

Reference snapshot: 2026-05-15. Audit performed: 2026-07-12.

## Executive verdict

The project had strong provenance ambitions and useful schema-level tests, but several headline forecasts were not statistically defensible. The main failure was pseudo-replication: repeated leaderboard snapshots and very large benchmark tables acted like independent evidence. A second failure was temporal leakage: ingestion timestamps were treated as evaluation dates. Together these could manufacture apparent progress where no comparable time series existed.

The pipeline has been corrected to prefer an honest refusal to forecast over a precise-looking extrapolation unsupported by the data.

## Critical findings and fixes

| Severity | Finding | Why it was wrong | Remediation |
|---|---|---|---|
| Critical | 59,170 rows shared the 2026-05-19 ingestion date, including undated Open LLM Leaderboard and SWE-bench evidence. | Report-generation or dataset-modification time is not a model evaluation date. It created an artificial 2026 observation. | Added `date_provenance`, `temporal_eligible`, and `as_of_eligible`; undated and snapshot-only rows remain cross-sectional evidence but cannot estimate trends. |
| Critical | Domain frontier history aggregated raw rows directly. | A benchmark with 40,000 repeated snapshots received roughly 40,000 votes while a small independent benchmark received only a few. | Collapse to one model/benchmark/year observation, then one benchmark frontier/year; combine benchmarks with equal first-order influence. |
| Critical | Cross-domain fallback slopes were assigned when a domain had no longitudinal panel. | Finance appeared to improve by 64.236 points/year because a 2025 FinanceBench scale was compared with a different 2026 QFBench scale. Legal and search inherited unrelated median progress. | Estimate velocity only within the same benchmark observed in at least two years. No comparable history means slope 0, `forecast_enabled=false`, and a flat scenario path. |
| High | Missing model-family signals were normalized to zero. | Lack of public disclosure was silently interpreted as zero capability, biasing rankings against under-observed families. | Missing min-max inputs now receive neutral 50; uncertainty is reported separately through coverage and effective-evidence fields. |
| High | Raw evidence row count shrank bootstrap noise. | Repeated rows created false certainty for families heavily represented in large public tables. | Rank stress-test noise now depends on independent source-family coverage, not raw row volume. |
| High | Family domain quality used a row-level 90th percentile over heterogeneous benchmarks. | Large or frequently refreshed benchmarks dominated the cost/quality join. | Compute family quality within each benchmark first, then use a bounded benchmark-weighted mean. |
| High | BLS major-group employment was copied to every detailed occupation. | A 5-10 million worker group total could be counted hundreds of times. Job-growth percentages were also eligible as fallback weights. | Allocate major-group totals across unmatched occupations, preserve provenance, and never use growth rates as population weights. |
| High | Current OpenRouter prices grouped by model release year were treated as a historical price trend. | A current catalog is a survivor-biased cross-section, not a record of past prices. | Keep the cohort slope only as a diagnostic; future price paths use explicit scenario assumptions and are labeled as such. |
| Medium | “Next frontier probabilities” resemble calibrated forecasts although they only perturb component weights and noise. | No transition model, backtest, base rate, or longitudinal lab dynamics supports a literal probability interpretation. | Retain `simulation_win_share` only as a ranking stress test; documentation explicitly prohibits interpreting it as probability. |
| Medium | Contract tests asserted files, ranges, and labels but not scientific invariants. | Invalid estimates could pass every test. | Added tests for as-of cutoffs, temporal eligibility, flat no-history forecasts, effective observations, neutral missingness, and employment allocation. |

## Remaining limitations

- Benchmark panels still mix preference, accuracy, agent-scaffold, and task-specific metrics. They are scenario inputs, not a universal capability scale.
- LMArena methodology changed over time; within-snapshot ranking reduces scale drift but does not remove every comparability break.
- Most domains have too little stable longitudinal evidence for confident 5- or 10-year extrapolation.
- Labor indexes remain transparent heuristics built from observed Claude usage, task text, and bottleneck rules; they are not causal estimates of job loss.
- The checked-in upstream snapshot is stale relative to the audit date. A fresh data run is required for current-market claims, but freshness alone must not override the 2026-05-15 reproducibility cutoff without intentionally changing the study reference date.

## Validation contract

The repaired pipeline must satisfy all of the following:

1. No dated analytical row exceeds `REFERENCE_DATE`.
2. Ingestion timestamps and dataset snapshot dates cannot be used as evaluation dates.
3. A domain with zero longitudinal benchmarks has zero used slope and a flat forecast path.
4. Effective observations never exceed raw rows.
5. Missing public model evidence is neutral, not zero.
6. Allocated employment rows cannot carry a full major-group total.
7. Simulation shares and scenario envelopes are never labeled calibrated probabilities or confidence intervals.
