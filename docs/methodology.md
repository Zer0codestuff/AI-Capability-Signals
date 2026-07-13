# Methodology

Reference date: 2026-05-15.

This project favors source preservation over fragile live scraping. Each fetch writes raw snapshots to `data/raw/`; processed tables are regenerated from those snapshots.

## Normalization

- Dates are parsed into ISO dates with a `release_precision` field.
- Prices from OpenRouter are normalized from USD/token to USD/1M tokens.
- Training compute is kept in FLOP.
- Benchmark rows preserve benchmark identity and unit; incompatible benchmarks are not averaged.
- Model identity is split into `vendor`, `model_family`, and `product_line`. Vendor names alone do not assign model family.

## Heuristic Analysis

- Company scores are heuristic indexes. See `data/analysis/company_score_methodology.csv` for component weights and `data/analysis/company_score_sensitivity.csv` for weight sensitivity.
- Direct model benchmark matching is name-based and explicitly confidence-labeled in `data/analysis/model_benchmark_match_audit.csv`. Only `exact`, `normalized_exact`, and `alias_match` are treated as direct model evidence; `family_only` remains proxy evidence.
- Vendor/company aggregation is a companion view to family ranking. It combines a vendor's flagship family signal with an evidence-weighted portfolio mean and keeps the flagship family visible.
- Coverage diagnostics are first-class artifacts. Source tables and family/vendor coverage are inspectable through `source_coverage_diagnostics.csv` and `family_coverage_matrix.csv`.
- Rank-stability intervals are evidence-scaled bootstrap stress tests, not calibrated confidence intervals.
- Forecasts are capped scenarios. See `data/analysis/forecast_input_diagnostics.csv` for raw slopes, fit windows, and cap policy.
- Forecast bands are scenario envelopes across conservative, base, and aggressive settings; they should not be described as statistical uncertainty bands.
- Business-domain views are occupation-derived translation layers. They help non-technical review but do not replace the underlying occupation-level evidence.
- Release-calendar/oracle outputs are demoted exploratory checks. Slow-moving planetary features are dominated by the sample year distribution and are not treated as causal signals.

## Statistical Guardrails

- The analysis is an as-of snapshot at the reference date. Rows dated after that cutoff are excluded from analytical panels.
- Ingestion time, dataset `lastModified`, and report-generation time are not evaluation dates. Undated SWE-bench and Open LLM Leaderboard rows remain usable as cross-sectional evidence but are ineligible for trend estimation.
- Historical LMArena rows are repeated leaderboard snapshots. Scores are normalized within each category and publication snapshot, then collapsed to one model/benchmark/year observation before estimating a frontier.
- Domain histories are benchmark-first panels: each benchmark contributes one annual frontier value. Raw row count never acts as an implicit statistical weight.
- Domain velocity is the median within-benchmark slope among benchmarks observed in at least two years. Domains without comparable longitudinal history receive no borrowed cross-domain slope and are held flat in scenario tables.
- Domain forecasts are bounded scenario extrapolations, not calibrated predictive intervals. `forecast_enabled=false` means the project found insufficient comparable history and deliberately refused to extrapolate.
- Missing public evidence is neutral in min-max components rather than scored as zero capability. Coverage and effective evidence counts are reported separately.
- BLS major-group employment totals are allocated across unmatched detailed occupations instead of being copied onto every occupation. Growth-rate fields are never used as population weights.
- OpenRouter is a current catalog. Grouping current prices by model release year is a cross-sectional cohort diagnostic, not a historical price series; forward price paths therefore use explicit scenario assumptions.

These guardrails reduce pseudo-replication and temporal leakage, but they do not turn heterogeneous public benchmarks into a causal or calibrated forecasting dataset.
