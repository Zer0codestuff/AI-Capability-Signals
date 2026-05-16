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
- Forecasts are capped scenarios. See `data/analysis/forecast_input_diagnostics.csv` for raw slopes, fit windows, and cap policy.
- Release-calendar/oracle outputs are demoted exploratory checks. Slow-moving planetary features are dominated by the sample year distribution and are not treated as causal signals.
