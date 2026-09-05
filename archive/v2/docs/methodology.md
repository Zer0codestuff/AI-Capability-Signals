# Methodology

## Design rules

1. A published number is either a direct measurement from a named source field, or the output of a named estimator in `aicap.stats` accompanied by an uncertainty interval.
2. When the data cannot support a claim, the pipeline records a refusal and emits no number. Capping an implausible extrapolation and publishing the cap is not an option.
3. No composite score across incommensurable metrics. Whether such a score would be coherent is itself measured (`benchmark_agreement`); the project publishes the measurement instead of the composite.
4. Prefer authoritative source fields over inferred ones (Epoch accessibility, OpenRouter `hugging_face_id`, LMArena licence).
5. Tests must recover known answers on synthetic data. Schema checks alone are not enough.

## Reference date

Derived at runtime as the *freshest* source observation horizon. Sources that lag behind are listed
in the freshness table; when a source is more than 90 days behind, analyses that depend on it refuse
present-tense claims rather than quietly using stale data. A run can pin a date with
`--reference-date` for reproduction.

This replaces the previous version's hardcoded `REFERENCE_DATE = "2026-05-15"`, which went stale
silently.

## Estimators

| Quantity | Estimator | Interval |
|---|---|---|
| Disclosure rate | Binomial proportion | Wilson score |
| Disclosure difference (open vs closed) | Difference of proportions | Two-sample bootstrap; BH across fields |
| Compute growth | OLS with HC3 SEs; Theil–Sen alongside | HC3 / distribution-free |
| Compute forecast | Linear trend, only if backtest beats last-value | Empirical backtest error quantiles |
| Price median | Sample median | Percentile bootstrap over models |
| Price ~ quality + openness | Multiple OLS, HC3 | HC3 + model-level bootstrap on openness coefficient |
| Benchmark agreement | Kendall τ-b | Model-level paired bootstrap |
| Arena gap | Difference of published ratings | Combined published SEs (conservative) |
| Catch-up lag | Kaplan–Meier median | Right-censored levels retained |
| Calendar clustering | Year-preserving permutation | Benjamini–Hochberg across the feature family |
| Surface contrast (usage) | Paired mean difference | Occupation-level paired bootstrap |

## What is deliberately not done

- Min-max scaling of heterogeneous metrics into a weighted index.
- Monte Carlo over index weights presented as a probability of future leadership.
- Treating a current price catalogue as a historical price series.
- Cross-domain slope borrowing for domains without longitudinal history.
- Name-based joins that fall back to the family's highest-scoring model.
- Summing repeated leaderboard snapshots as independent evidence.
- Keyword-built labour-replacement indexes.

## Sources

See the source registry in `aicap.config.SOURCES` and `data/analysis/sources.csv` after a run. Each entry states what the source is used for and its known limitations.
