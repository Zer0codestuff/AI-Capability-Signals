# Data dictionary

Tables are written to `data/analysis/` by `uv run aicap`. Float columns are rounded to 6 d.p. at write time; in-memory frames keep full precision.

## Always present

| Table | Grain | Contents |
|---|---|---|
| `sources.csv` | source | Registry: URL, licence, used_for, caveat |
| `source_freshness.csv` | source | Latest observation date and whether it binds the reference date |
| `data_quality_findings.csv` | check | Blocking / constraining / informational findings |
| `refusals.csv` | claim | Claims declined, with reason and unblock condition |
| `analysis_manifest.csv` | table | Inventory of tables written this run |

## Disclosure

| Table | Grain |
|---|---|
| `disclosure_by_year.csv` | year × field |
| `disclosure_by_weights_class.csv` | weights_class × field |
| `disclosure_open_vs_closed.csv` | field (BH-corrected difference) |
| `disclosure_by_vendor.csv` | vendor |

## Compute scaling

| Table | Grain |
|---|---|
| `compute_frontier_by_year.csv` | year |
| `compute_trend_estimates.csv` | specification × estimator |
| `compute_selection_sensitivity.csv` | restriction |
| `compute_forecast_backtest.csv` | horizon |
| `compute_frontier_forecast.csv` | horizon (only if backtest beats baseline) |

## Prices

| Table | Grain |
|---|---|
| `price_distribution_by_weights.csv` | group × metric |
| `price_tier_impact.csv` | snapshot |
| `price_quality_frontier.csv` | model (Pareto flag) |
| `cheapest_at_quality_threshold.csv` | threshold |
| `price_spread_at_matched_quality.csv` | quality band |
| `price_quality_regression.csv` | regression term |

## Benchmark agreement

| Table | Grain |
|---|---|
| `benchmark_panel.csv` | model |
| `benchmark_crosswalk.csv` | source name → catalogue id |
| `benchmark_pair_agreement.csv` | benchmark pair |
| `benchmark_rank_instability.csv` | model |
| `composite_score_verdict.csv` | one row |

## Open-weight lag

| Table | Grain |
|---|---|
| `arena_frontier_by_date.csv` | leaderboard date |
| `open_weights_lag.csv` | closed-frontier date |
| `open_weights_lag_summary.csv` | one row (Kaplan–Meier) |
| `open_closed_gap_trend.csv` | methodology regime |
| `open_closed_gap_by_category.csv` | arena × category |

## Usage composition

| Table | Grain |
|---|---|
| `usage_surface_summary.csv` | surface × metric |
| `usage_top_occupations.csv` | surface × occupation |
| `usage_surface_contrast.csv` | surface pair |
| `usage_major_group_composition.csv` | surface × SOC major group |

## Other

| Table | Grain |
|---|---|
| `calendar_release_tests.csv` | calendar feature |
| `calendar_release_counts.csv` | feature × bucket |
| `swebench_frontier_history.csv` | submission |
| `swebench_top_systems.csv` | submission |
| `swebench_scaffold_spread.csv` | model tag |
