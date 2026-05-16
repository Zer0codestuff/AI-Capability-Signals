# AI Capability Signals

Hiring-portfolio data project for tracking frontier AI model capability signals with public sources, reproducible ingestion, explicit caveats, and defensible heuristic analysis.

Published report: https://zer0codestuff.github.io/AI-Capability-Signals/

## What This Project Shows

- Public-source ingestion from Epoch AI, OpenRouter, LMArena, LiveBench, SWE-bench Verified, Hugging Face, OpenAlex, GitHub, and Anthropic Economic Index files.
- Normalized model metadata with separate `vendor`, `model_family`, and `product_line` fields so product families are not inferred from vendor names alone.
- Benchmark and pricing tables that keep incompatible metrics separate instead of averaging them into a fake universal model score.
- A deep-analysis layer with heuristic score methodology, sensitivity checks, direct model benchmark matching, vendor portfolio aggregation, source coverage diagnostics, rank-stability stress tests, skeptical failure-mode audits, business-domain implications, release cadence, labor clusters, whole-job replacement feasibility, bounded forecast scenarios, and forecast diagnostics.
- A dashboard-first HTML report that opens with navigable findings, sortable/filterable evidence tables, and drill-down links into the detailed analysis.
- A small sample mode for quick local verification without network access.

## Reproduce

Fast sample run:

```bash
uv run python -m frontier_ai.pipeline --sample
uv run python -m unittest discover -s tests
```

Sample mode writes only to `data/sample/processed/` and, when `--write-reports` is passed, `data/sample/report/`. It does not overwrite the full-run `data/processed/` tables or publishable reports.

Full data refresh:

```bash
uv run python -m frontier_ai.pipeline --overwrite --write-reports
uv run python -m frontier_ai.dataset_factory --overwrite
uv run python -m frontier_ai.deep_analysis --overwrite-sources
uv run python -m unittest discover -s tests
```

Use `--skip-reports` on `frontier_ai.deep_analysis` only when you want the derived tables and figures without rewriting the Markdown/HTML deep-dive report. The core pipeline no longer rewrites the README, tests, or notebook stubs as a side effect.

## Main Outputs

- `data/processed/`: normalized core model, pricing, benchmark, source, and release-calendar tables.
- `data/dataset/`: optional rich dataset package built from broader ecosystem sources.
- `data/analysis/`: analytical tables including heuristic model-family scores, direct benchmark match audits, vendor portfolio scores, source coverage diagnostics, rank stability intervals, skeptical failure modes, business-domain pressure, release cadence, next-frontier simulation shares, leadership audit, open/closed category gaps, price-performance frontiers, job exposure indexes, labor clusters, replacement feasibility, bounded forecasts, and diagnostics.
- `data/analysis/dashboard_key_findings.csv`: dashboard entry points that connect each headline finding to its evidence label and primary artifact.
- `report/`: generated reports when `--write-reports` is used.

Large raw, processed, and rich dataset dumps are intentionally ignored by git. The smaller deep-analysis result tables and deep-analysis figures are kept publishable so reviewers can inspect the actual findings without regenerating the full million-row dataset.

## Publication Policy

This repository publishes code, tests, documentation, small derived analysis CSVs, curated figures, and generated Markdown reports. It intentionally does not version bulk raw/source snapshots under `data/raw/`, full normalized tables under `data/processed/`, or the optional rich dataset under `data/dataset/`.

Third-party source data keeps its original license and terms. See `THIRD_PARTY_DATA.md` and `docs/data_policy.md` before redistributing any generated dataset package.

## Method Notes

- `frontier_momentum_heuristic_index` is a transparent composite index, not a calibrated truth score.
- `company_score_methodology.csv` records components, source signals, transforms, weights, and rationale.
- `company_score_sensitivity.csv` reruns rankings under baseline, no-ecosystem, no-price, and equal-weight variants.
- `model_benchmark_match_audit.csv` separates exact, normalized, alias, family-only, and unmatched benchmark evidence so family proxies are not presented as direct model proof.
- `direct_model_price_performance.csv` restricts deployability rows to direct model-level benchmark evidence and keeps the family-proxy frontier separate.
- `vendor_frontier_scores.csv` adds a true company/vendor portfolio view beside the model-family view.
- `source_coverage_diagnostics.csv` and `family_coverage_matrix.csv` expose freshness, row counts, and missingness before the report leans on rankings.
- `rank_stability_intervals.csv` stress-tests family ranks with evidence-scaled bootstrap draws; these intervals are not calibrated confidence intervals.
- `claim_failure_modes.csv` and `underobserved_family_audit.csv` document assumptions, failure modes, and sparse-evidence families.
- `business_domain_ai_pressure.csv` maps occupation-level labor pressure into business domains while preserving example occupations.
- `release_cadence_by_family.csv` and `release_cadence_by_vendor.csv` summarize visible public product/model cadence.
- `company_next_frontier_probabilities.csv` stress-tests likely 2/5/10-year leaders with Monte Carlo component-weight uncertainty across separate frontier-quality, balanced-execution, and open-ecosystem-upside scenarios. The headline column is `simulation_win_share`, not a calibrated probability.
- `leadership_model_audit.csv` records why the corrected forecast separates raw frontier model leadership from open/cost adoption upside.
- `open_closed_gap_by_category.csv` breaks open-vs-closed model gaps by LMArena category instead of using one generic catchup claim.
- `price_performance_frontier.csv` separates absolute model quality from deployable capability per dollar.
- `job_replacement_feasibility.csv` gates task exposure through physical, trust, regulatory, and task-coverage bottlenecks before calling a job replaceable.
- `labor_cluster_profiles.csv` groups occupations by exposure, bottlenecks, and task-domain structure.
- `forecast_input_diagnostics.csv` records fit windows, raw slopes, and cap policies used for forecasts.
- The release-calendar/oracle appendix is demoted exploratory material and is not part of the headline portfolio claims.

## Known Limitations

- Public source catalogs drift; all current-model and price claims are snapshot-dependent.
- Entity classification is explicit and tested, but still heuristic when source metadata is vague.
- Benchmarks are not directly comparable across tasks, prompts, judge methods, and submission rules.
- Direct model matching is conservative but still name-based; inspect `model_benchmark_match_audit.csv` before treating a row as model-level proof.
- Forecast scenarios are capped transparent assumptions, not calibrated predictions.
- Large generated datasets are excluded from git by design; reproducibility depends on public-source availability and cached local snapshots.
