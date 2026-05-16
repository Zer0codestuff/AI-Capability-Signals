# Analysis Completion Roadmap

This file lists the highest-value additions that would make the Frontier AI analysis more complete, credible, and portfolio-ready. The goal is not to add more charts for volume; it is to add evidence layers that make the current claims harder to challenge.

## Recommended Build Order

1. Add direct model-level benchmark matching.
2. Add vendor/company-level aggregation beside model-family ranking.
3. Add data freshness, coverage, and missingness diagnostics.
4. Add uncertainty bands and rank-stability analysis.
5. Add a skeptical limitations and failure-modes section.
6. Add business/task-domain implication views.
7. Improve the HTML report with interactive affordances.

## P0: Direct Model-Level Benchmark Matching

Current gap:

The price-performance frontier uses a family-level LMArena rating proxy. This is clearly labeled, but a reviewer may still ask which OpenRouter models have direct benchmark evidence.

What to add:

- A direct model matching layer between OpenRouter IDs/names and LMArena, SWE-bench, LiveBench, and Open LLM Leaderboard model names.
- A conservative matching confidence column: `exact`, `normalized_exact`, `alias_match`, `family_only`, `unmatched`.
- A direct-evidence price-performance table that only includes rows with model-level benchmark matches.
- A comparison chart: direct model-level frontier vs family-proxy frontier.

Expected artifacts:

- `data/analysis/model_benchmark_match_audit.csv`
- `data/analysis/direct_model_price_performance.csv`
- `figures/deep_analysis/direct_vs_proxy_price_performance.png`
- Report section: `Direct Model Evidence vs Family Proxy`

Why it matters:

This is the strongest next credibility upgrade. It separates defensible model-level claims from useful but weaker family-level proxy claims.

## P0: Vendor / Company Aggregation

Current gap:

The report ranks model families such as GPT, Claude, Gemini, Qwen, and Mistral. That is methodologically cleaner than pretending they are companies, but the report would benefit from a separate true company/vendor view.

What to add:

- Aggregate model-family signals to vendors: OpenAI, Anthropic, Google, Alibaba, Meta, Mistral, DeepSeek, xAI, Microsoft, Command.
- Preserve both views:
  - model-family frontier score;
  - vendor frontier portfolio score.
- Show where the rankings diverge.

Expected artifacts:

- `data/analysis/vendor_frontier_scores.csv`
- `data/analysis/vendor_score_components.csv`
- `figures/deep_analysis/vendor_frontier_scores.png`
- `figures/deep_analysis/family_vs_vendor_rank_shift.png`
- Report section: `Family Ranking vs Vendor Portfolio Ranking`

Why it matters:

Reviewers often think in terms of companies. Adding this layer makes the analysis easier to understand without weakening the current family-level methodology.

## P1: Data Freshness And Coverage Dashboard

Current gap:

The report uses many public sources, but the reader has to trust that coverage is broad and current. A dashboard would make source quality inspectable.

What to add:

- Source-level captured dates and row counts.
- Missingness by core field: release date, price, access class, benchmark rating, context window, organization/vendor.
- Benchmark coverage by family/vendor.
- Price coverage by family/vendor.
- Labor-data coverage by occupation group.

Expected artifacts:

- `data/analysis/source_coverage_diagnostics.csv`
- `data/analysis/family_coverage_matrix.csv`
- `figures/deep_analysis/source_coverage_dashboard.png`
- `figures/deep_analysis/family_signal_coverage_heatmap.png`
- Report section: `Data Freshness And Coverage`

Why it matters:

This turns a large data pipeline into an auditable data product. It also protects the report from overclaiming where sources are sparse.

## P1: Uncertainty And Rank Stability

Current gap:

The report has sensitivity scenarios and simulation shares, but it could make uncertainty more visible in the charts.

What to add:

- Bootstrap resampling for frontier-family scores.
- Rank interval estimates: best, median, worst rank under resampling.
- Confidence bands around key forecast lines.
- Stability labels based on rank variance, not only sensitivity scenarios.

Expected artifacts:

- `data/analysis/frontier_score_bootstrap.csv`
- `data/analysis/rank_stability_intervals.csv`
- `figures/deep_analysis/frontier_rank_uncertainty.png`
- `figures/deep_analysis/forecast_uncertainty_bands.png`
- Report section: `Uncertainty And Rank Stability`

Why it matters:

This makes the analysis look more mature and prevents readers from treating point estimates as more precise than they are.

## P1: Skeptical Limitations And Failure Modes

Current gap:

The report includes caveats, but a dedicated skeptical section would make it more credible.

What to add:

- A structured table of claims, assumptions, failure modes, and mitigation.
- Explicit cases where the analysis should not be trusted.
- Examples of source drift that would change conclusions.
- A list of under-observed families/vendors.

Expected artifacts:

- `data/analysis/claim_failure_modes.csv`
- `data/analysis/underobserved_family_audit.csv`
- Report section: `Where This Analysis Is Weak`

Why it matters:

Strong portfolio work is not only impressive; it is honest about where it can fail.

## P2: Business And Task-Domain Implications

Current gap:

The labor analysis is occupation-level. A non-technical reviewer may understand it faster if it is mapped to business domains.

What to add:

- Domain grouping for:
  - software engineering;
  - customer support;
  - legal and compliance;
  - marketing and content;
  - finance and analysis;
  - healthcare administration;
  - education;
  - operations and back office.
- Domain-level disruption, augmentation, and replacement feasibility indexes.
- Example workflows for each domain.

Expected artifacts:

- `data/analysis/business_domain_ai_pressure.csv`
- `data/analysis/domain_workflow_examples.csv`
- `figures/deep_analysis/business_domain_pressure_matrix.png`
- Report section: `Business Domain Implications`

Why it matters:

This makes the analysis easier to discuss in interviews and portfolio reviews because it translates model/labor signals into business language.

## P2: Model Release Velocity And Product Cadence

Current gap:

Release velocity is included as a component, but the report does not deeply explain product cadence.

What to add:

- Family/vendor release cadence over time.
- Time between major model releases.
- Recent-release concentration by family.
- Comparison between research releases, API catalog entries, and open-weight releases.

Expected artifacts:

- `data/analysis/release_cadence_by_family.csv`
- `data/analysis/release_cadence_by_vendor.csv`
- `figures/deep_analysis/release_cadence_timeline.png`
- `figures/deep_analysis/recent_release_velocity.png`
- Report section: `Release Velocity And Product Cadence`

Why it matters:

Frontier leadership is not only benchmark quality. Cadence shows execution capacity and market pressure.

## P2: HTML Report Interactivity

Current gap:

The HTML report is visually stronger now, but it is still static.

What to add:

- Sortable tables.
- Collapsible methodology blocks.
- Figure lightbox.
- Download links beside each table.
- Badges for evidence type:
  - `observed`;
  - `direct_match`;
  - `family_proxy`;
  - `scenario`;
  - `speculative`.
- Sticky key-number summary while scrolling.

Expected artifacts:

- Updated `report/deep_frontier_ai_forecast.html`
- Optional `report/assets/report.js`
- Optional `report/assets/report.css`

Why it matters:

Interactive affordances make the report easier to inspect without changing the underlying analysis.

## What Not To Add Yet

- Do not add unsupported prediction claims.
- Do not add a single universal AI score that averages incompatible benchmarks.
- Do not expand astrology/oracle-style analysis into the main report.
- Do not add more generated charts unless each chart answers a specific audit question.
- Do not imply calibrated probabilities unless there is an actual calibration method.

## Best Next Step

Start with direct model-level benchmark matching. It addresses the clearest methodological weakness in the current report and unlocks a cleaner price-performance section.
