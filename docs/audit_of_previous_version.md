# Audit of the previous version

This document records why the previous version of this repository (package `frontier_ai`,
commit `6a272f0`) was replaced rather than patched. It is kept in the repository because the
failure modes below are the most useful part of the project's history: they are the exact
mistakes that a data project of this shape invites.

Every claim in this document is checkable against the previous code, which remains in git
history at `6a272f0:src/frontier_ai/`.

## Summary verdict

The previous version was large, fast-moving and confident. It produced 49 CSV tables, 44
figures, an 836-line generated report, and a published HTML site. Its ingestion layer was
real: it downloaded genuine public data from Epoch AI, OpenRouter, LMArena, LiveBench,
SWE-bench, Hugging Face, OpenAlex, GitHub and the Anthropic Economic Index.

The analysis layer on top of that ingestion was not sound. The central problem was not a
single bug. It was a structural one: **the project decided what it wanted to say, then built
numerical machinery that would say it.** Composite indices, Monte Carlo simulations, ten-year
forecasts and "probabilities" gave heuristic opinions the surface appearance of measurement.
Several published tables were literal Python constants.

The previous version was also aware of much of this. `docs/statistical_audit.md` documented a
prior self-audit, and the code carried honest caveat strings such as
`"not a calibrated probability model."` This is the most instructive part: the caveats were
added while the indefensible outputs were kept and continued to be headlined. A caveat in a
`method` column does not repair a number that is presented as a finding.

## A. Defects in the analysis

### A1. Critical: a composite index of incomparable metrics presented as a leadership ranking

`frontier_momentum_heuristic_index` was the headline number of the whole project. It was built
by min-max scaling LMArena Bradley-Terry ratings, SWE-bench percentages, Open LLM Leaderboard
fractions, log prices, Hugging Face download counts and release counts onto a common 0-100
range, then combining them with hand-chosen weights:

```python
COMPONENT_WEIGHTS = {
    "performance_component": 0.31,
    "release_velocity_component": 0.16,
    ...
}
```

Three independent problems compound here:

1. The inputs are not measurements of a common quantity. A Bradley-Terry rating is on an
   interval scale defined only up to an affine transform; a resolve rate is a bounded
   proportion; a download count is a heavy-tailed popularity count. Min-max scaling makes them
   look commensurable without making them commensurable.
2. Min-max scaling is defined by the sample extremes, so adding or removing one model family
   rescales every other family's score. The index is not stable under the addition of data.
3. The weights have no source. They are not estimated, elicited, or derived from a stated
   objective. Reporting them in `company_score_methodology.csv` documents them; it does not
   justify them.

The project never tested the assumption the index depends on — that these signals agree about
model ordering. The rebuilt version measures that agreement directly (see
`benchmark_agreement`), finds it is only moderate, and therefore publishes no composite score.

### A2. Critical: "probabilities" and a "bootstrap" that resampled neither data nor models

`company_next_frontier_probabilities.csv` published a column `simulation_win_share`, and the
report headlined values like "58.6% simulation share" for who leads the frontier in ten years.
The generating procedure drew Dirichlet weights over the index components and added Gaussian
noise to the already-computed scores:

```python
sampled_weights = rng.dirichlet(alpha)
noise = rng.normal(0, 2.5 + horizon * 0.18, size=raw_scores.shape[0])
simulated = raw_scores @ sampled_weights + noise
winner = int(np.argmax(simulated))
```

There is no transition model, no base rate, no historical record of lab leadership changes and
no backtest. Nothing in the procedure connects to the future. The noise scale
(`2.5 + horizon * 0.18`) is invented, and it is what determines the answer.

`frontier_score_bootstrap.csv` had the same shape: it perturbed derived component scores rather
than resampling the underlying observations, so it could not estimate sampling uncertainty. The
noise was shrunk by `12 / sqrt(evidence)` where `evidence` was a **sum of raw row counts**
across leaderboard tables. Because leaderboard tables contain repeated snapshots, a family that
appeared in more snapshots received tighter "intervals" — the exact opposite of correct
behaviour.

### A3. Critical: today's price catalogue used as a historical price series

`llm_message_cost_trends.csv` and the cost-forecast tables reported message costs for 2023,
2024, 2025 and 2026. The prices for all four years came from one download of the current
OpenRouter catalogue, subset by model release year:

```python
available = price_panel[price_panel["release_year"].le(year)]
input_price = float(available["input_price_clean"].quantile(q))
```

A current catalogue contains current prices for models that are *still listed*. It says nothing
about what those models cost in 2023, and it silently excludes every model that was withdrawn.
That is survivorship bias plus a units error: a cohort cross-section was labelled a time series.
The code contained a comment acknowledging exactly this, directly above the line that used the
result to fit a slope.

### A4. Critical: multi-year forecasts of a bounded quantity built from mixed units

`domain_capability_forecasts.csv` reported values such as software engineering reaching `97.37`
in 2036 and language reaching `99.5`, plus `estimated_threshold_year` values like `2030.0`. The
underlying "frontier score" pooled arena ratings (roughly 1000-1600, no upper bound), accuracy
percentages (0-100), and fractions (0-1) onto one axis:

```python
if "rating" in unit or "elo" in unit:
    return float(value)
if "fraction" in unit or (0 <= float(value) <= 1 ...):
    return float(value) * 100
```

Slopes were then estimated from as little as one benchmark observed in two years, clipped to an
arbitrary 0-12 points/year, and pushed through exponential gap-closure toward 100. The
"fastest improving domain" headline (agentic terminal work, "12.0 points/year used") came from a
single HTML-scraped benchmark whose raw slope was 22.45 and was clipped to the cap. A number
determined by its own clipping constant is not an estimate.

Macro scenarios showed the same pathology in the open: the 2036 context-window multiplier was
reported as `64` with the accompanying fields `raw=9351.33x capped=True`. When the cap is doing
all the work, the extrapolation has failed and should be withdrawn, not capped.

### A5. High: labor-market conclusions from keyword matching and invented coefficients

`job_replacement_feasibility.csv`, `job_exposure_scores.csv` and
`business_domain_ai_pressure.csv` ranked occupations by "disruption" and "replacement
feasibility". The indices were linear combinations with unexplained coefficients:

```python
jobs["substitution_pressure_index"] = (
    100 * (0.48 * exposure + 0.26 * directive + 0.26 * chance.fillna(chance.median()))
    * (1 - 0.62 * bottleneck)
).round(2)
```

`bottleneck` came from counting substring hits of hand-written keyword lists in O\*NET task
text, normalised by `max(2, len(words) * 0.35)`. The dashboard labelled these rows with
evidence level `observed`.

Ranking real occupations by their probability of being replaced, on this basis, is the most
serious claim in the previous project and had the weakest support. The rebuilt version reports
only metrics that Anthropic itself publishes per occupation from observed usage, and states
plainly that usage composition is not a labor-market outcome.

### A6. High: seven published tables were hardcoded prose

These files sat in `data/analysis/` alongside genuine derived tables, indistinguishable to a
reader:

| File | What it actually was |
|---|---|
| `historical_analogy_index.csv` | Eight technology waves with hand-typed 0-100 scores on seven invented dimensions, cosine-similarity'd against a hand-typed AI vector `[86, 91, 93, 86, 82, 88, 52]`. Published `ai_similarity_score` of `99.17` for cloud/SaaS. |
| `claim_failure_modes.csv` | Five pre-written essays about how the analysis might fail. |
| `forecast_claims.csv` | Four pre-written claims with confidence labels. |
| `cost_external_evidence.csv` | A static bibliography. |
| `domain_workflow_examples.csv` | Eight hand-written workflow tuples. |
| `company_score_methodology.csv` | The weight constants, restated. |
| `deep_analysis_source_registry.csv` | A static source list. |

A further 26 of the 49 tables were data mixed with author-chosen scenario constants. Roughly
16 were reasonably direct transforms of ingested data.

Editorial framing is legitimate. Shipping it as a CSV in the analysis output directory, with a
row count in the manifest, is not.

### A7. High: name-based joins with a fallback that attached the wrong model's score

Benchmark scores were joined to price rows by normalising model names. The normaliser stripped
years and version fragments, then the matcher accepted the first alias overlap in iteration
order rather than the best one. Worse, the `family_only` fallback attached **the highest-scoring
model in the family** to the query:

```python
if family_indexes:
    candidate = max(
        (self.candidates[index] for index in family_indexes),
        key=lambda row: float(row.get("sort_score") or 0),
    )
    return _match("family_only", candidate, query_family)
```

In the published audit table, 813 rows matched only at family level against 504 exact matches.
The price-performance frontier then attached family-best arena ratings to every SKU in the
family, including the cheap small ones, and still computed and ranked a
`price_performance_index`.

The rebuilt version avoids the join almost entirely: OpenRouter now publishes per-model
benchmark indices *inside the same record as the price*, so quality and price come from one row
with no matching step. Where a join is unavoidable, matches are tiered and the tiers that are
not model-identical are excluded from published claims rather than labelled and used anyway.

### A8. High: repeated leaderboard snapshots treated as independent evidence

`build_open_closed_gap_by_category` read the full LMArena history and summed `vote_count`
across every published snapshot, so a model present in 40 snapshots contributed 40 times its
battle count. The same function took `max` of `rating` over all snapshots, which is a maximum
over both models and time, biased upward by however many times a model was re-published.

There is a subtler version of this that the previous version also missed. In the current
upstream `latest` split, some `(model, category, publish_date)` keys carry **six** rows with
different ratings and different variances — an upstream artifact. Taking `max` over those
duplicates biases affected models upward relative to unaffected ones. The rebuilt version
detects the duplicates, reports them as a data-quality finding, and combines them by
inverse-variance weighting with the between-replicate spread folded into the standard error.

### A9. Medium: authoritative source fields replaced by weaker heuristics

Epoch AI's dataset already contains `Model accessibility` with values such as
`Open weights (unrestricted)`, `Open weights (restricted use)`, `API access`, `Unreleased`, plus
a boolean `Open model weights?` and a curated `Frontier model` flag. The previous version
ignored all of these and inferred access class from substring hints on model names:

```python
OPEN_WEIGHT_HINTS = ("meta-llama", "llama", "mistral", "mixtral", "qwen", ...)
```

producing a `likely_open_weight` category that did not need to exist. This is strictly worse
than the source data, and it silently misclassifies any open-weight model from an organisation
not on the hint list.

### A10. Medium: right-censoring ignored in release-count trends

`plot_release_timeline` plotted Epoch model counts per year through 2026. Epoch's dataset is
curated with a lag: at the time of writing it holds 949 models published in 2024, 572 in 2025
and 93 in 2026. Read as a trend, that is a collapse in AI model releases. Read correctly, it is
curation latency plus a partial year. The previous version published the chart without a
censoring boundary.

### A11. Medium: SWE-bench Verified treated as a current signal

The public `SWE-bench/experiments` directory for Verified has 134 submissions, the most recent
dated 2025-12-15. The previous version used it as a live component of a 2026 frontier score. A
leaderboard that has not been updated in seven months can support historical statements only.

### A12. Low, but corrosive: the astrology appendix

The pipeline computed moon phases, approximate Mercury-retrograde windows and geocentric
planetary zodiac signs for model release dates, using the NASA/JPL DE421 ephemeris through
`skyfield`, and rendered a zodiac heatmap.

The stated intent was a demonstration of spurious pattern-finding, which is a genuinely useful
lesson. The execution undermined it: the appendix ran per-feature permutation tests with no
correction for the number of features tested, which is the very error the demonstration was
supposedly about. The rebuilt version keeps the lesson and drops the ephemeris: it tests
calendar features for release clustering, applies Benjamini-Hochberg across the whole family of
tests, and reports how many nominally significant results survive.

## B. Defects in the data handling

- **No schema validation.** Every source was parsed with chained `.get()` calls and silent
  defaults. A renamed upstream column produced empty columns and a successful run.
- **`unified_model_index` was not entity resolution.** It concatenated four namespaces
  (Hugging Face, OpenRouter, Epoch, Open LLM) with no cross-source key and no deduplication,
  under a name that implies the opposite.
- **Brittle scrapers.** Several benchmark domains were populated by regex over HTML pages and
  over a JavaScript bundle whose filename contains a content hash. These had no fixtures and no
  regression tests, so upstream redesigns would degrade the analysis silently.
- **Sampling caps presented as coverage.** Hugging Face ingestion used a hardcoded list of 29
  organisations; Open LLM ingestion capped at 180 directories; OpenAlex and GitHub used keyword
  search with per-query caps. Recall was unknown and unreported.
- **Tiered pricing dropped.** OpenRouter exposes `pricing.overrides`, which raises the
  per-token price above a prompt-length threshold for 44 of 341 current models. The previous
  version read only the base rate, understating long-context costs.
- **Silent failure paths.** Parquet write failures were swallowed (`except Exception:
  parquet_written = False`), and per-source fetch errors were written to a file while the run
  continued and reported success.

## C. Defects in verification

The test suite (642 lines) checked that files existed, that columns were present, that values
fell in plausible ranges, and that certain disclaimer strings had not been deleted. Those are
useful contract tests and they were the strongest part of the previous version.

They did not test any estimator. There was no test that a slope estimator recovers a known
slope, that a confidence interval attains its nominal coverage, that a matcher achieves a
measured precision against labelled pairs, or that a forecast beats a naive baseline. An
arbitrarily wrong number would pass, as long as it was in range and the column existed.

Two brittleness traps were also baked into verification itself: `validate_outputs` raised unless
the OpenRouter catalogue contained `gpt-5.5` **and** `claude-opus-4.7`, and a contract test
asserted the same. When those model IDs are superseded, a correct pipeline run fails. Pinning a
pipeline's success criterion to specific product names guarantees breakage and pressures a
future maintainer to fake the presence of a model.

## D. Documentation inconsistencies

- `docs/methodology.md` stated that domains without longitudinal history are held flat, while
  the generated report stated that they "fall back to the cross-domain median". The code did the
  former. The report text was left stale after the fix.
- `docs/statistical_audit.md` described the cross-domain fallback as removed, while the report
  still described its behaviour.
- `docs/analysis_completion_roadmap.md` listed already-implemented items as future work.
- The report flagged the GPT family as `underobserved=True` with reason
  `below_median_evidence_depth` while simultaneously ranking it first with 120 direct matches
  and a coverage score of 99.5.
- The reference snapshot was 2026-05-15 and the report was generated 2026-07-12, with a
  freshness field claiming 2026-05-15. At the time of this audit the catalogue had moved on by
  two model generations (Claude Opus 5 and the GPT-5.6 family are current), so the report's
  "current frontier models" section was describing superseded products.

## E. What was kept

Not everything was wrong, and the rebuilt version reuses these ideas:

- Provenance discipline: content hashes, run manifests, a source registry, and an explicit
  reference date.
- Coverage and missingness diagnostics as first-class outputs rather than footnotes.
- Refusing to average incompatible benchmarks into one score — stated as a principle in the
  previous README, and now actually enforced, because the composite index that violated it has
  been removed.
- A sample/offline mode so the pipeline can be exercised without network access.
- Contract tests on generated artifacts, now sitting underneath tests of the estimators
  themselves.

## F. The rule the rebuild follows

Every published number must be one of:

1. a direct measurement read from a named source snapshot, with the field named; or
2. a statistic computed by a named estimator, reported with an uncertainty interval whose
   construction is stated, and validated by a test that the estimator recovers known answers on
   synthetic data; or
3. absent.

Anything that would require a fourth category — a plausible-looking number with no estimator
behind it — is not published. Where the honest answer is "the available public data cannot
support this claim", the pipeline says so and emits a refusal record instead of a number. The
`refusals` table in the analysis output is the direct descendant of this audit.
