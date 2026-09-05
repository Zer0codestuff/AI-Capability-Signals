# Methodology

## What this project measures

There is no universal AI intelligence number here. Four questions use separate evidence:

| Question | Evidence | Meaning |
| --- | --- | --- |
| What size of task can an agent handle? | METR Time Horizon 1.1 | Estimated human task duration at a specified success threshold |
| What does a fixed token workload cost? | OpenRouter current catalogue | Arithmetic on current list rates, not actual invoices |
| How large is a model? | Selected Epoch records and their publications | Parameters, including unavailable values and explicitly documented active counts |
| What if an observed rate continued? | Local fit and declared assumptions | Conditional scenarios, not probabilities |

## Retrieval and provenance

`pipeline/refresh.py` downloads three public sources. Each exact response is SHA-256 hashed and cached locally using deterministic gzip. The published JSON contains the URL, source page, hash, and retrieval time. The raw snapshots are not redistributed.

Schema validation happens before output replacement. Missing benchmark values are never filled with another model's scores. Models have stable source IDs; there is no cross-source family-name join.

The displayed retrieval date is not a release date, benchmark run date, or observation date. The METR chart uses the model release dates supplied by METR, including retrospective evaluations. The price data is a current cross-section.

## Task horizons

The source is `https://metr.org/assets/benchmark_results_1_1.yaml`. The parser requires `METR-Horizon-v1.1` in both the document and each included model record. The download also contains three TH 1.0 records; those are excluded instead of stitching incompatible versions together.

Each model retains:

- The source model ID and release date.
- `p50_horizon_length` and `p80_horizon_length`, each with estimate and interval endpoints.
- Source agent configuration descriptions.

METR's published methodology describes its intervals as 95% bootstrap intervals. That confidence level is documented by METR, not exposed as a separate field in the YAML. The site attributes the intervals to METR rather than calculating replacements.

The solid chart line connects successive record-high point estimates. Other observations remain visible. This line is a visual guide to the observed records, not a fitted growth model. The selected model's interval is drawn separately.

Time is measured by how long a human expert would take, not by an AI's elapsed runtime. The tasks primarily concern software engineering, ML, and cybersecurity. This is not a workforce sample or a test of arbitrary real-world autonomy.

METR's documented reliable measurement boundary is 960 minutes. `pipeline/policy.py` is the single policy source; its value travels into the JSON and charts. Measurements beyond the boundary remain visible with a warning, but are excluded from the corresponding trend fit.

Reference: [METR time horizons](https://metr.org/time-horizons/) and [limitations](https://metr.org/notes/2026-01-22-time-horizon-limitations/).

## Conditional task scenarios

For each success threshold independently:

1. Use eligible TH 1.1 measurements at or below the documented range.
2. Select the highest point on each release date to avoid duplicate-date weighting.
3. Fit `log2(minutes) = intercept + slope * elapsed_days` using ordinary least squares.
4. Also calculate a median pairwise slope as a robustness comparison.
5. Anchor scenarios to the highest eligible measured point at its actual model release date.

The historical rate is descriptive and sensitive to model selection, release-date density, retrospective reevaluation, agent configuration, and measurement error. It is not METR's own published trend estimate.

`scenario_minutes = anchor_minutes * 2 ** (days * pace / doubling_days)`

- Flat: pace 0.
- Half pace: pace 0.5.
- Same pace: pace 1.

These constants define sensitivity cases. They are not samples from a probability distribution, fitted slowdown estimates, or confidence bounds. No probability is assigned to them. The horizon control spans 0 to 24 months after the measured baseline; it does not invent a current baseline or GPT release date. A month is 365.25 / 12 days. A displayed day means 24 hours of equivalent human task time, not an eight-hour workday.

### Historical check

For each later release date, fit using only strictly earlier dates if at least six prior points exist. Predict the held-out measurement and compare absolute log2 error against the highest value on the last prior date. Equal-date observations never enter each other's training history.

This is a retrospective next-release check using today's revised benchmark snapshot. It is not a vintage-data backtest and does not validate a fixed two-year forecast. The app reports both errors, including when the fitted trend loses. No empirical prediction band is constructed from these few errors.

## Current price comparisons

Rows must have text-only output, positive prompt and completion list prices, and at least one usable Artificial Analysis index. Exclude special routing variants, free variants, records with an additional per-request charge, and missing-score records. Exclusion counts account for all catalogue rows.

For each chart, keep only the selected index and models that support the stated total context and published maximum output length.

`cost = request_count * (input_tokens * input_rate + output_tokens * output_rate)`

The applicable price override has the greatest `min_prompt_tokens` not exceeding the input length. Each missing override rate falls back to its base rate, not the previous tier's rate.

The three workloads are deliberately chosen token budgets:

| Workload | Input | Output | Requests |
| --- | ---: | ---: | ---: |
| Short exchange | 1,000 | 500 | 1,000 |
| Document analysis | 16,000 | 2,000 | 1,000 |
| Long context | 128,000 | 4,000 | 1,000 |

Workload names are editorial descriptions, not measured task executions. Additional reasoning tokens, caching, tool calls, retries, taxes, and model-specific tokenization differences are not estimated. An equal token budget is not equal task success.

The frontier keeps models for which no cheaper model has an equal or higher selected score. The budget control finds the highest available index under the declared workload budget, breaking ties by cost. It is not a model recommendation.

“General” is the Artificial Analysis Intelligence Index. Coding and agentic indices are separate measures. No average is calculated. OpenRouter does not expose their evaluation version or reasoning configuration, so the site does not claim a fully controlled comparison or construct a historical series.

## Parameter examples

The four illustrative releases are Llama 3.1 405B, DeepSeek V3, Qwen3 235B, and GPT-4.1. They were chosen to explain model size, mixture-of-experts active parameters, and unknown values. They are not the latest-model ranking.

Total values come from exact Epoch records. Active values 37B and 22B are retained only while the source notes explicitly attach those counts to “active” or “activated”. Missing values remain null, not zero. Epoch's record confidence is retained but is not a proof that a particular number was officially disclosed.

## Conditional price scenarios

`future_cost = current_workload_cost * (1 - assumed_annual_drop) ** years`

The reference workload is 1,000 requests, each with 16,000 input and 2,000 output tokens. The default annual drop of 25% is a declared illustrative assumption, not a trend fitted to historical prices. Input and output prices change at the same rate. The horizon is one to three years from retrieval.

The calculation holds token volume fixed, not capability, availability, usage demand, or reliability. Spending may increase even if token prices fall. It is not a forecast of any future GPT's price.
