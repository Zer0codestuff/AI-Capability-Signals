# AI Capability Signals: Scaling, Prices, Open Weights, and Benchmark Caveats

Reference date: 2026-05-15. Data captured at: 2026-05-16T07:05:53+00:00.

This repository is a hiring-portfolio research project focused on reproducible public-data ingestion and defensible caveats. It combines public model metadata, API catalogs and benchmark datasets to analyze frontier AI systems without collapsing unrelated signals into one fake universal score.

## Executive Findings

- The processed project currently includes **3,525 Epoch AI model rows**, **356 OpenRouter API catalog rows**, **9,988 LMArena leaderboard rows**, and **134 SWE-bench Verified public submissions**.
- GPT-5.5 appears in the public API catalog with a **1,050,000 token context window**, **$5.00/1M input tokens** and **$30.00/1M output tokens**.
- Claude Opus 4.7 appears with a **1,000,000 token context window**, **$5.00/1M input tokens** and **$25.00/1M output tokens**.
- Across the current OpenRouter snapshot, the median listed output price is **$10.00/1M tokens** for closed/API-classified models and **$0.51/1M tokens** for likely open-weight models. This is not a capability-adjusted claim; it is a public-catalog pricing snapshot.
- The largest disclosed training compute row in the normalized Epoch AI slice is **Grok 4** at approximately **5.0e+26 FLOP**.

## Surprising Results

- The public API catalog snapshot shows likely open-weight models with a median output price about **19.6x lower** than closed/API-classified models, before any quality adjustment.
- Context length has become decoupled from frontier branding: several catalog models list **2M-token context windows**, while GPT-5.5 and Claude Opus 4.7 sit around the 1M-token tier in this snapshot.
- The highest selected LMArena rows are dominated by closed frontier systems, but the pricing table shows that open-weight/API-hosted alternatives compete on a very different cost curve.

## Key Current Models

| model                             | openrouter_id                  | release_date   |   context_window |   input_usd_per_1m |   output_usd_per_1m | source_url                                           |
|:----------------------------------|:-------------------------------|:---------------|-----------------:|-------------------:|--------------------:|:-----------------------------------------------------|
| Anthropic: Claude Opus 4          | anthropic/claude-opus-4        | 2025-05-22     |           200000 |              15    |                75   | https://openrouter.ai/anthropic/claude-opus-4        |
| Anthropic: Claude Opus 4.1        | anthropic/claude-opus-4.1      | 2025-08-05     |           200000 |              15    |                75   | https://openrouter.ai/anthropic/claude-opus-4.1      |
| Anthropic: Claude Opus 4.5        | anthropic/claude-opus-4.5      | 2025-11-24     |           200000 |               5    |                25   | https://openrouter.ai/anthropic/claude-opus-4.5      |
| Anthropic: Claude Opus 4.6        | anthropic/claude-opus-4.6      | 2026-02-04     |          1000000 |               5    |                25   | https://openrouter.ai/anthropic/claude-opus-4.6      |
| Anthropic: Claude Opus 4.6 (Fast) | anthropic/claude-opus-4.6-fast | 2026-04-07     |          1000000 |              30    |               150   | https://openrouter.ai/anthropic/claude-opus-4.6-fast |
| Anthropic: Claude Opus 4.7        | anthropic/claude-opus-4.7      | 2026-04-16     |          1000000 |               5    |                25   | https://www.anthropic.com/news/claude-opus-4-7       |
| Anthropic: Claude Opus 4.7 (Fast) | anthropic/claude-opus-4.7-fast | 2026-05-12     |          1000000 |              30    |               150   | https://www.anthropic.com/news/claude-opus-4-7       |
| OpenAI: GPT-5                     | openai/gpt-5                   | 2025-08-07     |           400000 |               1.25 |                10   | https://openrouter.ai/openai/gpt-5                   |
| OpenAI: GPT-5 Chat                | openai/gpt-5-chat              | 2025-08-07     |           128000 |               1.25 |                10   | https://openrouter.ai/openai/gpt-5-chat              |
| OpenAI: GPT-5 Mini                | openai/gpt-5-mini              | 2025-08-07     |           400000 |               0.25 |                 2   | https://openrouter.ai/openai/gpt-5-mini              |
| OpenAI: GPT-5 Nano                | openai/gpt-5-nano              | 2025-08-07     |           400000 |               0.05 |                 0.4 | https://openrouter.ai/openai/gpt-5-nano              |
| OpenAI: GPT-5 Codex               | openai/gpt-5-codex             | 2025-09-23     |           400000 |               1.25 |                10   | https://openrouter.ai/openai/gpt-5-codex             |

## Scaling and Accessibility

![Notable AI model releases by accessibility class](../figures/model_release_timeline.png)

![Training compute trend](../figures/training_compute_trend.png)

The important professional caveat is disclosure. Closed frontier systems often report less about parameters, tokens and compute than open-weight research releases. The transparency index below measures field coverage in the public Epoch AI dataset; it should be interpreted as public disclosure coverage, not model quality.

![Transparency index by vendor](../figures/transparency_index.png)

## Price, Context and Catalog Reality

![Context window versus output price](../figures/context_window_price.png)

Large context windows have become a product dimension in their own right. The chart deliberately separates context and price from benchmark claims: a one-million-token context window is useful only when the model can retrieve and reason over that context reliably, which must be tested separately.

Top context windows in the OpenRouter snapshot:

| canonical_model                             | openrouter_id                             | model_family   |   context_window |   input_usd_per_1m |   output_usd_per_1m |
|:--------------------------------------------|:------------------------------------------|:---------------|-----------------:|-------------------:|--------------------:|
| Meta: Llama 4 Scout                         | meta-llama/llama-4-scout                  | Llama          |         10000000 |               0.08 |                 0.3 |
| xAI: Grok 4.20 Multi-Agent                  | x-ai/grok-4.20-multi-agent                | Grok           |          2000000 |               2    |                 6   |
| xAI: Grok 4.20                              | x-ai/grok-4.20                            | Grok           |          2000000 |               1.25 |                 2.5 |
| OpenAI: GPT-5.4 Pro                         | openai/gpt-5.4-pro                        | GPT            |          1050000 |              30    |               180   |
| OpenAI: GPT-5.5 Pro                         | openai/gpt-5.5-pro                        | GPT            |          1050000 |              30    |               180   |
| OpenAI GPT Latest                           | ~openai/gpt-latest                        | GPT            |          1050000 |               5    |                30   |
| OpenAI: GPT-5.5                             | openai/gpt-5.5                            | GPT            |          1050000 |               5    |                30   |
| OpenAI: GPT-5.4                             | openai/gpt-5.4                            | GPT            |          1050000 |               2.5  |                15   |
| Owl Alpha                                   | openrouter/owl-alpha                      | Other          |          1048756 |               0    |                 0   |
| Google: Gemini 3.1 Pro Preview Custom Tools | google/gemini-3.1-pro-preview-customtools | Gemini         |          1048756 |               2    |                12   |
| Google: Gemini 2.5 Pro                      | google/gemini-2.5-pro                     | Gemini         |          1048576 |               1.25 |                10   |
| Google: Gemini 3.1 Pro Preview              | google/gemini-3.1-pro-preview             | Gemini         |          1048576 |               2    |                12   |

## Benchmarks Without Fake Omniscience

The project does not average unrelated benchmarks into a single artificial "AI score." LMArena, SWE-bench and LiveBench measure different things and are stored separately.

![Top SWE-bench Verified submissions](../figures/swebench_top.png)

| system_name                                                   | model                                                                                                      | org                |   score |   resolved |   total | source_url                                                                                                               |
|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:-------------------|--------:|-----------:|--------:|:-------------------------------------------------------------------------------------------------------------------------|
| live-SWE-agent + Claude 4.5 Opus medium (20251101)            | claude-opus-4-5-20251101                                                                                   | UIUC               |    79.2 |        396 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251215_livesweagent_claude-opus-4-5             |
| Sonar Foundation Agent + Claude 4.5 Opus                      | claude-opus-4-5                                                                                            | Sonar              |    79.2 |        396 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251205_sonar-foundation-agent_claude-opus-4-5   |
| TRAE + Doubao-Seed-Code                                       | Doubao-Seed-Code, Doubao-Seed-1.6                                                                          | ByteDance          |    78.8 |        394 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250928_trae_doubao_seed_code                    |
| OpenHands + Claude Opus 4.5                                   | OpenHands + Claude Opus 4.5                                                                                |                    |    77.6 |        388 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251127_openhands_claude-opus-4-5                |
| live-SWE-agent + Gemini 3 Pro Preview (2025-11-18)            | gemini-3-pro-preview                                                                                       | UIUC               |    77.4 |        387 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251120_livesweagent_gemini-3-pro-preview        |
| EPAM AI/Run Developer Agent v20250719 + Claude 4 Sonnet       | claude-sonnet-4-20250514                                                                                   | EPAM Systems, Inc. |    76.8 |        384 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250804_epam-ai-run-claude-4-sonnet              |
| Atlassian Rovo Dev (2025-09-02)                               | claude-sonnet-4-20250514, gpt-5                                                                            | Atlassian          |    76.8 |        384 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250902_atlassian-rovo-dev                       |
| ACoder                                                        | claude-4-sonnet, claude-4.1-opus, gpt-5-0807-global, gemini-2.5-pro-06-17                                  | ACoder             |    76.4 |        382 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250819_ACoder                                   |
| Warp                                                          | gpt-5, claude-sonnet-4-20250514                                                                            | Warp               |    75.6 |        378 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250901_warp                                     |
| TRAE + Claude Sonnet 4 + Opus 4 + Sonnet 3.7 + Gemini 2.5 Pro | claude-4-sonnet-20250522, claude-4-opus-20250522, claude-3-7-sonnet-20250219, gemini-2.5-pro-preview-06-05 | TRAE               |    75.2 |        376 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250612_trae                                     |
| Sonar Foundation Agent + Claude 4.5 Sonnet                    | claude-sonnet-4-5                                                                                          | Sonar              |    74.8 |        374 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251103_sonar-foundation-agent_claude-sonnet-4-5 |
| Harness AI                                                    | claude-sonnet-4-20250514                                                                                   | Harness            |    74.8 |        374 |     500 | https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20250731_harness_ai                               |

![LMArena rating distribution by access class](../figures/lmarena_access_distribution.png)

Top rows from the selected LMArena latest snapshots:

| benchmark      | category   | model                         |   score | votes   | access_class   |
|:---------------|:-----------|:------------------------------|--------:|:--------|:---------------|
| LMArena webdev | webdev     | claude-opus-4-7-thinking      | 1580.75 |         | closed_or_api  |
| LMArena webdev | webdev     | claude-opus-4-7               | 1559.28 |         | closed_or_api  |
| LMArena webdev | webdev     | claude-sonnet-4-6             | 1556.58 |         | closed_or_api  |
| LMArena text   | text       | gpt-5.5                       | 1553.57 |         | closed_or_api  |
| LMArena webdev | webdev     | claude-opus-4-6               | 1550.87 |         | closed_or_api  |
| LMArena text   | text       | claude-opus-4-6               | 1546.36 |         | closed_or_api  |
| LMArena webdev | webdev     | claude-opus-4-6-thinking      | 1545.57 |         | closed_or_api  |
| LMArena text   | text       | claude-opus-4-6-thinking      | 1542.96 |         | closed_or_api  |
| LMArena text   | text       | claude-opus-4-7-thinking      | 1540.53 |         | closed_or_api  |
| LMArena text   | text       | gemini-3.1-pro-preview        | 1538.3  |         | closed_or_api  |
| LMArena webdev | webdev     | gpt-5.5-xhigh (codex-harness) | 1536.76 |         | closed_or_api  |
| LMArena text   | text       | gpt-5.5-high                  | 1534.61 |         | closed_or_api  |

## Open-Weight vs Closed

The strongest result is not that open-weight models always beat closed models or vice versa. The real story is structural:

- Open-weight systems disclose more internals and make replication/inspection easier.
- Closed systems dominate some API/product leaderboards, but opacity makes scaling analysis harder.
- API price competition is compressing fast enough that the cost frontier can move independently from the capability frontier.
- The "open source" label is too coarse. This repo uses `open_weight`, `likely_open_weight`, `closed_or_api`, and `unknown` rather than treating all public weights as fully open-source systems.

## Limits

- Some current frontier model details are intentionally undisclosed by vendors.
- OpenRouter is an API catalog, not an official benchmark authority.
- Public benchmark formats change; extraction code preserves raw snapshots and avoids fabricating missing scores.
- Entity classification is explicit but still heuristic when source metadata is vague.

## Reproduce

```bash
uv run python -m frontier_ai.pipeline --write-reports
uv run python -m unittest discover -s tests
```
