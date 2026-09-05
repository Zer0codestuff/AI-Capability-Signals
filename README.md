# AI Capability Signals

A guided, interactive data story about what AI can do, what it costs to use, and what might come next.

The site separates published measurements, calculations, and conditional scenarios. It does not invent specifications for GPT-7, GPT-8, or GPT-9.

## Run locally

Node.js 22.12+ or 24 and npm are required for the website. The committed data bundle is enough to run it.

```bash
npm ci
npm run dev
```

Open the local address printed by Vite. The server binds to loopback, not the public network.

```bash
npm run build
npm run preview
```

The production output is `dist/`. It is a static site, supports subdirectory hosting, and needs no backend or browser API credentials. No deployment is configured by this rebuild.

## The story

1. **Progress:** human task durations at 50% or 80% success, using METR Time Horizon 1.1 only. Explore models and their published intervals.
2. **Cost:** current list prices for declared token workloads versus separate Artificial Analysis indices supplied by OpenRouter. Explore the price frontier, budget, and catalogue.
3. **Scale:** total and active parameters for four illustrative releases. Missing values stay missing.
4. **Futures:** flat, slower, and historical-pace task-horizon scenarios, plus an explicitly assumed token-price scenario. These are not calibrated predictions.

The interface is dark, with one accent color. Charts use accessible SVG, keyboard selection, model selectors, and alternative data tables. Details are in native expandable sections. Fonts are self-hosted; there is no analytics or third-party browser request.

## Refresh the evidence

Install [uv](https://docs.astral.sh/uv/) and then:

```bash
uv sync --frozen
npm run data:refresh
```

This downloads METR, OpenRouter, and Epoch AI, validates the source schemas, and replaces `public/data/story.json` only after successful analysis. It records source URLs, retrieval times, SHA-256 hashes, exclusions, intervals, and trend checks.

Exact gzip-compressed source snapshots are retained locally in `data/snapshots/`, outside Git. Rebuild without network access:

```bash
npm run data:offline
```

This verifies each cached source hash and preserves its retrieval time. A fresh clone has the published JSON but not the raw snapshots, so offline *rebuilding* first needs a successful online refresh. Historical snapshots are deliberately retained for provenance and never automatically deleted. Manage disk retention only after deciding which past runs you need.

## Validate

```bash
npm run lint
npm test
npm run data:test
npm run build
```

Tests cover known-answer estimators, strict past-date backtests, benchmark-version isolation, source coverage, intervals, parameter missingness, cost tiers, workload limits, Pareto selection, and scenario math. CI validates the committed snapshot without depending on live upstream availability. Raw-hash checks skip on fresh clones without local snapshots.

## Layout

```text
pipeline/           Python ingestion, policies, and calculations
src/components/     Story chapters and SVG charts
src/lib/            Shared pure calculations and viewport hook
public/data/        Published evidence bundle
tests/              Python data and estimator tests
docs/               Methodology, sources, and validation notes
archive/v2/         Original technical report, preserved intact
```

## Current data and limits

The 5 September 2026 snapshot includes 23 comparable METR measurements, 124 catalogue models with at least one usable benchmark and price, and four model-size examples. Individual price charts have fewer models because benchmark and workload coverage differ.

METR's latest model in this snapshot was released on 7 April 2026. Downloading that dataset today does not make it a measurement of today's frontier. METR warns that task horizons above 16 hours are unreliable with the current suite.

OpenRouter does not expose the version or reasoning configuration of its embedded Artificial Analysis scores. Treat those charts as indicative catalogue comparisons, not a controlled experiment, universal intelligence scale, or historical capability series.

The task-horizon trend does not beat a last-observation baseline in this snapshot's retrospective next-release check. The site reports that result and presents conditional paths instead of forecast probabilities.

See [methodology](docs/methodology.md), [sources and terms](THIRD_PARTY_DATA.md), and [validation](docs/validation.md).

## Previous version

Version 2 is preserved under `archive/v2/` and in Git at `46b17f2`. It is not part of the new build, data pipeline, or tests. Its previously published conclusions have not been silently carried into the new site.

Code is MIT licensed. Source data and fonts retain their respective terms.
