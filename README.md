# AI Capability Signals

How fast is AI moving? Ten measured trends about what AI models can do, what they cost and what it takes to build them, explained for readers who are not specialists and backed by detail for those who are.

Every number on the page is computed from public datasets by a script. Nothing is typed by hand.

## What the page answers

| # | Question | Data |
|---|----------|------|
| 01 | Are models getting smarter? | Epoch Capabilities Index, 270 models |
| 02 | How long a task can AI finish on its own? | METR time horizons |
| 03 | Is AI getting cheaper to use? | Launch prices matched to the capability index |
| 04 | Are models getting bigger? | Parameter counts and how often labs publish them |
| 05 | How much computing goes into training? | Training compute of frontier models |
| 06 | What does it cost to train a top model? | Training cost estimates |
| 07 | Do chips give more for the money? | AI chip launch prices and speed |
| 08 | How big are the machines that train AI? | AI cluster sizes |
| 09 | How far behind are open models? | Open and closed frontiers, lag in months |
| 10 | Who is ahead? | US and Chinese frontiers, best model per lab |

A searchable table of every indexed model and a method section with sources close the page.

## Run locally

Node.js 22.12+ or 24. The committed data file is enough to run the site.

```bash
npm ci
npm run dev
```

```bash
npm run build      # static site in dist/, works from any subdirectory
npm run preview
```

## Refresh the data

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --frozen
npm run data:refresh   # download every source and rebuild public/data/signals.json
npm run data:offline   # rebuild from the last download, no network
```

The pipeline replaces `public/data/signals.json` only after every chapter is computed and the sanity checks pass. Raw downloads are cached in `data/snapshots/` (not in git); their SHA-256 hashes are published inside the data file.

## Method

The same procedure runs on every metric (`pipeline/stats.py`):

1. **Pick the frontier.** A trend is fitted on the models that led at the time of release: record setters, or the ten largest so far. Membership never depends on later releases.
2. **Fit one straight line.** Ordinary least squares, on log10 of the value for anything that grows by multiplication, so a straight line means steady exponential change.
3. **Ask if the pace changed.** Find the split date where two lines fit best and compare the slopes. The search for the split is repeated on 2,000 bootstrap resamples, and a change counts only if 95% of them agree on its direction. The recent part must cover at least two years. If the pace changed, the projection uses the recent part.
4. **Backtest.** Rerun the whole procedure on earlier cutoff dates and compare its projection with the models released 6 to 18 months later. A projection (80% band) is drawn only if the method beat assuming no change and the fit has at least 10 points.

Prices (`pipeline/prices.py`) are attached to index models in a fixed order of trust: a price Epoch AI recorded at the time, the vendor list price before a documented change, today's first party list price, then OpenRouter's pass through price for closed models. Today's third party hosting prices of open models are never projected into the past.

Known source errors are fixed through `pipeline/corrections.json`, each with a reason, and reported on the page. A correction applies only while the source still holds the wrong value.

## Validate

```bash
npm run lint        # eslint and ruff
npm test            # formatting and scale helpers
npm run data:test   # statistics, price matching, published bundle
npm run build
```

## Publishing

`.github/workflows/publish.yml` runs on every push to `main`, every Monday and on demand. It downloads fresh sources, runs every check and deploys `dist/` to GitHub Pages, so the live site is refreshed weekly without data commits. If a source changes shape the run fails and the previous site stays online.

This requires the repository's Pages source to be set to **GitHub Actions** (Settings, Pages). With the older "deploy from branch" setting Pages serves the unbuilt `index.html` and the site is blank.

## Layout

```text
pipeline/           sources, statistics, price matching, chapter builders
tests/              Python tests
src/                React app: one generic chart, one section component, copy in signals.tsx
public/data/        signals.json, the only data the browser loads
.github/workflows/  validation and weekly publish
```

## Sources

| Source | Used for | License |
|--------|----------|---------|
| [Epoch AI models database](https://epoch.ai/data/ai-models) | size, compute, training cost | CC BY 4.0 |
| [Epoch AI Benchmarking Hub](https://epoch.ai/benchmarks) | Epoch Capabilities Index | CC BY 4.0 |
| [Epoch AI hardware database](https://epoch.ai/data/machine-learning-hardware) | chip speed and launch price | CC BY 4.0 |
| [Epoch AI supercomputers database](https://epoch.ai/data/ai-supercomputers) | cluster sizes | CC BY 4.0 |
| [Epoch AI price trends](https://epoch.ai/data-insights/llm-inference-price-trends) | prices recorded 2021 to early 2025 | CC BY 4.0 |
| [METR time horizons](https://metr.org/time-horizons/) | task length | attribution required |
| [llm-prices.com](https://github.com/simonw/llm-prices) | vendor list prices and price changes | no license declared |
| [models.dev](https://models.dev) | current first party list prices | MIT |
| [OpenRouter](https://openrouter.ai/models) | current list prices of closed models | public API |

Many Epoch AI values are researcher estimates, not company disclosures. Code is MIT licensed. Data keeps the terms of each source.

Versions 2 and 3 of this project are in the Git history (`46b17f2` and `d6b4922`).
