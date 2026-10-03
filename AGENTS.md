# AI Capability Signals

## Purpose and architecture

A public, single page data story that answers "how fast is AI moving?" for non experts while staying technical and detailed: intelligence, task length, usage price, model size, training compute, training cost, chips, clusters, open versus closed, countries and labs.

- `pipeline/` (Python, standard library plus PyYAML) downloads public sources and writes `public/data/signals.json`, the only file the browser loads.
  - `sources.py`: source registry, download, hash, gzip snapshot cache in `data/snapshots/` (git ignored).
  - `audit.py`: the guards that decide which source values may be used (vetted only, no unit slips, no unreviewed tenfold leaps). `reviewed.json` lists leaps a person confirmed.
  - `stats.py`: the one procedure used everywhere: OLS (log10 for multiplicative metrics), bootstrap ranges, data driven split to test a change of pace, rolling origin backtest.
  - `prices.py`: attaches a launch price to capability index models, with `aliases.json` for names that do not match and `launch_prices.json` for launch prices restored by hand with a source.
  - `build.py`: one function per chapter, sanity `check()`, CLI. `corrections.json` holds documented fixes of source errors.
- `src/` (React 19, TypeScript, Vite, no chart library): `components/Chart.tsx` is the single generic SVG chart, `components/Signal.tsx` the section template, `signals.tsx` all chapter copy built from the data, `lib/format.ts` and `lib/scale.ts` the helpers.
- `.github/workflows/validate.yml` checks branches and pull requests. `publish.yml` refreshes data, tests, builds and deploys to GitHub Pages on push to `main`, weekly and on demand.

## Run, build, test

```bash
npm ci && uv sync --frozen
npm run dev
npm run data:refresh     # online
npm run data:offline     # replay cached snapshot
npm run lint && npm test && npm run data:test && npm run build
```

Node 22.12+ or 24, uv for Python.

## Current status (3 October 2026)

- Version 4 is a full rewrite, published on 3 October 2026: `rebuild/v4` was pushed and fast forwarded into `main` at `c825e2c` with the user's approval, and the Pages source was switched to GitHub Actions. The first `publish.yml` run passed and the live site at https://gabrielemonni.me/AI-Capability-Signals/ loads all eleven charts with no console errors. Started in Antigravity (thread `49a22d68`), finished in Claude Code after the Antigravity quota ran out.
- The user chose: all eight metric groups, English only, projections with uncertainty bands plus a historical check labelled "if the trend continues", keep the dark black and lime style, rewrite on a new branch with weekly automatic refresh, show the local result before publishing.
- Data through 28 September 2026: 270 indexed models, 162 with a price, 3,614 models in the Epoch database.
- Headline results: capability index +14.0 points a year; METR task length doubles every 4.1 months (METR publishes 129 days, this fit gives 125); training compute x4.7 a year; training cost x2.8 a year; chips x1.4 a year in computing per dollar; GPT-4 level price down 682 times since March 2023; open models 7 months behind closed.
- Verified locally: 17 Python tests, 3 TypeScript tests, eslint, ruff, production build (256 kB JS, 80 kB gzipped), desktop 1440 px and phone 390 and 320 px without horizontal overflow, hover tooltips, keyboard stepping, projection toggle, table search and sort, no console errors.

## Data audit of 3 October 2026 (after the user caught BaGuaLu)

The user pointed out that BaGuaLu's 174 trillion parameters, shown as the largest model, is the size the system could handle, not a model that was trained. The paper's abstract and Epoch AI's own note confirm it. The cause was taking records over every row of the Epoch AI database. The audit found and fixed the same class of problem elsewhere:

- Records, trends and quoted facts now use only vetted rows: curated notable set and rated Confident or Likely. This removed BaGuaLu, M6-10T and Wu Dao 2.0 from the size chart, and speculative values such as Grok 4 from the compute and cost records (the cost record went from $388M to $366M).
- Chip value mixed number formats: 32 bit for old chips, 8 and 4 bit for new ones. It now uses the fastest 32 or 16 bit speed for every chip. The pace went from x1.5 to x1.4 a year, in line with Epoch AI's published figure.
- Epoch AI's price file holds early 2025 prices for some models. Gemini 1.5 Pro and Flash (May 2024) carried prices from after Google's cuts; launch prices are restored with Google's announcements as sources. Hosting prices of open models are now dated to the day they were recorded.
- The capability index dated GPT-4 Turbo (Nov 2023) to January 2024; corrected to 6 November 2023.
- New automatic guards: unit slip detection on parameter notes, a hold on recent tenfold leaps until reviewed, a name versus date check on the index. Everything set aside is listed in the page's method section.
- The parameter and training cost trends now both fail their backtest, so neither has a projection.

## Problems found in version 3 and fixed

- The live site was blank: Pages deploys the `main` branch root, so it serves the unbuilt `index.html` pointing at `/src/main.tsx`.
- The original questions were lost: size was four examples, there was no cost or price trend over time, intelligence was 23 METR points, the Epoch Capabilities Index was unused, data stopped at 5 September.
- The trend lost to a last value baseline with no consequence. Now a projection is drawn only when the backtest beats the baseline.
- 28 MB of archived version 2 files lived in the active tree. Removed; they remain in Git history.

## Preferences and constraints

- English artifacts and interface. No em dashes anywhere. Avoid hyphenated compounds in page copy. Sentence case headings.
- Dark minimal style: page `#000000`, cards `#181818`, single lime accent `#bcf17b`, Geist for text, Geist Mono for numbers and axes, weights up to 600, no italics, flat surfaces, motion with `cubic-bezier(0.32,0.72,0,1)`, reveals through IntersectionObserver, reduced motion respected.
- Chart series colors in fixed order: lime `#bcf17b`, blue `#3987e5`, magenta `#d55181`, amber `#c98500`. This set passed the dataviz skill validator on the card surface for color vision deficiency and contrast (lime is outside its lightness band on purpose, it is the brand color). A series keeps its color everywhere. Text never takes a series color except the lime highlight in answers.
- Plain answer first, chart second, uncertainty always visible, details on demand in folds. Every chart has a table view.
- Keep Zer0codestuff as the sole commit author, no co-author trailers. The publish workflow makes no commits for that reason.
- Show the local result before publishing. Large changes need approval before push.

## Known issues and next steps

- The workflows use action versions built for Node.js 20 (`actions/checkout@v4`, `setup-node@v4`, `configure-pages@v5`, `upload-pages-artifact@v3`, `deploy-pages@v4`, `astral-sh/setup-uv@v6`). GitHub runs them on Node.js 24 with a deprecation warning. Bump them when newer majors are confirmed.
- The weekly schedule (Mondays 06:17 UTC) has not fired yet. Check the first scheduled run.
- Price coverage is 162 of 270 models. Retired models with no public keyless price record (for example Qwen 3.8 Max) are missing. Artificial Analysis would fill gaps but needs an API key.
- METR's latest measured model is from April 2026, cluster data stops in July 2025, training cost estimates are sparse after 2023. The page says so in each chapter.
- The parameter and training cost trends fail their backtest, so those charts have no projection. Expected, not a bug.
- Closed model prices from Epoch AI's undated file are placed at the release date and assumed unchanged since launch. Only the two Gemini cuts are corrected; others may exist (Mistral Large is a likely one, below the levels tracked).
- The OpenAI page for the GPT-4o mini launch price could not be fetched by script. The reviewed entry rests on Epoch AI's recorded price matching the announced one.
- The capability index trend reads "no clear change" although 2023 was flat: the early period has too few records for the split test. Revisit as data accumulates.
- No `og:image` yet.
- The working copy lives in `~/.gemini/antigravity/scratch/AI-Capability-Signals`. Consider moving it to a regular projects folder.

## Do not

- Do not type numbers into copy. Every figure must come from `signals.json`.
- Do not let an unvetted, speculative or unrated source value set a record, enter a trend or appear in a sentence. Do not take a maximum over raw database rows.
- Do not mix number formats or definitions inside one series. One yardstick per chart.
- Do not date a price earlier than the day it is known to have applied.
- Do not add a name to `reviewed.json` or a price to `launch_prices.json` without a source.
- Do not draw a projection the backtest did not earn, and do not present projections as forecasts.
- Do not assign today's third party hosting price of an open model to its release date.
- Do not treat parameter count, compute or cost as intelligence.
- Do not invent specifications, dates or scores for unreleased models.
- Do not add a second y axis, cycle series colors, or add decorative colors.
- Do not push large or risky changes, or change repository settings, without approval. Small scoped fixes can be committed and pushed directly.
- Do not add bot or agent authorship to commits.
