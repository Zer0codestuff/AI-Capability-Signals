# AI Capability Signals

## Purpose and architecture

Rebuild the previous technical report as a public-facing, guided data story about AI progress, usage prices, model scale, and conditional futures. Keep observed measurements, source estimates, and scenarios visibly distinct. The existing version is the reference, not the design template.

The rebuild uses a Python data preparation layer and a static React 19 + TypeScript + Vite application. `pipeline/refresh.py` ingests METR, OpenRouter, and Epoch; `pipeline/analysis.py` computes the descriptive trends and historical checks; `pipeline/policy.py` owns METR's version and reliable-range policy. `public/data/story.json` is the browser's only data input. `src/components/` contains the story chapters and native SVG charts. `src/lib/math.ts` owns tier-aware prices, Pareto selection, scales, and scenarios. Exact snapshots are cached locally by hash in `data/snapshots/`.

## Run, build, and test

```bash
npm ci
uv sync --frozen
npm run dev
npm run data:refresh
npm run data:offline
npm run lint
npm test
npm run data:test
npm run build
npm run preview -- --port 4173 --strictPort
```

Node 22.12+ or 24 is required. The committed JSON is enough to build/run the website; offline data rebuilding also needs local raw snapshots from a prior refresh. No secrets, server-side runtime, or live provider calls are needed in the browser. CI validates the committed snapshot without a network refresh.

## Current status

- Original repository cloned at `46b17f2`.
- Rebuild branch: `rebuild/data-story`, published to GitHub on 5 September 2026 together with `main` at commit `71287c3`.
- The user approved a guided story with explorable charts, equal emphasis on measured progress and future scenarios, and a dark minimal design with one accent color.
- Source investigation confirmed that populated Epoch fields can be researcher estimates, not manufacturer disclosures. Artificial Analysis benchmark versions change over time. METR time horizons are human task durations at a specified success rate, not AI runtime or guaranteed autonomy.
- The v3 local rebuild is complete. The 5 September 2026 snapshot has 23 same-version METR records, 124 models with at least one benchmark and usable price, and four illustrative size records. Three embedded TH 1.0 records are excluded.
- The original 115 implementation/report files are byte-identical in `archive/v2/`, excluded from active builds and tests. Legacy imports are not used.
- The site includes measured task horizons, current workload prices and budgets, total/active/unknown parameter examples, conditional task and price scenarios, source notes, and downloadable evidence.
- Recent review fixes: data-driven chart date bounds, centralized reliable-range policy, correct tier fallback, empty price selections, nullable backtest results, explicit active-parameter evidence, direct hash links after data loading, and secondary-text contrast.
- Verified locally: 17 Python tests, 16 TypeScript/render tests, lint, production build, offline reconstruction, browser interactions at desktop and phone widths, and reduced motion. Automatic WCAG 2 A/AA audit has zero violations and one incomplete SVG contrast check category. See `docs/validation.md`.
- Production preview is at `http://127.0.0.1:4173`. No commit, push, or public deployment has been made for the rebuild.
- The `frontend-design` skill (from `anthropics/skills`) is installed at project level with the skills CLI. Its files live in the git-ignored `.agents/skills/` directory; `skills-lock.json` at the root records the source and hash so any agent can restore it with `skills experimental_install`.

## Preferences and constraints

- English project artifacts and interface. No em dashes.
- Large readable charts, little visible prose, details available on demand.
- Use accessible HTML and SVG for data, never image-generated quantities.
- Preserve the previous version through Git and a clearly marked local archive.
- Test data transformations and scenario math. Check the running site at desktop and mobile widths, keyboard interactions, reduced motion, and horizontal bounds.
- Show the local result before any publication. This is a large rebuild, not an automatically pushed small fix.

## Known issues and next steps

- METR's latest included model release is 7 April 2026, regardless of retrieval date. Measurements above 16 hours are explicitly unreliable with the current suite.
- Embedded OpenRouter benchmark version/reasoning settings are unknown; the price comparison is indicative, not a controlled historical capability series.
- The descriptive task trend loses to the last-value baseline in the recorded retrospective check. Keep the conditional scenario framing.
- Retain raw snapshots deliberately for provenance; do not automatically prune them. Formatted JSON is intentional for review.
- Embedded desktop screenshots may tile at full resolution. A 1440 CSS-pixel viewport at device scale 0.4 produces a usable overview; mobile scale 1 works normally. Do not mistake the capture artifact for repeated application content.
- Published on 5 September 2026 as commit `71287c3` on `rebuild/data-story`, fast-forwarded to `main`; GitHub Pages builds the site from `main`. Review source-specific terms before commercial or bulk redistribution.

## Do not

- Do not manufacture GPT-7, GPT-8, or GPT-9 specifications, release dates, IQ scores, or leadership probabilities.
- Do not treat parameter count or training compute as intelligence.
- Do not turn today's price catalogue into a historical price series.
- Do not pool changing benchmark versions into a continuous progress curve.
- Do not turn task success or AI usage into job-replacement claims.
- Do not mistake a populated source field for a value officially disclosed by a developer.
- Do not overwrite unrelated work, push, or deploy without approval.
