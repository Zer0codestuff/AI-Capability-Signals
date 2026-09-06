# Validation

Local validation performed on 5 September 2026, repeated for the UI rebuild on 6 September 2026.

## 6 September 2026, UI rebuild

| Check | Result |
| --- | --- |
| `npm run lint` | ESLint and Ruff pass |
| `npm test` | 16 tests pass, including the three empty state render tests |
| `npm run build` | TypeScript and Vite pass; 259 kB JS (80 kB gzipped), 37 kB CSS, Geist and Geist Mono subsets |
| axe-core 4.10 WCAG 2 A/AA in the dev build | 0 violations; incomplete contrast checks are SVG text, select chevron images, the hero text gradient and elements under the fixed nav |

Browser checks at 1440 px, 390 px and 320 px: no document level horizontal overflow; island nav active state follows the chapters; phone menu opens with the hamburger, closes with Escape and after choosing a chapter, and restores body scrolling; arrow keys move the chart selection and the readout follows; the 80% toggle shrinks the horizon; `#faq` as a direct hash lands with the nav clearance; `prefers-reduced-motion: reduce` shows all reveal blocks and tagline words immediately and removes transitions; the loading skeleton and the not found page render with the same tokens.

## Automated checks

| Check | Result |
| --- | --- |
| `npm run lint` | ESLint and Ruff pass |
| `npm test` | 16 tests pass |
| `npm run data:test` | 17 tests pass, including raw snapshot hashes |
| `npm run build` | TypeScript and Vite production build pass |
| `npm run data:offline` | Rebuild succeeds from hash-verified local snapshots |
| Archive integrity | 115 original files are byte-identical to commit `46b17f2` |

Empty-state regression tests render the components with missing benchmark coverage, insufficient context limits, and an unavailable backtest. The page reports the limitation instead of crashing.

The generated JavaScript bundle is about 235 kB, 74 kB gzipped. The JSON evidence bundle is about 80 kB and is kept formatted for inspection and review. Only the required font subsets are loaded by the browser.

## Browser checks

Checked the running application at 1440 px, 390 px, and 320 px widths. Document width matched viewport width in each case, with no document-level horizontal overflow. Data tables intentionally scroll within their own containers.

Verified:

- Model selection by chart point, selector, and left/right keyboard arrows.
- 50% and 80% success toggles and corresponding task-horizon changes.
- Price benchmark categories and all three token workloads.
- Long-context filtering and recalculated prices.
- Budget control, catalogue search, and zero-result search.
- Total and active parameter views, including missing-value labels.
- Flat task scenario preserves its starting value at both timeline endpoints.
- Cost scenario with a 0% price drop preserves the current estimate.
- Cost scenario with a 75% annual drop over three years follows the compound formula.
- Chapter links and direct hash navigation after asynchronous data loading.
- Reduced-motion preference disables smooth scrolling and transitions.
- Production build loads the story, CSS, JavaScript, data, and fonts from the local origin only.
- No browser errors reported during the final interaction checks.

The automatic WCAG 2 A/AA audit reports zero violations after fixing secondary-text contrast. It leaves one incomplete check category for SVG text contrast, which the audit tool cannot determine automatically. This is not a full accessibility certification.

The embedded browser's full-resolution desktop screenshot capture tiled the visible surface incorrectly. Desktop layout bounds were checked at 1440 CSS pixels, and a 0.4 device-scale capture produced an intact desktop overview. Phone captures use device scale 1. Local screenshots are in the gitignored `artifacts/` directory.

## Preview

The production preview is served by:

```bash
npm run preview -- --port 4173 --strictPort
```

It binds to `127.0.0.1`. The earlier development server was stopped. This is a local review server, not a public deployment.

## Remaining limits

- No new model evaluations were run; the project analyzes published evidence.
- METR's latest included model release is 7 April 2026.
- OpenRouter does not publish embedded benchmark versions or reasoning settings.
- The task trend loses to the last-value baseline in the recorded retrospective check.
- Source licensing should be reviewed before commercial or bulk data redistribution.
- CI configuration is present but has not run remotely because the branch has not been pushed.
