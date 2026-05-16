# Third-Party Data Notice

This repository contains code, documentation, small derived analysis tables, and curated figures. It does not version the bulk raw datasets, full normalized tables, or optional rich dataset package.

Third-party data keeps its original license and terms. Do not treat this repository's code license as a license for upstream datasets.

## Source Notes

| Source | Project use | Redistribution note |
|---|---|---|
| Epoch AI model data | Historical model metadata and scaling fields | Epoch AI states the data is free to use, distribute, and reproduce with Creative Commons Attribution credit. |
| LMArena leaderboard dataset | Category leaderboard ratings and open-vs-closed comparisons | Hugging Face lists the dataset license as `cc-by-4.0`; attribution is required. |
| Anthropic Economic Index | Occupation exposure, task penetration, and labor companion inputs | Hugging Face lists the dataset license as `mit`; verify the dataset card before redistributing snapshots. |
| OpenAlex | Research and model-mention metadata | OpenAlex states its data is CC0 and free to use and distribute. |
| BLS public data | Wage and employment companion fields where available | BLS states its published material is public domain, with source citation requested. |
| O*NET data | Task text and occupation bottleneck features | O*NET content is CC BY 4.0 and requires attribution, including noting modifications. |
| OpenRouter public API | Current model catalog, prices, and context windows | Treat raw API snapshots as terms-sensitive; publish derived summaries and reproducible fetch code rather than bulk raw API dumps. |
| GitHub API | Repository-level ecosystem indicators | Avoid redistributing bulk GitHub-derived rows; keep aggregate or derived indicators and source links. |
| Hugging Face model metadata | Model ecosystem indicators | Metadata can reference many model repos with different licenses; review source-specific licenses before publishing bulk dumps. |

## Publishing Rule

For the public GitHub portfolio, commit the reproducible pipeline and the small derived analysis outputs under `data/analysis/*.csv`. Keep `data/raw/`, `data/processed/`, `data/dataset/`, `data/sample/`, and generated parquet files out of Git.
