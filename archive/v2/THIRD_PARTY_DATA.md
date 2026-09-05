# Third-party data

This project downloads public datasets and APIs at runtime. Source data is **not** redistributed in the repository. Digests of the snapshots used for a run are recorded in `report/run_manifest.json`.

| Source | URL | Licence / terms | Used for |
|---|---|---|---|
| Epoch AI notable models | https://epoch.ai/data/all_ai_models.csv | CC-BY (see Epoch terms) | Publication dates, compute, disclosure, accessibility |
| OpenRouter models API | https://openrouter.ai/api/v1/models | Public API; OpenRouter terms | Listed prices, context, vendor benchmark indices |
| LMArena leaderboard dataset | https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset | See dataset card | Bradley–Terry ratings with published variances |
| SWE-bench Verified experiments | https://github.com/SWE-bench/experiments | MIT | Dated agent resolve rates |
| Anthropic Economic Index | https://huggingface.co/datasets/Anthropic/EconomicIndex | CC-BY | Observed usage composition by occupation |

Do not treat analysis outputs as a substitute for the upstream datasets. Re-fetch from the URLs above for any use that needs the original records.
