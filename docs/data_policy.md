# Data policy

This repository publishes code, tests, documentation, small derived analysis CSVs, curated figures and generated reports.

It does **not** version:

- `data/raw/` — content-addressed source snapshots (SHA-256 digests are in `report/run_manifest.json`)
- `data/interim/` — parsed parquet frames reproducible from raw

Third-party source data keeps its original licence and terms. See `THIRD_PARTY_DATA.md` before redistributing any generated dataset package.

Analysis tables in `data/analysis/` are derived aggregates and may be published. They do not contain raw conversation text, personal data or bulk source dumps.
