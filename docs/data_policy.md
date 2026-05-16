# Data Policy

This repository is intended to be reviewed as a hiring portfolio, so git should contain code, tests, documentation, lockfiles, and small reproducibility fixtures rather than hundreds of megabytes of generated data.

## Versioned

- Source code under `src/`.
- Tests under `tests/`.
- Documentation under `README.md`, `docs/`, `appendix/`, and generated Markdown reports when useful.
- Dependency and build metadata such as `pyproject.toml`, `uv.lock`, and `Makefile`.
- Reviewable deep-analysis CSVs under `data/analysis/*.csv`.
- Deep-analysis report figures under `figures/deep_analysis/*.png`.

## Not Versioned

- `data/raw/`: public-source caches and downloaded snapshots.
- `data/processed/`: regenerated full-run normalized tables.
- `data/sample/`: isolated no-network sample outputs.
- `data/dataset/`: optional rich dataset package.
- `data/analysis/*.parquet`: regenerated analytical binary tables.
- `figures/` outside the curated `figures/deep_analysis/*.png` report set.
- Local environments and Python caches.

Use `uv run python -m frontier_ai.pipeline --sample` for a quick no-network smoke run. Sample mode writes to `data/sample/processed/`, so it does not overwrite full-run `data/processed/` tables. Use the full commands in `README.md` when large regenerated outputs are needed.
