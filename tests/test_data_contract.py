from pathlib import Path
import unittest
import pandas as pd
from frontier_ai.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROCESSED = ROOT / "data" / "sample" / "processed"


class DataContractTests(unittest.TestCase):
    def test_processed_core_tables_exist(self):
        for name in [
            "models_master.csv",
            "models_openrouter_normalized.csv",
            "models_epoch_normalized.csv",
            "sources.csv",
            "oracle_pattern_tests.csv",
        ]:
            self.assertTrue((ROOT / "data" / "processed" / name).exists(), name)

    def test_every_master_row_has_source_url(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "models_master.csv")
        self.assertTrue(df["source_url"].notna().all())
        self.assertTrue((df["source_url"].astype(str).str.len() > 0).all())
        self.assertIn("model_family", df.columns)
        self.assertIn("product_line", df.columns)

    def test_key_current_models_present(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "key_current_models.csv")
        ids = " ".join(df["openrouter_id"].astype(str).str.lower())
        self.assertIn("gpt-5.5", ids)
        self.assertIn("claude-opus-4.7", ids)

    def test_prices_are_per_million_tokens(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "pricing_openrouter.csv")
        numeric = df["output_usd_per_1m"].dropna()
        self.assertGreaterEqual(len(numeric), 2)
        self.assertGreater(numeric.max(), 1)

    def test_full_processed_artifacts_are_not_sample_outputs(self):
        processed = ROOT / "data" / "processed"
        master = pd.read_csv(processed / "models_master.csv")
        benchmarks = pd.read_csv(processed / "benchmarks_all.csv")
        self.assertGreaterEqual(len(master), 1_000)
        self.assertGreaterEqual(len(benchmarks), 1_000)
        for path in processed.glob("*.csv"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("sample://", text, path.name)

    def test_swebench_respects_reference_date(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "benchmarks_swebench_verified.csv")
        dated = df["submission"].astype(str).str.extract(r"^(\d{8})_")[0].dropna()
        self.assertTrue((pd.to_datetime(dated, format="%Y%m%d") <= pd.Timestamp("2026-05-15")).all())

    def test_rich_dataset_manifest_when_present(self):
        manifest_path = ROOT / "data" / "dataset" / "dataset_manifest.csv"
        if not manifest_path.exists():
            self.skipTest("rich dataset has not been generated")
        manifest = pd.read_csv(manifest_path)
        required = {
            "huggingface_models",
            "openalex_ai_papers",
            "github_ai_repositories",
            "swebench_instance_outcomes",
            "unified_model_index",
        }
        self.assertTrue(required.issubset(set(manifest["table"])))
        self.assertGreater(manifest["rows"].sum(), 1_000_000)
        for _, row in manifest.iterrows():
            self.assertTrue((ROOT / row["csv_path"]).exists(), row["csv_path"])
            self.assertTrue((ROOT / row["parquet_path"]).exists(), row["parquet_path"])

    def test_sample_pipeline_runs_without_network(self):
        run_pipeline(sample=True, skip_plots=True)
        df = pd.read_csv(SAMPLE_PROCESSED / "models_master.csv")
        self.assertIn("model_family", df.columns)
        self.assertIn("GPT", set(df["model_family"]))
        self.assertTrue((SAMPLE_PROCESSED / "sources.csv").exists())


if __name__ == "__main__":
    unittest.main()
