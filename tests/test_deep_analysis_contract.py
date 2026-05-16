from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class DeepAnalysisContractTests(unittest.TestCase):
    def test_deep_analysis_tables_exist(self):
        required = [
            "company_frontier_scores",
            "company_score_components",
            "company_score_methodology",
            "company_score_sensitivity",
            "company_next_frontier_probabilities",
            "leadership_model_audit",
            "open_closed_gap_by_category",
            "lmarena_category_leaders",
            "price_performance_frontier",
            "job_exposure_scores",
            "job_replacement_feasibility",
            "labor_cluster_profiles",
            "labor_market_exposure_summary",
            "task_domain_exposure_heatmap",
            "capability_forecasts",
            "capability_frontier_history",
            "forecast_input_diagnostics",
            "historical_analogy_index",
            "forecast_claims",
            "counterintuitive_findings",
            "deep_analysis_source_registry",
            "analysis_manifest",
        ]
        for name in required:
            self.assertTrue((ROOT / "data" / "analysis" / f"{name}.csv").exists(), name)

    def test_company_scores_cover_major_frontier_families(self):
        df = pd.read_csv(ROOT / "data" / "analysis" / "company_frontier_scores.csv")
        families = set(df["model_family"] if "model_family" in df.columns else df["family"])
        for family in ["GPT", "Claude", "Gemini", "DeepSeek", "Qwen", "Llama"]:
            self.assertIn(family, families)
        self.assertTrue(df["frontier_momentum_heuristic_index"].between(0, 100).all())
        self.assertIn("sensitivity_label", df.columns)
        self.assertGreaterEqual(df["evidence_count"].max(), 1000)

    def test_company_score_methodology_and_sensitivity(self):
        methodology = pd.read_csv(ROOT / "data" / "analysis" / "company_score_methodology.csv")
        self.assertAlmostEqual(methodology["baseline_weight"].sum(), 1.0, places=6)
        sensitivity = pd.read_csv(ROOT / "data" / "analysis" / "company_score_sensitivity.csv")
        self.assertEqual(set(sensitivity["scenario"]), {"baseline", "no_ecosystem", "no_price", "equal_weight"})
        self.assertTrue(sensitivity["heuristic_index"].between(0, 100).all())
        probabilities = pd.read_csv(ROOT / "data" / "analysis" / "company_next_frontier_probabilities.csv")
        self.assertEqual(set(probabilities["scenario"]), {"frontier_quality", "balanced_lab_execution", "open_ecosystem_upside"})
        self.assertEqual(set(probabilities["horizon_years"]), {2, 5, 10})
        self.assertIn("simulation_win_share", probabilities.columns)
        self.assertNotIn("win_probability", probabilities.columns)
        for _, group in probabilities.groupby(["scenario", "horizon_years"]):
            self.assertAlmostEqual(group["simulation_win_share"].sum(), 1.0, places=4)
        audit = pd.read_csv(ROOT / "data" / "analysis" / "leadership_model_audit.csv")
        self.assertIn("audit_note", audit.columns)
        fq_10y = probabilities[(probabilities["scenario"].eq("frontier_quality")) & (probabilities["horizon_years"].eq(10))]
        self.assertNotEqual(fq_10y.sort_values("simulation_win_share", ascending=False).iloc[0]["model_family"], "Mistral")
        frontier = pd.read_csv(ROOT / "data" / "analysis" / "price_performance_frontier.csv")
        self.assertIn("price_performance_frontier", frontier.columns)
        self.assertIn("quality_proxy_level", frontier.columns)
        self.assertGreater(frontier["price_performance_frontier"].astype(bool).sum(), 0)
        gap = pd.read_csv(ROOT / "data" / "analysis" / "open_closed_gap_by_category.csv")
        self.assertIn("open_closed_best_gap", gap.columns)
        self.assertIn("comparison_note", gap.columns)
        self.assertGreaterEqual(len(gap), 3)

    def test_job_exposure_scores_are_task_grounded(self):
        df = pd.read_csv(ROOT / "data" / "analysis" / "job_exposure_scores.csv")
        self.assertGreaterEqual(len(df), 700)
        for col in [
            "observed_exposure",
            "capability_exposure_index",
            "substitution_pressure_index",
            "augmentation_index",
            "human_bottleneck_index",
            "near_term_disruption_index",
            "full_job_automation_feasibility_index",
            "dominant_outcome",
        ]:
            self.assertIn(col, df.columns)
        self.assertGreater(df["task_count"].notna().sum(), 500)
        self.assertGreater(df["directive_share"].notna().sum(), 400)
        self.assertGreater(df["bls_employment"].notna().sum(), 500)
        replacement = pd.read_csv(ROOT / "data" / "analysis" / "job_replacement_feasibility.csv")
        self.assertTrue(replacement["full_job_automation_feasibility_index"].between(0, 100).all())
        clusters = pd.read_csv(ROOT / "data" / "analysis" / "labor_cluster_profiles.csv")
        self.assertGreaterEqual(len(clusters), 5)
        self.assertIn("cluster_label", clusters.columns)

    def test_forecasts_have_three_scenarios_and_horizons(self):
        df = pd.read_csv(ROOT / "data" / "analysis" / "capability_forecasts.csv")
        self.assertEqual(set(df["scenario"]), {"conservative", "base", "aggressive"})
        self.assertEqual(set(df["horizon_years"]), {2, 5, 10})
        required_metrics = {
            "frontier_training_compute_multiplier",
            "frontier_output_price_factor",
            "open_weight_lmarena_gap_remaining",
            "share_of_us_occupation_tasks_materially_touched",
        }
        self.assertTrue(required_metrics.issubset(set(df["metric"])))
        context = df[df["metric"].eq("frontier_context_window_multiplier")]
        self.assertLessEqual(context["value"].max(), 128)
        diagnostics = pd.read_csv(ROOT / "data" / "analysis" / "forecast_input_diagnostics.csv")
        self.assertIn("raw_log10_slope_per_year", diagnostics.columns)
        self.assertIn("scenario_assumed_log10_slope_per_year", diagnostics.columns)
        self.assertIn("fallback_or_cap_policy", diagnostics.columns)

    def test_figures_and_report_exist(self):
        for name in [
            "company_frontier_scores.png",
            "company_score_component_stack.png",
            "company_score_evidence_scatter.png",
            "job_exposure_top.png",
            "job_exposure_wage_scatter.png",
            "labor_task_forecast.png",
            "forecast_scenario_dashboard.png",
            "cost_forecast_scenarios.png",
            "open_closed_catchup.png",
            "historical_analogy_index.png",
            "task_domain_exposure_heatmap.png",
            "company_next_frontier_probabilities.png",
            "leadership_scenario_matrix.png",
            "open_closed_gap_by_category.png",
            "open_closed_category_levels.png",
            "price_performance_frontier.png",
            "price_context_rating_map.png",
            "labor_cluster_profiles.png",
            "labor_outcome_mix.png",
            "job_replacement_feasibility.png",
        ]:
            path = ROOT / "figures" / "deep_analysis" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)
        self.assertTrue((ROOT / "report" / "deep_frontier_ai_forecast.md").exists())
        self.assertTrue((ROOT / "report" / "deep_frontier_ai_forecast.html").exists())


if __name__ == "__main__":
    unittest.main()
