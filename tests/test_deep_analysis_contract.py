from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class DeepAnalysisContractTests(unittest.TestCase):
    def test_deep_analysis_tables_exist(self):
        required = [
            "company_frontier_scores",
            "dashboard_key_findings",
            "company_score_components",
            "company_score_methodology",
            "company_score_sensitivity",
            "model_benchmark_match_audit",
            "direct_model_price_performance",
            "vendor_frontier_scores",
            "vendor_score_components",
            "source_coverage_diagnostics",
            "family_coverage_matrix",
            "frontier_score_bootstrap",
            "rank_stability_intervals",
            "claim_failure_modes",
            "underobserved_family_audit",
            "business_domain_ai_pressure",
            "domain_workflow_examples",
            "release_cadence_by_family",
            "release_cadence_by_vendor",
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

    def test_dashboard_key_findings_are_auditable_entry_points(self):
        df = pd.read_csv(ROOT / "data" / "analysis" / "dashboard_key_findings.csv")
        self.assertGreaterEqual(len(df), 10)
        for col in ["section", "question", "headline", "metric", "evidence_level", "primary_artifact", "priority_order"]:
            self.assertIn(col, df.columns)
        self.assertTrue(set(df["evidence_level"]).issubset({"observed", "direct_match", "family_proxy", "scenario", "speculative"}))
        self.assertIn("company_frontier_scores.csv", set(df["primary_artifact"]))
        self.assertTrue(df["priority_order"].is_monotonic_increasing)

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

    def test_direct_model_matching_vendor_and_coverage_contracts(self):
        match_audit = pd.read_csv(ROOT / "data" / "analysis" / "model_benchmark_match_audit.csv")
        self.assertTrue({"exact", "normalized_exact", "alias_match", "family_only", "unmatched"}.issuperset(set(match_audit["match_confidence"])))
        self.assertTrue(match_audit["direct_model_match"].isin([True, False]).all())
        self.assertGreater(match_audit["direct_model_match"].astype(bool).sum(), 50)
        direct = pd.read_csv(ROOT / "data" / "analysis" / "direct_model_price_performance.csv")
        self.assertIn("direct_lmarena_rating", direct.columns)
        self.assertIn("quality_proxy_level", direct.columns)
        self.assertGreater(len(direct), 10)
        self.assertFalse(direct["quality_proxy_level"].eq("family_level_proxy").any())
        vendors = pd.read_csv(ROOT / "data" / "analysis" / "vendor_frontier_scores.csv")
        self.assertIn("vendor_frontier_portfolio_score", vendors.columns)
        self.assertIn("OpenAI", set(vendors["vendor"]))
        self.assertIn("Google", set(vendors["vendor"]))
        self.assertTrue(vendors["vendor_frontier_portfolio_score"].between(0, 100).all())
        coverage = pd.read_csv(ROOT / "data" / "analysis" / "family_coverage_matrix.csv")
        self.assertIn("coverage_score", coverage.columns)
        self.assertIn("direct_benchmark_match_count", coverage.columns)
        self.assertTrue(coverage["coverage_score"].between(0, 100).all())
        self.assertGreaterEqual(len(coverage), 8)

    def test_uncertainty_limits_domains_and_cadence_contracts(self):
        intervals = pd.read_csv(ROOT / "data" / "analysis" / "rank_stability_intervals.csv")
        self.assertIn("rank_stability_label", intervals.columns)
        self.assertTrue(set(intervals["rank_stability_label"]).issubset({"stable", "moderate", "unstable"}))
        self.assertTrue((intervals["best_rank"] <= intervals["worst_rank"]).all())
        bootstrap = pd.read_csv(ROOT / "data" / "analysis" / "frontier_score_bootstrap.csv")
        self.assertGreaterEqual(bootstrap["draw"].nunique(), 500)
        failures = pd.read_csv(ROOT / "data" / "analysis" / "claim_failure_modes.csv")
        self.assertIn("failure_mode", failures.columns)
        self.assertGreaterEqual(len(failures), 5)
        domains = pd.read_csv(ROOT / "data" / "analysis" / "business_domain_ai_pressure.csv")
        expected_domains = {
            "software_engineering",
            "customer_support",
            "legal_compliance",
            "marketing_content",
            "finance_analysis",
            "healthcare_administration",
            "education",
            "operations_back_office",
        }
        self.assertEqual(set(domains["business_domain"]), expected_domains)
        self.assertTrue(domains["disruption_index"].between(0, 100).all())
        workflows = pd.read_csv(ROOT / "data" / "analysis" / "domain_workflow_examples.csv")
        self.assertEqual(set(workflows["business_domain"]), expected_domains)
        cadence = pd.read_csv(ROOT / "data" / "analysis" / "release_cadence_by_family.csv")
        self.assertIn("cadence_label", cadence.columns)
        self.assertGreaterEqual(len(cadence), 8)

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
            "direct_vs_proxy_price_performance.png",
            "vendor_frontier_scores.png",
            "family_vs_vendor_rank_shift.png",
            "source_coverage_dashboard.png",
            "family_signal_coverage_heatmap.png",
            "frontier_rank_uncertainty.png",
            "forecast_uncertainty_bands.png",
            "business_domain_pressure_matrix.png",
            "release_cadence_timeline.png",
            "recent_release_velocity.png",
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
