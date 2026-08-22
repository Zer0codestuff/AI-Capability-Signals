from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReportContractTests(unittest.TestCase):
    def test_readme_demotes_oracle_and_names_limits(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("Oracle heatmap", text)
        self.assertNotIn("Weird appendix", text)
        self.assertIn("Known Limitations", text)
        self.assertIn("heuristic", text.lower())

    def test_main_report_has_no_oracle_headline_when_present(self):
        report = ROOT / "report" / "frontier_ai_analysis.md"
        if not report.exists():
            self.skipTest("generated report has not been written")
        text = report.read_text(encoding="utf-8")
        self.assertNotIn("The Weird Appendix", text)
        self.assertNotIn("Oracle appendix finds", text)
        self.assertNotIn("_No rows available._", text)
        self.assertNotIn("sample://", text)

    def test_deep_report_uses_current_methodology_labels(self):
        report = ROOT / "report" / "deep_frontier_ai_forecast.md"
        if not report.exists():
            self.skipTest("deep report has not been written")
        text = report.read_text(encoding="utf-8")
        self.assertIn("Model Family Frontier Score", text)
        self.assertIn("Dashboard Snapshot", text)
        self.assertIn("dashboard_key_findings.csv", text)
        self.assertIn("How To Read This Report", text)
        self.assertIn("Capability Domains", text)
        self.assertIn("Domain Improvement Velocity", text)
        self.assertIn("Domain Capability Forecasts", text)
        self.assertIn("domain_benchmark_results.csv", text)
        self.assertIn("domain_capability_forecasts.csv", text)
        self.assertIn("company_score_component_stack.png", text)
        self.assertIn("forecast_scenario_dashboard.png", text)
        self.assertIn("Direct Model Evidence vs Family Proxy", text)
        self.assertIn("LLM Cost Per Message vs Fixed Task Cost", text)
        self.assertIn("llm_message_cost_trends.csv", text)
        self.assertIn("fixed_task_cost_curves.csv", text)
        self.assertIn("cost_divergence_scenarios.csv", text)
        self.assertIn("Family Ranking vs Vendor Portfolio Ranking", text)
        self.assertIn("Data Freshness And Coverage", text)
        self.assertIn("Uncertainty And Rank Stability", text)
        self.assertIn("Where This Analysis Is Weak", text)
        self.assertIn("Business Domain Implications", text)
        self.assertIn("Release Velocity And Product Cadence", text)
        self.assertIn("direct_model_price_performance.csv", text)
        self.assertIn("rank_stability_intervals.csv", text)
        self.assertIn("simulation_win_share", text)
        self.assertIn("not calibrated confidence intervals", text)
        self.assertNotIn("Company Frontier Score", text)
        self.assertNotIn("win_probability", text)
        self.assertNotIn("| nan", text.lower())

    def test_deep_html_report_has_premium_layout(self):
        report = ROOT / "report" / "deep_frontier_ai_forecast.html"
        if not report.exists():
            self.skipTest("deep HTML report has not been written")
        text = report.read_text(encoding="utf-8")
        for marker in [
            "class='topbar'",
            "class='topnav'",
            "id='progress-bar'",
            "class='hero'",
            "class='kicker'",
            "class='meta-chips'",
            "class='badge-legend'",
            "class='analysis-dashboard'",
            "class='dashboard-tiles'",
            "class='dashboard-links'",
            "data-table-filter",
            "class='figure-panel'",
            "class='table-card'",
            "data-sortable='true'",
            "data-lightbox='figure'",
            "Download table index",
            "assets/report.js",
            "assets/report.css",
            "Back to top",
        ]:
            self.assertIn(marker, text)
        self.assertGreaterEqual(text.count("class='figure-panel'"), 15)
        self.assertGreaterEqual(text.count("class='table-card'"), 10)
        self.assertIn("domain benchmark velocity", text)
        self.assertNotIn("win_probability", text)
        self.assertTrue((ROOT / "report" / "assets" / "report.js").exists())
        self.assertTrue((ROOT / "report" / "assets" / "report.css").exists())

    def test_pdf_report_is_current_and_reproducible(self):
        pdf = ROOT / "report" / "deep_frontier_ai_forecast.pdf"
        renderer = ROOT / "scripts" / "render_report_pdf.py"
        self.assertTrue(pdf.exists())
        self.assertGreater(pdf.stat().st_size, 1_000_000)
        self.assertTrue(renderer.exists())
        self.assertIn("render_report_pdf.py", (ROOT / "Makefile").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
