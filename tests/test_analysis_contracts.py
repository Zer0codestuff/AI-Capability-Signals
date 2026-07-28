"""Analysis contracts: no composite scores, refusals are recorded, censoring is respected."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aicap.analysis import compute_scaling, disclosure, open_weights_lag
from aicap.analysis.benchmark_agreement import _verdict
from aicap.provenance import RunContext
from aicap.sources.epoch import censoring_boundary
from aicap.stats import kaplan_meier_median


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_epoch(years: list[int], counts: list[int]) -> pd.DataFrame:
    rows = []
    for year, count in zip(years, counts, strict=True):
        for index in range(count):
            rows.append(
                {
                    "model": f"m-{year}-{index}",
                    "organization": "Lab",
                    "publication_date": pd.Timestamp(year=year, month=6, day=1),
                    "publication_year": year,
                    "date_precision": "day",
                    "domain": "Language",
                    "parameters": 1e9 if index % 2 == 0 else np.nan,
                    "training_compute_flop": 10 ** (20 + (year - 2018) * 0.5) if index % 3 == 0 else np.nan,
                    "training_tokens": np.nan,
                    "training_hardware": "A100" if index % 4 == 0 else np.nan,
                    "accessibility_raw": "Open weights (unrestricted)" if index % 2 == 0 else "API access",
                    "open_weights_raw": "Yes" if index % 2 == 0 else "No",
                    "weights_class": "open_weights" if index % 2 == 0 else "closed_weights",
                    "vendor": "Lab",
                    "family": "Other",
                    "country": "US",
                    "frontier_flag": False,
                    "discloses_parameters": index % 2 == 0,
                    "discloses_training_compute_flop": index % 3 == 0,
                    "discloses_training_tokens": False,
                    "discloses_training_hardware": index % 4 == 0,
                    "discloses_accessibility_raw": True,
                }
            )
    return pd.DataFrame(rows)


class CensoringTests(unittest.TestCase):
    def test_detects_monotonic_decline(self) -> None:
        frame = _synthetic_epoch([2020, 2021, 2022, 2023, 2024], [40, 60, 80, 50, 20])
        self.assertEqual(censoring_boundary(frame), 2023)

    def test_complete_series_has_no_censoring(self) -> None:
        frame = _synthetic_epoch([2020, 2021, 2022], [20, 30, 40])
        self.assertEqual(censoring_boundary(frame), 2023)


class DisclosureContractTests(unittest.TestCase):
    def test_open_vs_closed_excludes_circular_field_and_corrects(self) -> None:
        frame = _synthetic_epoch([2019, 2020, 2021, 2022], [40, 40, 40, 40])
        context = RunContext(reference_date_override="2022-12-31")
        tables = disclosure.build(frame, context)
        comparison = tables["disclosure_open_vs_closed"]
        self.assertFalse(comparison.empty)
        self.assertNotIn("accessibility_raw", set(comparison["field"]))
        self.assertIn("q_value_bh", comparison.columns)
        self.assertIn("significant_after_bh", comparison.columns)


class ComputeScalingContractTests(unittest.TestCase):
    def test_forecast_requires_backtest_skill(self) -> None:
        # A flat frontier: the trend adds nothing, so the forecast must be refused.
        rows = []
        for year in range(2015, 2024):
            for index in range(10):
                rows.append(
                    {
                        "model": f"m-{year}-{index}",
                        "organization": "Lab",
                        "publication_date": pd.Timestamp(year=year, month=1, day=1),
                        "publication_year": year,
                        "training_compute_flop": 1e23,
                        "domain": "Language",
                        "weights_class": "open_weights",
                        "vendor": "Lab",
                        "frontier_flag": False,
                    }
                )
        frame = pd.DataFrame(rows)
        context = RunContext(reference_date_override="2024-01-01")
        tables = compute_scaling.build(frame, context)
        # Either no forecast table, or a refusal was recorded.
        if "compute_frontier_forecast" not in tables:
            claims = [refusal.claim for refusal in context.refusals]
            self.assertTrue(any("Forecast" in claim or "forecast" in claim for claim in claims))


class CompositeScoreContractTests(unittest.TestCase):
    def test_weak_agreement_makes_composite_indefensible(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "benchmark_a": "a",
                    "benchmark_b": "b",
                    "kendall_tau_b": 0.4,
                    "tau_ci_low": 0.2,
                    "tau_ci_high": 0.6,
                    "both_vendor_composites": False,
                },
                {
                    "benchmark_a": "a",
                    "benchmark_b": "c",
                    "kendall_tau_b": 0.5,
                    "tau_ci_low": 0.3,
                    "tau_ci_high": 0.7,
                    "both_vendor_composites": False,
                },
            ]
        )
        verdict = _verdict(pairs)
        self.assertFalse(bool(verdict.iloc[0]["composite_score_defensible"]))


class LagCensoringContractTests(unittest.TestCase):
    def test_lag_summary_uses_kaplan_meier(self) -> None:
        lag = pd.DataFrame(
            {
                "lag_days": [30, 40, 50, 200, 200],
                "is_right_censored": [False, False, False, True, True],
            }
        )
        summary = open_weights_lag.lag_summary(lag)
        self.assertEqual(summary.iloc[0]["estimator"], "kaplan_meier_median_with_right_censoring")
        self.assertGreater(
            float(summary.iloc[0]["kaplan_meier_median_lag_days"]),
            float(summary.iloc[0]["naive_median_of_completed_lags_days"]),
        )


class NoLegacyArtifactsTests(unittest.TestCase):
    def test_legacy_package_removed(self) -> None:
        self.assertFalse((ROOT / "src" / "frontier_ai").exists())

    def test_no_composite_index_in_analysis_outputs_when_present(self) -> None:
        analysis = ROOT / "data" / "analysis"
        if not analysis.exists():
            self.skipTest("analysis outputs not yet generated")
        forbidden = {
            "company_frontier_scores.csv",
            "company_next_frontier_probabilities.csv",
            "historical_analogy_index.csv",
            "job_replacement_feasibility.csv",
            "frontier_score_bootstrap.csv",
        }
        present = {path.name for path in analysis.glob("*.csv")}
        self.assertEqual(present & forbidden, set())


class ReportDensityTests(unittest.TestCase):
    """The published report should stay chart-dense and navigable like the prior dashboard."""

    def test_figure_catalog_is_dense(self) -> None:
        figures = ROOT / "figures"
        if not figures.exists() or not any(figures.glob("*.png")):
            self.skipTest("figures not yet generated")
        pngs = list(figures.glob("*.png"))
        self.assertGreaterEqual(len(pngs), 20, "report should ship a dense chart set")

    def test_html_dashboard_embeds_charts_and_assets(self) -> None:
        html_path = ROOT / "report" / "frontier_signals.html"
        if not html_path.exists():
            self.skipTest("report not yet generated")
        html = html_path.read_text(encoding="utf-8")
        self.assertIn('class="shell"', html)
        self.assertIn('class="side-nav"', html)
        self.assertIn('class="metric-card', html)
        self.assertGreaterEqual(html.count('class="chart"'), 20)
        self.assertIn("assets/report.css", html)
        self.assertIn("assets/report.js", html)
        self.assertTrue((ROOT / "report" / "assets" / "report.css").exists())
        self.assertTrue((ROOT / "report" / "assets" / "report.js").exists())


if __name__ == "__main__":
    unittest.main()
