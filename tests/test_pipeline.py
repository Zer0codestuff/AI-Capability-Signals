import copy
import gzip
import hashlib
import json
import math
import unittest
from pathlib import Path

import yaml

from pipeline.analysis import backtest, build_trends, fit_log_trend, scenario
from pipeline.refresh import parse_catalogue, parse_metr, positive, validate_bundle

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def test_known_doubling_time(self):
        fit = fit_log_trend([(i * 90, 2 ** (i / 2)) for i in range(12)])
        self.assertAlmostEqual(fit["doubling_days"], 180)
        self.assertAlmostEqual(fit["robust_doubling_days"], 180)

    def test_trend_rejects_flat_dates_and_bad_values(self):
        for points in ([(0, 1)] * 6, [(x, -1) for x in range(8)]):
            with self.assertRaises(ValueError):
                fit_log_trend(points)

    def test_backtest_recovers_exponential(self):
        result = backtest([(i * 90, 2 ** (i / 2)) for i in range(12)])
        self.assertEqual(result["n"], 6)
        self.assertLess(result["mae_log2"], 1e-10)
        self.assertGreater(result["baseline_mae_log2"], 0)

    def test_same_day_never_counts_as_prior_history(self):
        points = [(i, 2 ** i) for i in range(5)] + [(5, 32), (5, 64)]
        self.assertEqual(backtest(points)["n"], 0)

    def test_scenario_is_conditional_and_exact(self):
        self.assertEqual(scenario(60, 360, 180), 240)
        self.assertEqual(scenario(60, 360, 180, 0), 60)
        self.assertEqual(scenario(60, 360, 180, 0.5), 120)
        with self.assertRaises(ValueError):
            scenario(60, -1, 180)

    def test_numeric_missingness_is_not_zero(self):
        for value in ("", None, "nan", "-1", "inf"):
            self.assertIsNone(positive(value))
        self.assertIsNone(positive(0))
        self.assertEqual(positive(0, zero=True), 0)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads((ROOT / "public/data/story.json").read_text())

    def test_bundle_schema_and_finite_values(self):
        validate_bundle(self.bundle)
        self.assertGreaterEqual(len(self.bundle["horizons"]), 12)
        self.assertGreaterEqual(len(self.bundle["prices"]), 20)

    def test_only_one_horizon_benchmark_version(self):
        self.assertEqual(self.bundle["benchmark"]["version"], "METR-Horizon-v1.1")
        self.assertEqual(self.bundle["benchmark"]["excluded_versions"]["METR-Horizon-v1.0"], 3)
        self.assertNotIn("gpt2", {m["id"] for m in self.bundle["horizons"]})

    def test_reliability_and_intervals(self):
        for model in self.bundle["horizons"]:
            self.assertLessEqual(model["p80"]["estimate"], model["p50"]["estimate"])
            for threshold in ("p50", "p80"):
                row = model[threshold]
                self.assertLessEqual(row["ci_low"], row["estimate"])
                self.assertLessEqual(row["estimate"], row["ci_high"])

    def test_trend_selection_and_anchor(self):
        computed = build_trends(self.bundle["horizons"])
        self.assertEqual(computed, self.bundle["trends"])
        for threshold, trend in computed.items():
            selected = [m for m in self.bundle["horizons"] if m["id"] in trend["model_ids"]]
            self.assertTrue(all(m[threshold]["estimate"] <= 960 for m in selected))
            self.assertEqual(
                trend["anchor_minutes"], max(m[threshold]["estimate"] for m in selected)
            )

    def test_size_missing_value_stays_unknown(self):
        unknown = next(m for m in self.bundle["sizes"] if m["epoch_name"] == "GPT-4.1")
        self.assertIsNone(unknown["total_billions"])
        for model in self.bundle["sizes"]:
            if model["active_billions"]:
                self.assertLessEqual(model["active_billions"], model["total_billions"])

    def test_catalogue_coverage_accounts_for_all_rows(self):
        coverage = self.bundle["price_coverage"]
        self.assertEqual(
            coverage["catalogue_rows"], coverage["included"] + sum(coverage["excluded"].values())
        )
        self.assertIsNone(coverage["benchmark_version"])
        for model in self.bundle["prices"]:
            self.assertTrue(math.isfinite(model["input"]) and model["input"] > 0)
            self.assertTrue(model["url"].startswith("https://openrouter.ai/"))

    def test_provenance_is_complete(self):
        for source in self.bundle["sources"]:
            self.assertEqual(len(source["sha256"]), 64)
            self.assertTrue(source["url"].startswith("https://"))
            self.assertIn("T", source["retrieved_at"])

    def test_invalid_catalogue_fails_instead_of_empty_charts(self):
        with self.assertRaises(ValueError):
            parse_catalogue(b'{"data": []}')

    def test_metr_version_change_fails(self):
        with self.assertRaises(ValueError):
            parse_metr(yaml.safe_dump({"benchmark_name": "v2"}).encode())

    def test_future_observation_rejected(self):
        broken = copy.deepcopy(self.bundle)
        broken["horizons"][0]["release_date"] = "2099-01-01"
        with self.assertRaises(ValueError):
            validate_bundle(broken)

    def test_cached_sources_match_hashes_when_available(self):
        checked = 0
        for source in self.bundle["sources"]:
            path = ROOT / "data/snapshots" / f"{source['sha256']}.gz"
            if path.exists():
                self.assertEqual(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(),
                                 source["sha256"])
                checked += 1
        if not checked:
            self.skipTest("Raw snapshots are local-only; run data:refresh to check hashes.")


if __name__ == "__main__":
    unittest.main()
