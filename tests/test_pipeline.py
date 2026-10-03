"""Known-answer checks for the statistics, the price matching and the published bundle."""

import json
import math
import random
import unittest
from datetime import date, timedelta

from pipeline import audit, prices, stats
from pipeline.build import OUTPUT, lag, metr_name


def series(start: date, months: int, value) -> list[tuple[date, float]]:
    return [(start + timedelta(days=30.4375 * i), value(i / 12)) for i in range(months)]


class Statistics(unittest.TestCase):
    def test_recovers_a_known_doubling_time(self):
        rng = random.Random(1)
        points = series(date(2020, 1, 1), 60, lambda t: 5 * 2 ** (t * 2) * 10 ** rng.gauss(0, 0.02))
        trend = stats.Trend(points, log=True, reps=300)
        self.assertAlmostEqual(stats.rate(trend.line.fit.slope, True), 4.0, delta=0.1)
        self.assertAlmostEqual(stats.doubling_months(trend.line.fit.slope), 6.0, delta=0.2)
        self.assertEqual(trend.shape.verdict, "steady")
        self.assertEqual(trend.basis, "full")

    def test_detects_a_real_change_of_pace_and_projects_from_the_recent_part(self):
        rng = random.Random(2)
        points = series(
            date(2018, 1, 1), 96, lambda t: (t if t < 4 else 4 + 10 * (t - 4)) + rng.gauss(0, 0.3)
        )
        trend = stats.Trend(points, log=False, reps=300)
        self.assertEqual(trend.shape.verdict, "speeding_up")
        self.assertEqual(trend.basis, "recent")
        self.assertAlmostEqual(trend.line.fit.slope, 10.0, delta=0.5)

    def test_the_recent_part_is_never_shorter_than_two_years(self):
        rng = random.Random(3)
        points = series(
            date(2020, 1, 1), 60, lambda t: (t if t < 4 else 4 + 30 * (t - 4)) + rng.gauss(0, 0.3)
        )
        trend = stats.Trend(points, log=False, reps=300)
        self.assertGreaterEqual(trend.xs[-1] - trend.shape.split, stats.MIN_RECENT_YEARS)
        # The burst alone would be 30 a year; two years of data dilute it.
        self.assertLess(trend.line.fit.slope, 20)

    def test_prediction_band_contains_the_central_estimate(self):
        points = series(date(2021, 1, 1), 36, lambda t: 100 * 3**t)
        mid, low, high = stats.Trend(points, log=True, reps=200).predict(date(2025, 1, 1))
        self.assertTrue(low <= mid <= high)
        self.assertAlmostEqual(math.log10(mid), math.log10(100 * 3**4), delta=0.01)

    def test_crossing_date(self):
        points = series(date(2021, 1, 1), 36, lambda t: 2**t)
        crossing = stats.Trend(points, log=True, reps=200).crossing(2**6)
        self.assertLess(abs((crossing[0] - date(2027, 1, 1)).days), 10)

    def test_records_and_top_at_release(self):
        raw = [(date(2020, 1, d), v, str(v)) for d, v in [(1, 5), (2, 3), (3, 8), (4, 7), (5, 9)]]
        self.assertEqual([p[1] for p in stats.running_records(raw)], [5, 8, 9])
        self.assertEqual([p[1] for p in stats.running_records(raw, lowest=True)], [5, 3])
        self.assertEqual([p[1] for p in stats.top_at_release(raw, k=2)], [5, 3, 8, 7, 9])

    def test_backtest_uses_only_the_past_and_beats_a_flat_guess_on_a_steady_trend(self):
        rng = random.Random(4)
        points = series(date(2016, 1, 1), 100, lambda t: 10 ** (t / 2 + rng.gauss(0, 0.05)))
        result = stats.backtest(points, log=True)
        self.assertLess(result["typical_error"], result["naive_error"])
        self.assertLess(result["typical_error"], 0.1)
        self.assertGreater(result["coverage"], 0.6)


class Matching(unittest.TestCase):
    def test_names(self):
        self.assertEqual(prices.normalise("Llama-3.1-Instruct-405B"), prices.normalise("Llama 3.1-405B"))
        self.assertEqual(prices.keys("GPT-4o (May 2024)"), ["gpt 4o may 2024", "gpt 4o"])
        self.assertEqual(metr_name("claude_opus_4_6_inspect"), "Claude Opus 4.6")
        self.assertEqual(metr_name("gpt_5_3_codex"), "GPT-5.3 Codex")

    def test_price_precedence_and_open_model_policy(self):
        models = [
            {"name": "Alpha 1", "org": "OpenAI", "access": "closed"},
            {"name": "Beta 2", "org": "OpenAI", "access": "closed"},
            {"name": "Gamma 3", "org": "Meta AI", "access": "open"},
        ]
        observed = b"Model Name,Release Date,USD per 1M Tokens\nAlpha-1,2024-01-01,9\nAlpha-1,2023-01-01,12\n"
        attached = prices.attach(
            models,
            epoch_files=[(observed, True)],
            history={"prices": []},
            models_dev={
                "openai": {"models": {"alpha-1": {"name": "Alpha 1", "cost": {"input": 1, "output": 1}}}}
            },
            openrouter={
                "data": [
                    {
                        "id": "openai/beta-2",
                        "name": "OpenAI: Beta 2",
                        "pricing": {"prompt": "0.000002", "completion": "0.000006"},
                    },
                    {
                        "id": "meta-llama/gamma-3",
                        "name": "Meta: Gamma 3",
                        "pricing": {"prompt": "0.000001", "completion": "0.000001"},
                    },
                ]
            },
        )
        self.assertEqual(attached["Alpha 1"], prices.Price(12.0, "epoch_prices", "observed", "2023-01-01"))
        self.assertEqual(attached["Beta 2"], prices.Price(3.0, "openrouter", "list_current"))
        self.assertNotIn("Gamma 3", attached)

    def test_lag_counts_months_since_the_leader_reached_the_level(self):
        leader = [(date(2024, 1, 1), 100.0, "L1"), (date(2025, 1, 1), 120.0, "L2")]
        follower = [(date(2023, 6, 1), 90.0, "early"), (date(2024, 7, 1), 100.0, "F1")]
        result = lag(leader, follower, date(2025, 1, 1))
        self.assertEqual([(p["n"], round(p["v"])) for p in result], [("F1", 6), ("Today", 12)])


class Guards(unittest.TestCase):
    def test_only_vetted_rows_are_usable(self):
        row = {"notable": True, "confidence": "Likely", "slip": False}
        self.assertIsNone(audit.status(row, "params"))
        self.assertEqual(audit.status({**row, "confidence": "Speculative"}, "params"), "speculative")
        self.assertEqual(audit.status({**row, "confidence": ""}, "compute"), "unrated")
        self.assertEqual(audit.status({**row, "notable": False}, "cost"), "unvetted")
        self.assertEqual(audit.status({**row, "slip": True}, "params"), "unit")
        self.assertIsNone(audit.status({**row, "slip": True}, "compute"))

    def test_unit_slip_reads_the_note(self):
        self.assertEqual(audit.unit_slip(2.1e9, 'Elon said it "will be the 2.1T model"'), 2.1e12)
        self.assertIsNone(audit.unit_slip(2.8e12, "2.8T total, 104B active parameters"))
        self.assertIsNone(audit.unit_slip(2e12, "a 288 billion active parameter model"))
        self.assertIsNone(audit.unit_slip(1.75e11, "no size quoted"))

    def test_a_recent_tenfold_leap_is_held_back_but_history_is_kept(self):
        checks = audit.Audit()
        points = [
            (date(2012, 1, 1), 1.0, "old"),
            (date(2013, 1, 1), 50.0, "old leap"),
            (date(2025, 1, 1), 100.0, "steady"),
            (date(2025, 6, 1), 5000.0, "suspect"),
            (date(2025, 9, 1), 300.0, "next"),
        ]
        kept, held = audit.records(points, chart="test", today=date(2026, 1, 1), audit=checks)
        self.assertEqual([name for _, _, name in kept], ["old", "old leap", "steady", "next"])
        self.assertEqual(held, {"suspect"})
        self.assertEqual(checks.flags[0]["name"], "suspect")


class PublishedBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.story = json.loads(OUTPUT.read_text())

    def test_every_chart_is_drawable(self):
        for chapter in self.story["chapters"].values():
            for chart in chapter["charts"].values():
                self.assertIn(chart["scale"], ("log", "linear"))
                for entry in [*chart["points"], *(p for s in chart["series"] for p in s["points"])]:
                    self.assertTrue(math.isfinite(entry["v"]))
                    self.assertLessEqual(entry["d"], self.story["generated_on"])
                    if chart["scale"] == "log":
                        self.assertGreater(entry["v"], 0)

    def test_projections_exist_only_where_the_method_earned_them(self):
        for chapter in self.story["chapters"].values():
            for chart in chapter["charts"].values():
                for item in chart["series"]:
                    trend = item.get("trend")
                    if not trend:
                        continue
                    self.assertLessEqual(trend["rate"]["lo"], trend["rate"]["hi"])
                    if trend["band"]:
                        check = trend["backtest"]
                        self.assertLess(check["typical_error"], check["naive_error"])
                        for step in trend["band"]:
                            self.assertTrue(step["lo"] <= step["v"] <= step["hi"])
                    else:
                        self.assertFalse(trend["projectable"])

    def test_no_unvetted_value_reaches_a_record_or_a_trend(self):
        for name in ("size", "compute", "cost"):
            chart = next(iter(self.story["chapters"][name]["charts"].values()))
            set_aside = {(p["n"], p["d"]) for p in chart["points"] if p.get("q")}
            for entry in chart["series"][0]["points"]:
                self.assertNotIn((entry["n"], entry["d"]), set_aside)
            for entry in chart["points"]:
                if entry.get("g") == "frontier":
                    self.assertIsNone(entry.get("q"))
            self.assertIn(self.story["chapters"][name]["facts"]["largest"]["c"], audit.RATED)

    def test_sources_are_traceable(self):
        self.assertGreaterEqual(len(self.story["sources"]), 6)
        for source in self.story["sources"]:
            for file in source["files"]:
                self.assertEqual(len(file["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
