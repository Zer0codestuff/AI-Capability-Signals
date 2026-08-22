import unittest

import numpy as np
import pandas as pd

from frontier_ai.deep_analysis import (
    COMPONENT_WEIGHTS,
    LEADERSHIP_SCENARIO_MULTIPLIERS,
    TASK_CONTACT_ASSUMPTIONS,
    leadership_audit_note,
    minmax,
    scenario_weight_vector,
    weighted_component,
)
from frontier_ai.pipeline import model_key, safe_float


class DataFixRegressionTests(unittest.TestCase):
    def test_safe_float_does_not_rescue_first_number(self):
        self.assertIsNone(safe_float("1.5B"))
        self.assertIsNone(safe_float("~70%"))
        self.assertIsNone(safe_float("3.5T"))
        self.assertEqual(safe_float("12.5"), 12.5)
        self.assertIsNone(safe_float("not-a-number"))

    def test_model_key_collapses_provider_prefixed_names(self):
        self.assertEqual(model_key("GPT-5"), model_key("OpenAI: GPT-5"))
        self.assertEqual(model_key("meta-llama/Llama-4"), model_key("Llama-4"))
        self.assertNotEqual(model_key("gpt-5"), model_key("gpt-5.5"))

    def test_minmax_keeps_neutral_default_and_nan_option(self):
        scaled = minmax(pd.Series([1.0, 2.0, float("nan")]))
        self.assertEqual(float(scaled.iloc[2]), 50.0)
        unfilled = minmax(pd.Series([1.0, 2.0, float("nan")]), fill_missing=False)
        self.assertTrue(np.isnan(unfilled.iloc[2]))

    def test_weighted_component_renormalizes_over_available_inputs(self):
        scores = pd.DataFrame(
            {
                "a": [100.0, np.nan, 60.0],
                "b": [np.nan, 10.0, 2.0],
            }
        )
        out = weighted_component(scores, [("a", 0.75, {}), ("b", 0.25, {})])
        # Row 0 only has input a (max) -> pure a score, not blended with a fake
        # neutral 50 for the missing b input.
        self.assertAlmostEqual(float(out.iloc[0]), 100.0, places=6)
        self.assertGreater(float(out.iloc[1]), 50.0)
        # Row 2 has both inputs (a at min, b at min) -> low score.
        self.assertLess(float(out.iloc[2]), 50.0)

    def test_weighted_component_all_missing_is_neutral(self):
        scores = pd.DataFrame({"a": [np.nan]})
        out = weighted_component(scores, [("a", 1.0, {}), ("missing_col", 1.0, {})])
        self.assertEqual(float(out.iloc[0]), 50.0)

    def test_task_contact_assumptions_are_explicit_and_bounded(self):
        for scenario, horizons in TASK_CONTACT_ASSUMPTIONS.items():
            self.assertEqual(set(horizons), {2, 5, 10})
            for horizon, share in horizons.items():
                self.assertGreater(share, 0.0)
                self.assertLess(share, 0.9)
                if scenario == "conservative":
                    base = TASK_CONTACT_ASSUMPTIONS["base"][horizon]
                    self.assertLessEqual(share, base)

    def test_scenario_weights_are_derived_from_baseline(self):
        components = list(COMPONENT_WEIGHTS)
        for name, spec in LEADERSHIP_SCENARIO_MULTIPLIERS.items():
            for horizon in [2, 5, 10]:
                weights = scenario_weight_vector(spec, horizon)
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
                # Every derived weight must stay within a sane band of the
                # documented baseline it was multiplied from.
                baseline = np.array([COMPONENT_WEIGHTS[c] for c in components])
                self.assertTrue((weights > 0).all())
                ratio = weights / baseline
                self.assertTrue((ratio > 0.05).all(), (name, horizon))
                self.assertTrue((ratio < 8.0).all(), (name, horizon))

    def test_leadership_audit_note_is_derived_not_hardcoded(self):
        row = pd.Series(
            {
                "model_family": "SomeFamily",
                "performance_rank": 1,
                "release_velocity_rank": 3,
                "openness_rank": 5,
                "cost_efficiency_rank": 4,
                "family_total": 11,
            }
        )
        note = leadership_audit_note(row)
        self.assertIn("ranked 1 of 11", note)
        self.assertIn("derived from this snapshot", note)
        empty = leadership_audit_note(pd.Series({"model_family": "X"}))
        self.assertIn("caution", empty)


if __name__ == "__main__":
    unittest.main()
