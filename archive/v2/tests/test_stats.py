"""Tests that estimators recover known answers on synthetic data.

These are the tests the previous version did not have. A schema check that a column exists and
falls in a plausible range will pass for an arbitrarily wrong number. The tests below fail unless
the estimator is actually correct.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aicap.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    inverse_variance_combine,
    kaplan_meier_median,
    kendall_tau,
    ols_hc3,
    paired_bootstrap_ci,
    permutation_p_value,
    rolling_origin_backtest,
    theil_sen_slope,
    top_k_overlap,
    wilson_interval,
)


class WilsonIntervalTests(unittest.TestCase):
    def test_recovers_known_proportion(self) -> None:
        point, low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(point, 0.5)
        self.assertGreater(low, 0.35)
        self.assertLess(high, 0.65)
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_stays_inside_unit_interval_near_boundary(self) -> None:
        point, low, high = wilson_interval(1, 20)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertAlmostEqual(point, 0.05)

    def test_empty_denominator_returns_nan(self) -> None:
        point, low, high = wilson_interval(0, 0)
        self.assertTrue(math.isnan(point) and math.isnan(low) and math.isnan(high))


class BootstrapTests(unittest.TestCase):
    def test_mean_interval_covers_truth_on_gaussian_sample(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(10.0, 2.0, size=80)
        point, low, high = bootstrap_ci(sample, statistic=np.mean, draws=2000, seed=1, min_n=8)
        self.assertAlmostEqual(point, float(np.mean(sample)), places=6)
        self.assertLessEqual(low, 10.0)
        self.assertGreaterEqual(high, 10.0)

    def test_refuses_small_samples(self) -> None:
        point, low, high = bootstrap_ci([1.0, 2.0, 3.0], min_n=8)
        self.assertTrue(math.isnan(point) and math.isnan(low) and math.isnan(high))

    def test_paired_bootstrap_recovers_correlation(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=60)
        y = 0.8 * x + rng.normal(scale=0.2, size=60)
        pairs = list(zip(x, y, strict=True))
        point, low, high = paired_bootstrap_ci(
            pairs, statistic=lambda a, b: float(np.corrcoef(a, b)[0, 1]), draws=1500, seed=3, min_n=8
        )
        self.assertGreater(point, 0.7)
        self.assertLess(low, point)
        self.assertGreater(high, point)


class RegressionTests(unittest.TestCase):
    def test_ols_recovers_known_slope(self) -> None:
        x = np.arange(20, dtype=float)
        y = 3.0 + 1.5 * x + np.array([((-1) ** i) * 0.01 for i in range(20)])
        fit = ols_hc3(x, y)
        self.assertTrue(fit.valid)
        self.assertAlmostEqual(fit.slope, 1.5, places=2)
        self.assertAlmostEqual(fit.intercept, 3.0, places=1)
        self.assertLess(fit.slope_low, 1.5)
        self.assertGreater(fit.slope_high, 1.5)
        self.assertGreater(fit.r_squared, 0.99)

    def test_ols_rejects_constant_x(self) -> None:
        fit = ols_hc3([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(fit.valid)

    def test_theil_sen_resists_outlier(self) -> None:
        x = np.arange(15, dtype=float)
        y = 2.0 * x
        y[14] = 1000.0
        ols = ols_hc3(x, y)
        ts, low, high = theil_sen_slope(x, y)
        self.assertAlmostEqual(ts, 2.0, places=1)
        self.assertGreater(ols.slope, ts)
        self.assertLessEqual(low, 2.0)
        self.assertGreaterEqual(high, 2.0)


class RankAgreementTests(unittest.TestCase):
    def test_kendall_perfect_agreement(self) -> None:
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(kendall_tau(x, x), 1.0)

    def test_kendall_perfect_disagreement(self) -> None:
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(kendall_tau(x, -x), -1.0)

    def test_top_k_overlap(self) -> None:
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        self.assertAlmostEqual(top_k_overlap(x, y, 2), 0.5)


class MetaAnalysisTests(unittest.TestCase):
    def test_inverse_variance_prefers_precise_estimate(self) -> None:
        point, se = inverse_variance_combine([10.0, 12.0], [0.25, 4.0])
        self.assertLess(point, 11.0)
        self.assertGreater(point, 10.0)
        self.assertGreater(se, 0.0)

    def test_single_estimate_passthrough(self) -> None:
        point, se = inverse_variance_combine([7.5], [4.0])
        self.assertAlmostEqual(point, 7.5)
        self.assertAlmostEqual(se, 2.0)


class MultipleComparisonTests(unittest.TestCase):
    def test_benjamini_hochberg_controls_false_discoveries(self) -> None:
        p = [1e-6, 2e-5, 3e-4, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rejected, q = benjamini_hochberg(p, alpha=0.05)
        self.assertEqual(int(rejected.sum()), 3)
        self.assertTrue(np.all(q[rejected] <= 0.05))
        self.assertTrue(np.all(q[~rejected] > 0.05))

    def test_permutation_p_value_is_never_zero(self) -> None:
        p = permutation_p_value(1000.0, resample=lambda rng: float(rng.normal()), draws=200, seed=5)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 0.02)


class KaplanMeierTests(unittest.TestCase):
    def test_no_censoring_matches_sample_median(self) -> None:
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        observed = [True, True, True, True, True]
        median, follow_up = kaplan_meier_median(durations, observed)
        self.assertAlmostEqual(median, 30.0)
        self.assertAlmostEqual(follow_up, 50.0)

    def test_heavy_censoring_returns_nan_median(self) -> None:
        # One early death, then the remaining mass is censored: survival stays above 0.5.
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        observed = [True, False, False, False, False]
        median, follow_up = kaplan_meier_median(durations, observed)
        self.assertTrue(math.isnan(median))
        self.assertAlmostEqual(follow_up, 50.0)

    def test_naive_median_of_completed_understates_when_hard_cases_censored(self) -> None:
        durations = [30.0, 40.0, 50.0, 200.0, 200.0]
        observed = [True, True, True, False, False]
        km_median, _ = kaplan_meier_median(durations, observed)
        naive = float(np.median([30.0, 40.0, 50.0]))
        self.assertGreater(km_median, naive)


class BacktestTests(unittest.TestCase):
    def test_linear_series_beats_last_value_baseline(self) -> None:
        x = np.arange(20, dtype=float)
        y = 1.0 + 0.5 * x
        result = rolling_origin_backtest(x, y, horizon=1, min_train=6)
        self.assertGreater(result.folds, 0)
        self.assertTrue(result.beats_baseline)
        self.assertLess(result.skill_ratio, 0.2)

    def test_pure_noise_does_not_beat_baseline(self) -> None:
        rng = np.random.default_rng(7)
        x = np.arange(25, dtype=float)
        y = rng.normal(size=25)
        result = rolling_origin_backtest(x, y, horizon=1, min_train=8)
        self.assertGreaterEqual(result.skill_ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
