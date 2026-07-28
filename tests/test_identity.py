"""Matcher precision, version preservation, and the no-family-fallback invariant."""

from __future__ import annotations

import unittest

from aicap.identity import (
    GOLD_CANDIDATES,
    GOLD_PAIRS,
    STRICT_TIERS,
    TIER_CONFIGURATION_VARIANT,
    TIER_EXACT,
    TIER_UNMATCHED,
    Matcher,
    canonical_key,
    evaluate,
    split_configuration,
)


class CanonicalKeyTests(unittest.TestCase):
    def test_preserves_adjacent_versions(self) -> None:
        self.assertNotEqual(canonical_key("gpt-5.4"), canonical_key("gpt-5.5"))
        self.assertNotEqual(canonical_key("claude-opus-4.6"), canonical_key("claude-opus-4.7"))

    def test_strips_vendor_display_prefix(self) -> None:
        self.assertEqual(canonical_key("OpenAI: GPT-5.5"), canonical_key("gpt-5.5"))

    def test_size_words_are_identity_not_configuration(self) -> None:
        # mini/nano/flash/pro/max are different models, not settings of one model.
        base, suffixes = split_configuration(canonical_key("gpt-5.4-nano"))
        self.assertEqual(suffixes, ())
        self.assertIn("nano", base)


class MatcherTests(unittest.TestCase):
    def test_gold_precision_and_recall(self) -> None:
        evaluation = evaluate()
        self.assertEqual(evaluation.precision, 1.0)
        self.assertEqual(evaluation.recall, 1.0)
        self.assertEqual(evaluation.specificity, 1.0)

    def test_no_family_fallback(self) -> None:
        # A query with no candidate must return unmatched — never the family's best model.
        matcher = Matcher(GOLD_CANDIDATES)
        result = matcher.match("gpt-5.9")
        self.assertEqual(result.tier, TIER_UNMATCHED)
        self.assertIsNone(result.candidate_id)

    def test_configuration_variant_is_not_strict(self) -> None:
        matcher = Matcher(GOLD_CANDIDATES)
        result = matcher.match("gpt-5.4-high")
        self.assertEqual(result.tier, TIER_CONFIGURATION_VARIANT)
        self.assertFalse(result.is_strict)
        self.assertNotIn(result.tier, STRICT_TIERS)

    def test_sibling_size_is_rejected(self) -> None:
        matcher = Matcher(GOLD_CANDIDATES)
        result = matcher.match("gpt-5.4-nano")
        self.assertEqual(result.tier, TIER_UNMATCHED)

    def test_exact_match(self) -> None:
        matcher = Matcher(GOLD_CANDIDATES)
        result = matcher.match("OpenAI: GPT-5.5")
        self.assertEqual(result.tier, TIER_EXACT)
        self.assertEqual(result.candidate_id, "openai/gpt-5.5")

    def test_gold_pairs_cover_negatives(self) -> None:
        negatives = [expected for _, expected in GOLD_PAIRS if expected is None]
        self.assertGreaterEqual(len(negatives), 3)


if __name__ == "__main__":
    unittest.main()
