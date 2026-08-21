import unittest

from frontier_ai.model_matching import PreparedModelMatcher, find_best_model_match, normalize_model_name, normalized_aliases


class ModelMatchingTests(unittest.TestCase):
    def test_prepared_matcher_preserves_match_result(self):
        candidates = [{"model_name": "claude-sonnet-4-5-20250929", "family": "Claude", "sort_score": 80}]
        expected = find_best_model_match("Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5", "Claude", candidates)
        actual = PreparedModelMatcher(candidates).match("Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5", "Claude")
        self.assertEqual((actual.confidence, actual.benchmark_model_name), (expected.confidence, expected.benchmark_model_name))

    def test_exact_match(self):
        match = find_best_model_match(
            "gpt-5.5",
            "openai/gpt-5.5",
            "GPT",
            [{"model_name": "gpt-5.5", "family": "GPT", "sort_score": 99}],
        )
        self.assertEqual(match.confidence, "exact")
        self.assertTrue(match.direct_model_match)

    def test_normalized_exact_match(self):
        match = find_best_model_match(
            "OpenAI: GPT 5.5 Preview",
            "openai/gpt-5.5-preview",
            "GPT",
            [{"model_name": "gpt-5.5", "family": "GPT", "sort_score": 99}],
        )
        self.assertEqual(match.confidence, "normalized_exact")

    def test_alias_match_inside_family(self):
        match = find_best_model_match(
            "Anthropic: Claude Sonnet 4.5 (Fast)",
            "anthropic/claude-sonnet-4.5-fast",
            "Claude",
            [{"model_name": "claude-sonnet-4-5-20250929", "family": "Claude", "sort_score": 80}],
        )
        self.assertIn(match.confidence, {"normalized_exact", "alias_match"})
        self.assertTrue(match.direct_model_match)

    def test_family_only_match(self):
        match = find_best_model_match(
            "Qwen Experimental Model",
            "qwen/qwen-experimental",
            "Qwen",
            [{"model_name": "qwen3-235b-a22b", "family": "Qwen", "sort_score": 75}],
        )
        self.assertEqual(match.confidence, "family_only")
        self.assertFalse(match.direct_model_match)

    def test_unmatched(self):
        match = find_best_model_match(
            "Unknown Lab Model",
            "unknown/model",
            "Other",
            [{"model_name": "grok-4", "family": "Grok", "sort_score": 75}],
        )
        self.assertEqual(match.confidence, "unmatched")

    def test_family_aliases_cover_major_families(self):
        cases = {
            "GPT": "openai/gpt-5.5-preview",
            "Claude": "anthropic/claude-opus-4.7-fast",
            "Gemini": "google/gemini-3.1-pro",
            "Qwen": "qwen/qwen3-235b-a22b",
            "Llama": "meta-llama/llama-4-maverick",
            "Mistral": "mistralai/mistral-large",
            "DeepSeek": "deepseek/deepseek-r1",
            "Grok": "x-ai/grok-4",
            "Phi": "microsoft/phi-4",
            "Command": "command-r-plus",
        }
        for family, model_id in cases.items():
            with self.subTest(family=family):
                aliases = normalized_aliases(model_id, model_id, family)
                self.assertTrue(aliases)
                self.assertIn(normalize_model_name(family), aliases)

    def test_tier_tokens_do_not_collapse_distinct_models(self):
        candidates = [
            {"model_name": "o3", "family": "GPT", "sort_score": 90},
            {"model_name": "o3-mini", "family": "GPT", "sort_score": 70},
        ]
        matcher = PreparedModelMatcher(candidates)
        for query, expected in [("o3-mini", "o3-mini"), ("o3", "o3")]:
            with self.subTest(query=query):
                match = matcher.match(query, query, "GPT")
                self.assertEqual(match.benchmark_model_name, expected)

    def test_mini_variant_does_not_inherit_full_model_match(self):
        candidates = [{"model_name": "GPT-4o", "family": "GPT", "sort_score": 95}]
        match = find_best_model_match("GPT-4o-mini", "openai/gpt-4o-mini", "GPT", candidates)
        # Coarse generation aliases may rank a fallback, but they must never
        # pass as direct model-level evidence.
        self.assertIn(match.confidence, {"alias_match", "family_only"})
        self.assertFalse(match.direct_model_match)

    def test_claude_generations_do_not_cross_match(self):
        candidates = [
            {"model_name": "Claude Sonnet 3.5", "family": "Claude", "sort_score": 60},
            {"model_name": "Claude Opus 4", "family": "Claude", "sort_score": 85},
        ]
        match = find_best_model_match("Claude Sonnet 4", "anthropic/claude-sonnet-4", "Claude", candidates)
        self.assertNotIn(match.confidence, {"exact", "normalized_exact", "alias_match"})

    def test_overlap_prefers_strongest_candidate_not_first_listed(self):
        candidates = [
            {"model_name": "gpt-5.5-preview-old", "model_id": "", "family": "GPT", "sort_score": 10},
            {"model_name": "gpt-5.5", "model_id": "", "family": "GPT", "sort_score": 99},
        ]
        match = find_best_model_match("OpenAI: GPT 5.5 Preview", "openai/gpt-5.5-preview", "GPT", candidates)
        self.assertEqual(match.confidence, "normalized_exact")
        self.assertEqual(match.benchmark_model_name, "gpt-5.5")

    def test_newer_provider_prefixes_are_stripped(self):
        self.assertEqual(normalize_model_name("mistralai/mistral-large"), normalize_model_name("mistral-large"))
        self.assertEqual(normalize_model_name("deepseek-ai/deepseek-r1"), normalize_model_name("deepseek-r1"))


if __name__ == "__main__":
    unittest.main()
