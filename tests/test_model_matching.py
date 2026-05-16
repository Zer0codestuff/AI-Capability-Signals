import unittest

from frontier_ai.model_matching import find_best_model_match, normalize_model_name, normalized_aliases


class ModelMatchingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
