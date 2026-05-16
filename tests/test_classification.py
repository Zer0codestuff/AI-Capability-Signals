import unittest

from frontier_ai.pipeline import classify_access, classify_entity, classify_family
from frontier_ai.deep_analysis import frontier_family_from_model


class ClassificationTests(unittest.TestCase):
    def test_model_family_is_not_vendor_only(self):
        cases = [
            ("Google: Gemini 3 Pro", "google", "Gemini"),
            ("Google: Gemma 3 27B", "google", "Gemma"),
            ("BERT-Large", "Google", "Other"),
            ("AmoebaNet-A", "Google Brain", "Other"),
            ("OpenAI: GPT-5.5", "openai", "GPT"),
            ("Sora 2.0", "OpenAI", "Other"),
            ("Anthropic: Claude Opus 4.7", "anthropic", "Claude"),
            ("meta-llama/Llama-4", "meta", "Llama"),
            ("Qwen3 Max", "alibaba", "Qwen"),
            ("DeepSeek V4", "deepseek", "DeepSeek"),
            ("Microsoft: Phi 4", "microsoft", "Phi"),
            ("Command R", "independent", "Command"),
            ("xAI: Grok 4", "x-ai", "Grok"),
        ]
        for name, organization, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(classify_family(name, organization), expected)

    def test_entity_splits_vendor_family_and_product_line(self):
        entity = classify_entity("Google: Gemma 3 27B", "google")
        self.assertEqual(entity.vendor, "Google")
        self.assertEqual(entity.model_family, "Gemma")
        self.assertEqual(entity.product_line, "Gemma")

    def test_gpt_oss_is_open_weight_not_closed_by_vendor(self):
        self.assertEqual(classify_family("OpenAI: gpt-oss-120b", "openai"), "GPT")
        self.assertEqual(classify_access("OpenAI: gpt-oss-120b", "openai"), "open_weight")

    def test_deep_analysis_keeps_gemma_separate_from_gemini(self):
        self.assertEqual(frontier_family_from_model("Google: Gemma 3 27B", "google/gemma-3-27b", "google"), "Gemma")
        self.assertEqual(frontier_family_from_model("Google: Gemini 3 Pro", "google/gemini-3-pro", "google"), "Gemini")
        self.assertEqual(frontier_family_from_model("BERT-Large", "google/bert-large", "google"), "Other")


if __name__ == "__main__":
    unittest.main()
