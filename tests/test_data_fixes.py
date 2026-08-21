import unittest

from frontier_ai.pipeline import model_key, safe_float


class PipelineDataIntegrityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
