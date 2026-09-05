"""Taxonomy: authoritative source fields beat name heuristics."""

from __future__ import annotations

import unittest

from aicap.taxonomy import (
    WEIGHTS_CLOSED,
    WEIGHTS_OPEN,
    WEIGHTS_RESTRICTED,
    WEIGHTS_UNKNOWN,
    WEIGHTS_UNRELEASED,
    classify,
    epoch_weights_class,
    lmarena_weights_class,
    openrouter_weights_class,
)


class EpochWeightsTests(unittest.TestCase):
    def test_maps_authoritative_accessibility(self) -> None:
        self.assertEqual(epoch_weights_class("Open weights (unrestricted)", "Yes"), WEIGHTS_OPEN)
        self.assertEqual(epoch_weights_class("API access", "No"), WEIGHTS_CLOSED)
        self.assertEqual(epoch_weights_class("Unreleased", "No"), WEIGHTS_UNRELEASED)
        self.assertEqual(
            epoch_weights_class("Open weights (restricted use)", "Yes"), WEIGHTS_RESTRICTED
        )

    def test_inconsistency_becomes_unknown(self) -> None:
        # Accessibility says open, flag says no: do not pick a side silently.
        self.assertEqual(epoch_weights_class("Open weights (unrestricted)", "No"), WEIGHTS_UNKNOWN)


class OpenRouterWeightsTests(unittest.TestCase):
    def test_nan_string_is_not_a_huggingface_id(self) -> None:
        # pandas NaN stringified is "nan"; treating that as a real id would misclassify closed models.
        self.assertEqual(openrouter_weights_class(float("nan")), WEIGHTS_CLOSED)
        self.assertEqual(openrouter_weights_class("nan"), WEIGHTS_CLOSED)
        self.assertEqual(openrouter_weights_class(None), WEIGHTS_CLOSED)

    def test_real_id_is_open(self) -> None:
        self.assertEqual(openrouter_weights_class("meta-llama/Llama-3.1-70B"), WEIGHTS_OPEN)


class ArenaLicenceTests(unittest.TestCase):
    def test_proprietary_is_closed(self) -> None:
        self.assertEqual(lmarena_weights_class("Proprietary"), WEIGHTS_CLOSED)

    def test_named_open_licence(self) -> None:
        self.assertEqual(lmarena_weights_class("Apache 2.0"), WEIGHTS_OPEN)
        self.assertEqual(lmarena_weights_class("Llama 3.1 Community"), WEIGHTS_OPEN)

    def test_unknown_stays_unknown(self) -> None:
        self.assertEqual(lmarena_weights_class(""), WEIGHTS_UNKNOWN)
        self.assertEqual(lmarena_weights_class("CustomMysteryLicence"), WEIGHTS_UNKNOWN)


class ClassificationTests(unittest.TestCase):
    def test_vendor_and_family(self) -> None:
        self.assertEqual(classify("GPT-5.5", "OpenAI").vendor, "OpenAI")
        self.assertEqual(classify("GPT-5.5", "OpenAI").family, "GPT")
        self.assertEqual(classify("Claude Opus 5", "Anthropic").family, "Claude")
        self.assertEqual(classify("Llama 3.1 70B", "Meta").vendor, "Meta")
        self.assertEqual(classify("Gemma 3 27B", "Google").family, "Gemma")
        self.assertEqual(classify("Gemma 3 27B", "Google").vendor, "Google")


if __name__ == "__main__":
    unittest.main()
