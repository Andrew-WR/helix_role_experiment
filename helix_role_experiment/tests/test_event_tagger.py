import unittest

import numpy as np

from helix_role_experiment.event_tagger import (
    annotations_from_event_probabilities,
    encode_event_context,
    inferred_event_label,
    prior_event_memory,
    select_event_threshold,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        values = [ord(value) for value in text]
        return [1] + values + [2] if add_special_tokens else values

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, values):
        return [1] + list(values) + [2]


class TokenizersBackendLike:
    """Transformers 5-style backend without build_inputs_with_special_tokens."""

    def encode(self, text, add_special_tokens=False):
        values = [ord(value) for value in text]
        return [101] + values + [102] if add_special_tokens else values


def trace():
    return {
        "trace_id": "abc", "task_id": "p", "domain": "math", "split": "train",
        "prompt": "Find x.", "reference_answer": "x=2",
        "sentences": [
            {"sentence_id": "S0", "text": "Let us plan.", "is_reasoning": True},
            {"sentence_id": "S1", "text": "Thus x=3.", "is_reasoning": True},
            {"sentence_id": "S2", "text": "Actually that was wrong; x=2.", "is_reasoning": True},
            {"sentence_id": "S3", "text": "FINAL: 2", "is_reasoning": False},
        ],
    }


class EventTaggerTests(unittest.TestCase):
    def test_memory_contains_only_prior_events(self):
        annotations = [
            {"primary_label": "neutral_support"},
            {"primary_label": "forward_progress"},
            {"primary_label": "productive_backtrack"},
            {"primary_label": "final_answer"},
        ]
        value = prior_event_memory(trace()["sentences"], annotations, 2)
        self.assertEqual(value, [("forward_progress", "Thus x=3.")])

    def test_context_is_bounded_and_keeps_target(self):
        value = trace()
        value["prompt"] = "p" * 1000
        value["reference_answer"] = "r" * 1000
        ids, mask = encode_event_context(
            CharacterTokenizer(), value, 2,
            [("forward_progress", "e" * 1000)],
            recent_sentences=2, max_length=256,
        )
        decoded = "".join(chr(token) for token in ids if token > 2)
        self.assertLessEqual(len(ids), 256)
        self.assertEqual(len(ids), len(mask))
        self.assertIn("TARGET SENTENCE", decoded)
        self.assertIn("wrong; x=2", decoded)

    def test_context_supports_transformers_five_tokenizer_backend(self):
        ids, mask = encode_event_context(
            TokenizersBackendLike(), trace(), 2,
            [("forward_progress", "Thus x=3.")],
            recent_sentences=2, max_length=256,
        )
        self.assertEqual(ids[0], 101)
        self.assertEqual(ids[-1], 102)
        self.assertEqual(len(ids), len(mask))

    def test_threshold_optimizes_event_f1(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.3, 0.6, 0.9])
        threshold, metrics = select_event_threshold(labels, probabilities)
        self.assertGreater(threshold, 0.3)
        self.assertLessEqual(threshold, 0.6)
        self.assertEqual(metrics["f1"], 1.0)

    def test_threshold_respects_floor_and_precision_target(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.55, 0.60, 0.65, 0.90])
        threshold, metrics = select_event_threshold(
            labels, probabilities, minimum_threshold=0.5, target_precision=1.0
        )
        self.assertGreaterEqual(threshold, 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_annotations_preserve_event_and_final_schema(self):
        rows = annotations_from_event_probabilities(
            trace(), [0.1, 0.8, 0.9, None], threshold=0.5, review_margin=0.05
        )
        self.assertEqual(rows[1]["primary_label"], "forward_progress")
        self.assertEqual(rows[2]["primary_label"], "productive_backtrack")
        self.assertEqual(rows[3]["primary_label"], "final_answer")

    def test_backtrack_subtype_is_structural(self):
        self.assertEqual(
            inferred_event_label("However, that was wrong; return to x=2."),
            "productive_backtrack",
        )
        self.assertEqual(inferred_event_label("Therefore x=2."), "forward_progress")


if __name__ == "__main__":
    unittest.main()
