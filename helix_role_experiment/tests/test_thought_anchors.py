import unittest

import numpy as np

from helix_role_experiment.thought_anchors import (
    excess_kurtosis,
    forward_anchor_overlap,
    receiver_head_statistics,
    top_fraction_flags,
    vertical_scores,
)


class ThoughtAnchorTests(unittest.TestCase):
    def test_vertical_scores_ignore_nearby_sentences(self):
        matrix = np.zeros((7, 7), dtype=float)
        matrix[4:, 0] = 10.0
        scores = vertical_scores(matrix, proximity_ignore=4)
        self.assertTrue(np.isfinite(scores[0]))
        self.assertTrue(np.isnan(scores[-1]))
        self.assertGreater(scores[0], scores[1])

    def test_receiver_statistics_shapes(self):
        matrix = np.random.default_rng(2).random((2, 3, 10, 10))
        scores, kurtosis = receiver_head_statistics(matrix, proximity_ignore=2)
        self.assertEqual(scores.shape, (2, 3, 10))
        self.assertEqual(kurtosis.shape, (2, 3))

    def test_top_fraction_is_exact_and_excludes_nan(self):
        flags, percentiles = top_fraction_flags(
            np.asarray([0.1, 0.2, np.nan, 0.4, 0.3]), 0.25
        )
        self.assertEqual(flags.tolist(), [False, False, False, True, False])
        self.assertTrue(np.isnan(percentiles[2]))

    def test_spiky_values_have_positive_excess_kurtosis(self):
        self.assertGreater(excess_kurtosis(np.asarray([0] * 20 + [10])), 0)

    def test_forward_overlap_reports_both_directions(self):
        anchors = [{
            "trace_id": "t", "sentences": [
                {"sentence_id": "S0", "score": 0.9, "thought_anchor": True},
                {"sentence_id": "S1", "score": 0.2, "thought_anchor": False},
                {"sentence_id": "S2", "score": 0.8, "thought_anchor": True},
            ],
        }]
        annotations = [{
            "trace_id": "t", "source": "inkling_compact_trajectory",
            "annotations": [
                {"sentence_id": "S0", "primary_label": "forward_progress"},
                {"sentence_id": "S1", "primary_label": "forward_progress"},
                {"sentence_id": "S2", "primary_label": "neutral_support"},
            ],
        }]
        report = forward_anchor_overlap(anchors, annotations)["overall"]
        self.assertEqual(report["percent_llm_forward_that_are_anchors"], 50.0)
        self.assertEqual(report["percent_anchors_labeled_llm_forward"], 50.0)


if __name__ == "__main__":
    unittest.main()
