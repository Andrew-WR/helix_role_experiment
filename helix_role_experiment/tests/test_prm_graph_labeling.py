import unittest

import numpy as np

from helix_role_experiment.prm_graph_labeling import (
    direct_prm_features,
    temporal_graph_features,
    tolerant_event_metrics,
    validation_gate,
)


class PrmGraphLabelingTests(unittest.TestCase):
    def test_direct_prm_features_preserve_level_and_change(self):
        features = direct_prm_features(np.asarray([0.2, 0.2, 0.8]))
        self.assertEqual(features.shape, (3, 7))
        self.assertAlmostEqual(features[2, 0], 0.8)
        self.assertAlmostEqual(features[2, 1], 0.6)
        self.assertAlmostEqual(features[0, 1], 0.0)

    def test_temporal_graph_counts_later_retrieval(self):
        vectors = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.01],
        ])
        features = temporal_graph_features(vectors, k=1)
        self.assertEqual(features.shape, (3, 10))
        # The final node retrieves node zero as its closest prior state.
        self.assertGreater(features[0, 7], features[1, 7])

    def test_tolerant_metrics_are_one_to_one(self):
        metrics = tolerant_event_metrics([(
            np.asarray([0, 1, 0, 0, 1]),
            np.asarray([1, 0, 0, 1, 0]),
        )], tolerance=1)
        self.assertEqual(metrics["tp"], 2.0)
        self.assertEqual(metrics["fp"], 0.0)
        self.assertEqual(metrics["fn"], 0.0)

    def test_validation_gate_uses_exact_and_tolerant_metrics(self):
        settings = {
            "minimum_validation_precision": 0.25,
            "minimum_validation_recall": 0.25,
            "minimum_validation_lift": 2.0,
            "minimum_tolerant_f1": 0.25,
        }
        passed = validation_gate({
            "prevalence": 0.1,
            "exact": {"precision": 0.4, "recall": 0.3},
            "tolerant_1": {"f1": 0.5},
        }, settings)
        failed = validation_gate({
            "prevalence": 0.1,
            "exact": {"precision": 0.4, "recall": 0.1},
            "tolerant_1": {"f1": 0.5},
        }, settings)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
