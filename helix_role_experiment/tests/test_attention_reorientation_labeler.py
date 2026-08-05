import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "07b_attention_reorientation_labeler.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("attention_reorientation_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


class AttentionReorientationLabelerTests(unittest.TestCase):
    def test_features_are_trace_normalized(self):
        raw = {
            name: np.asarray([np.nan, 30.0, 10.0, 20.0])
            for name in LABELER.FEATURE_NAMES
        }
        values = LABELER.percentile_features(raw)
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertEqual(values[1, 0], 1.0)
        self.assertEqual(values[2, 0], 0.0)
        self.assertEqual(values[3, 0], 0.5)

    def test_default_target_is_high_precision(self):
        values = LABELER.settings({})
        self.assertEqual(values["target_precision"], 0.25)
        self.assertTrue(values["require_validation_gate"])

    def test_logistic_model_recovers_simple_signal(self):
        features = np.zeros((20, 5), dtype=float)
        features[10:, 0] = 1.0
        labels = np.asarray([0] * 10 + [1] * 10)
        model = LABELER.fit_logistic(features, labels, ridge=0.1)
        probabilities = LABELER.predict_logistic(model, features)
        self.assertLess(probabilities[:10].mean(), 0.5)
        self.assertGreater(probabilities[10:].mean(), 0.5)

    def test_validation_gate_requires_lift_and_recall(self):
        values = LABELER.settings({})
        passed = LABELER.validation_gate({
            "prevalence": 0.06, "precision": 0.20, "recall": 0.30,
        }, values)
        failed = LABELER.validation_gate({
            "prevalence": 0.06, "precision": 0.10, "recall": 0.30,
        }, values)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
