import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helix_role_experiment.prm_graph_labeling import (
    direct_prm_features,
    temporal_graph_features,
    tolerant_event_metrics,
    validation_gate,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "07b_evaluate_prm_graph_labelers.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prm_graph_labelers_07b", SCRIPT)
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(SCRIPT_MODULE)


class PrmGraphLabelingTests(unittest.TestCase):
    def test_saved_embedding_graph_probe_can_be_reloaded_for_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "tables"
            models = root / "models"
            tables.mkdir()
            models.mkdir()
            (tables / "prm_graph_labeler_benchmark.json").write_text(
                json.dumps({
                    "methods": {
                        "embedding_graph": {
                            "threshold": 0.7,
                            "validation_gate": {"passed": True},
                        }
                    }
                }),
                encoding="utf-8",
            )
            np.savez(
                models / "prm_graph_labeler_probes.npz",
                embedding_graph_mean=np.zeros(2),
                embedding_graph_scale=np.ones(2),
                embedding_graph_coefficients=np.ones(2),
                embedding_graph_intercept=np.asarray(-0.5),
                embedding_graph_threshold=np.asarray(0.7),
            )
            model, threshold, _ = SCRIPT_MODULE.load_embedding_graph_probe({
                "tables": tables, "models": models,
            })
            self.assertEqual(threshold, 0.7)
            self.assertEqual(len(model["coefficients"]), 2)

    def test_reasonflux_config_gets_tokenizer_padding_id(self):
        config = SimpleNamespace()
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
        result = SCRIPT_MODULE.ensure_prm_config_compatibility(config, tokenizer)
        self.assertEqual(result.pad_token_id, 151643)

    def test_reasonflux_revision_is_pinned(self):
        values = SCRIPT_MODULE.settings({})
        self.assertEqual(len(values["prm_revision"]), 40)

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
