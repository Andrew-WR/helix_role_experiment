import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07b_collect_thought_anchors.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("thought_anchor_07b", SCRIPT)
COLLECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COLLECTOR)


class ThoughtAnchorScriptTests(unittest.TestCase):
    def test_settings_are_memory_bounded(self):
        values = COLLECTOR.settings({})
        self.assertEqual(values["top_k_heads"], 16)
        self.assertEqual(values["proximity_ignore"], 1)
        self.assertEqual(values["query_tokens_per_sentence"], 4)
        self.assertEqual(values["query_chunk_size"], 32)
        self.assertEqual(values["teacher_force_chunk_tokens"], 256)
        self.assertEqual(values["minimum_teacher_force_chunk_tokens"], 64)
        self.assertTrue(values["label_guided_selection"])
        self.assertEqual(values["maximum_final_anchor_fraction"], 0.2)

    def test_query_sampling_covers_every_sentence(self):
        owners = np.asarray([-1, 0, 0, 0, 1, 1, 2, 2, 2, 2])
        positions, selected_owners = COLLECTOR.sampled_query_positions(
            owners, sentence_count=3, per_sentence=2
        )
        self.assertEqual(set(selected_owners.tolist()), {0, 1, 2})
        self.assertTrue(np.all(np.diff(positions) >= 0))

    def test_cached_chunk_uses_local_query_positions(self):
        owners = np.asarray([-1, 0, 0, 0, 1, 1, 2, 2, 2, 2])
        accumulator = COLLECTOR.SentenceAttentionAccumulator(
            owners, sentence_count=3, per_sentence=2, chunk_size=32
        )
        accumulator.query_offset = 4
        positions, selected_owners = accumulator.query_window(3)
        self.assertEqual(positions.tolist(), [0, 1, 2])
        self.assertEqual(selected_owners.tolist(), [1, 1, 2])

    def test_legacy_artifact_cannot_change_proximity_without_attention(self):
        trace = {
            "sentences": [
                {"sentence_id": "s0", "is_reasoning": True},
                {"sentence_id": "s1", "is_reasoning": True},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(
                path,
                sentence_ids=np.asarray(["s0", "s1"]),
                vertical_scores=np.zeros((1, 1, 2)),
                head_kurtosis=np.zeros((1, 1)),
            )
            self.assertTrue(COLLECTOR.valid_artifact(path, trace, 4))
            self.assertFalse(COLLECTOR.valid_artifact(path, trace, 1))

    def test_raw_attention_artifact_is_reusable_across_proximities(self):
        trace = {
            "sentences": [
                {"sentence_id": f"s{i}", "is_reasoning": True}
                for i in range(6)
            ]
        }
        matrix = np.zeros((1, 1, 6, 6), dtype=np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reusable.npz"
            np.savez(
                path,
                layers=np.asarray([3]),
                sentence_ids=np.asarray([f"s{i}" for i in range(6)]),
                vertical_scores=np.zeros((1, 1, 6)),
                head_kurtosis=np.zeros((1, 1)),
                sentence_attention=matrix,
                proximity_ignore=np.asarray([1]),
            )
            self.assertTrue(COLLECTOR.valid_artifact(path, trace, 1))
            self.assertTrue(COLLECTOR.valid_artifact(path, trace, 4))

    def test_label_guided_budget_is_fitted_on_train_and_capped(self):
        def record(trace_id, split):
            return {
                "trace_id": trace_id,
                "split": split,
                "sentences": [
                    {
                        "sentence_id": f"s{i}",
                        "within_trace_percentile": i / 19,
                        "anchor_of_anchor_percentile": (19 - i) / 19,
                        "anchor_of_anchor_candidate": i < 5,
                    }
                    for i in range(20)
                ],
            }

        records = [record("train", "train"), record("val", "val")]
        annotations = [{
            "trace_id": "train",
            "source": "human",
            "annotations": [
                {
                    "sentence_id": f"s{i}",
                    "primary_label": (
                        "forward_progress" if i < 4 else "neutral_support"
                    ),
                }
                for i in range(20)
            ],
        }]
        selector = COLLECTOR.apply_label_guided_anchor_budget(
            records, annotations, COLLECTOR.settings({})
        )
        self.assertEqual(selector["calibration_trajectory_count"], 1)
        for local in records:
            self.assertLessEqual(
                sum(row["thought_anchor"] for row in local["sentences"]), 4
            )


if __name__ == "__main__":
    unittest.main()
