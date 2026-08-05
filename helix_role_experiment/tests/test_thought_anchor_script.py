import importlib.util
import sys
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
        self.assertEqual(values["query_tokens_per_sentence"], 4)
        self.assertEqual(values["query_chunk_size"], 32)
        self.assertEqual(values["teacher_force_chunk_tokens"], 256)
        self.assertEqual(values["minimum_teacher_force_chunk_tokens"], 64)

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


if __name__ == "__main__":
    unittest.main()
