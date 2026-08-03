import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07b_modernbert_event_tagger.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("modernbert_event_tagger_07b", SCRIPT)
TAGGER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(TAGGER)


class ModernBertScriptTests(unittest.TestCase):
    def test_settings_use_compact_context(self):
        value = TAGGER.tagger_settings({"study": {"seed": 7}})
        self.assertEqual(value["model_id"], "answerdotai/ModernBERT-base")
        self.assertEqual(value["max_length"], 2048)
        self.assertEqual(value["recent_sentences"], 8)
        self.assertEqual(value["epochs"], 100)
        self.assertEqual(value["minimum_event_threshold"], 0.5)
        self.assertEqual(value["target_event_precision"], 0.5)

    def test_training_selection_keeps_events_and_nearby_negatives(self):
        trace = {
            "sentences": [
                {"is_reasoning": True} for _ in range(8)
            ]
        }
        annotations = [
            {"primary_label": "forward_progress" if index == 4 else "neutral_support"}
            for index in range(8)
        ]
        selected = TAGGER.selected_training_indices(
            trace, annotations, negative_ratio=2.0, seed=1
        )
        self.assertIn(4, selected)
        self.assertIn(3, selected)
        self.assertIn(5, selected)


if __name__ == "__main__":
    unittest.main()
