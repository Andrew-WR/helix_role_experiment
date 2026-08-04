import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07b_inkling_label_subgoal_events.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("inkling_labeler_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


def trace():
    return {
        "trace_id": "trace-1", "task_id": "task-1", "domain": "math",
        "split": "train", "prompt": "Solve it.", "reference_answer": "2",
        "sentences": [
            {"sentence_id": "S0", "text": "Plan.", "is_reasoning": True},
            {"sentence_id": "S1", "text": "Thus x=2.", "is_reasoning": True},
            {"sentence_id": "S2", "text": "FINAL: 2", "is_reasoning": False},
        ],
    }


class InklingLabelingTests(unittest.TestCase):
    def test_schema_constrains_compact_label_length(self):
        schema = LABELER.compact_schema(3)
        labels = schema["properties"]["labels"]
        self.assertEqual(labels["minLength"], 3)
        self.assertEqual(labels["maxLength"], 3)

    def test_payload_and_legacy_conversion(self):
        rows = LABELER.compact_to_annotations(
            trace(), {"labels": "NFA", "review": [0]}
        )
        self.assertEqual(
            [row["primary_label"] for row in rows],
            ["neutral_support", "forward_progress", "final_answer"],
        )
        self.assertTrue(rows[0]["needs_review"])

    def test_nonreasoning_sentence_must_be_final(self):
        with self.assertRaisesRegex(ValueError, "must be final_answer"):
            LABELER.validate_compact_payload(
                trace(), {"labels": "NFN", "review": []}
            )

    def test_defaults_match_modal_endpoint(self):
        values = LABELER.settings({})
        self.assertEqual(values["model"], "thinkingmachines/Inkling-NVFP4")
        self.assertEqual(values["reasoning_effort"], "none")
        self.assertEqual(values["audit_passes"], 1)

    def test_exact_modal_key_environment_name_is_supported(self):
        with patch.dict(os.environ, {"Modal-Key": "test-value"}, clear=True):
            self.assertEqual(LABELER.secret("Modal-Key"), "test-value")


if __name__ == "__main__":
    unittest.main()
