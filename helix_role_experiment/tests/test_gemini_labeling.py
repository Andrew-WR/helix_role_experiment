import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07b_gemini_label_subgoal_events.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_labeler_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


def trace():
    return {
        "trace_id": "trace-1",
        "task_id": "task-1",
        "domain": "math",
        "split": "train",
        "prompt": "Solve it.",
        "reference_answer": "2",
        "sentences": [
            {"sentence_id": "S0", "text": "Plan the calculation.", "is_reasoning": True},
            {"sentence_id": "S1", "text": "Thus x=2.", "is_reasoning": True},
            {"sentence_id": "S2", "text": "FINAL: 2", "is_reasoning": False},
        ],
    }


class GeminiLabelingTests(unittest.TestCase):
    def test_compact_payload_requires_one_code_per_sentence(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            LABELER.validate_compact_payload(
                trace(), {"labels": "FN", "review": []}
            )

    def test_compact_labels_expand_into_legacy_schema(self):
        rows = LABELER.compact_to_annotations(
            trace(), {"labels": "NFA", "review": []}, disagreement={0}
        )
        self.assertEqual(
            [row["primary_label"] for row in rows],
            ["neutral_support", "forward_progress", "final_answer"],
        )
        self.assertTrue(rows[0]["needs_review"])
        self.assertEqual(rows[1]["evidence"], "Thus x=2.")
        self.assertTrue(rows[2]["needs_review"])

    def test_productive_backtrack_is_preserved(self):
        rows = LABELER.compact_to_annotations(
            trace(), {"labels": "BNA", "review": []}
        )
        self.assertEqual(rows[0]["primary_label"], "productive_backtrack")
        self.assertEqual(rows[0]["advances_valid_path"], "yes")

    def test_final_answer_is_rejected_inside_reasoning(self):
        with self.assertRaisesRegex(ValueError, "cannot be final_answer"):
            LABELER.validate_compact_payload(
                trace(), {"labels": "ANR", "review": []}
            )

    def test_audit_prompt_contains_prior_labels(self):
        value = LABELER.trajectory_prompt(
            trace(), prior={"labels": "NFA", "review": [0]}
        )
        self.assertIn("AUDIT PASS", value)
        self.assertIn("PRIOR LABELS: NFA", value)


if __name__ == "__main__":
    unittest.main()
