import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07b_local_compact_label_subgoal_events.py"
)
SPEC = importlib.util.spec_from_file_location("local_compact_labeler_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return json.dumps(messages)


class LocalCompactLabelingTests(unittest.TestCase):
    def test_generic_schema_is_reusable_across_sentence_counts(self):
        schema = LABELER.generic_compact_schema()
        self.assertEqual(schema["properties"]["labels"]["type"], "string")
        self.assertNotIn("maxLength", schema["properties"]["labels"])

    def test_prompt_disables_qwen_thinking(self):
        tokenizer = FakeTokenizer()
        trace = {
            "trace_id": "t", "task_id": "p", "domain": "math", "split": "train",
            "prompt": "Solve.", "reference_answer": "2",
            "sentences": [
                {"sentence_id": "S0", "text": "x=2.", "is_reasoning": True},
                {"sentence_id": "S1", "text": "FINAL: 2", "is_reasoning": False},
            ],
        }
        value = LABELER.build_prompt(tokenizer, trace, None)
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertIn("COMPLETE IMMUTABLE TRAJECTORY", value)


if __name__ == "__main__":
    unittest.main()
