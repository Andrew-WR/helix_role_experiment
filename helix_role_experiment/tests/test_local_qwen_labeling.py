import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07b_local_qwen_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location("local_qwen_labeler_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"{kwargs['enable_thinking']}|{messages[-1]['content']}"


class LocalQwenLabelingTests(unittest.TestCase):
    def test_shards_whole_traces_without_splitting_them(self):
        rows = [
            {"trace_id": "a", "chunk_index": 0},
            {"trace_id": "a", "chunk_index": 1},
            {"trace_id": "b", "chunk_index": 0},
            {"trace_id": "c", "chunk_index": 0},
        ]
        zero = LABELER.selected_requests(rows, 0, 2)
        one = LABELER.selected_requests(rows, 1, 2)
        self.assertEqual({row["trace_id"] for row in zero}, {"a", "c"})
        self.assertEqual({row["trace_id"] for row in one}, {"b"})
        self.assertEqual(len(zero), 3)

    def test_prompt_disables_thinking_and_adds_retry_error(self):
        request = {"body": {"input": [{"role": "user", "content": "label"}]}}
        prompt = LABELER.build_prompt(FakeTokenizer(), request, "bad IDs")
        self.assertIn("False|", prompt)
        self.assertIn("bad IDs", prompt)

    def test_worker_environment_disables_nested_engine_process(self):
        LABELER.configure_worker_environment(1)
        self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertTrue(os.environ["TRITON_CACHE_DIR"].endswith("gpu1"))
        self.assertTrue(os.environ["TORCHINDUCTOR_CACHE_DIR"].endswith("gpu1"))


if __name__ == "__main__":
    unittest.main()
