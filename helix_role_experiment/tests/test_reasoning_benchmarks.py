import gzip
import json
import tempfile
import unittest
from pathlib import Path

from helix_role_experiment.reasoning_benchmarks import extract_humaneval_completion, load_humaneval_tasks


class ReasoningBenchmarkTests(unittest.TestCase):
    def test_extracts_only_completion(self):
        text = "</think>\nFINAL_CODE:\n```python\n    return x + 1\n```<|im_end|>"
        self.assertEqual(extract_humaneval_completion(text), "    return x + 1\n")

    def test_local_humaneval_fixture_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "HumanEval.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                for index in range(4):
                    handle.write(json.dumps({
                        "task_id": f"HumanEval/{index}", "prompt": "def f():\n",
                        "canonical_solution": "    return 1\n", "entry_point": "f", "test": "assert f()==1",
                    }) + "\n")
            first = load_humaneval_tasks(2, 7, source)
            second = load_humaneval_tasks(2, 7, source)
            self.assertEqual([row.task_id for row in first], [row.task_id for row in second])
            self.assertTrue(all(row.domain == "code" for row in first))


if __name__ == "__main__":
    unittest.main()
