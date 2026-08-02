import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "evaluate_humaneval_subset.py"
SPEC = importlib.util.spec_from_file_location("humaneval_subset_evaluator", SCRIPT)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EVALUATOR)


class HumanEvalSubsetEvaluatorTests(unittest.TestCase):
    def test_program_combines_prompt_completion_tests_and_entry_point(self):
        task = {
            "prompt": "def add(a, b):\n",
            "metadata": {
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
                "entry_point": "add",
            },
        }
        program = EVALUATOR.build_program(task, "    return a + b\n")
        self.assertIn("return a + b", program)
        self.assertTrue(program.endswith("check(add)"))

    def test_rebuild_inputs_uses_saved_baseline_and_steering_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces"
            tables = root / "tables"
            baseline = traces / "readiness_baseline"
            steering = traces / "readiness_steering"
            baseline.mkdir(parents=True)
            steering.mkdir(parents=True)
            tables.mkdir()
            (baseline / "a.json").write_text(json.dumps({
                "task_id": "HumanEval/1", "split": "test", "domain": "code",
                "humaneval_completion": "    return 1\n",
            }), encoding="utf-8")
            (steering / "b.json").write_text(json.dumps({
                "task_id": "HumanEval/1", "domain": "code", "condition": "gated",
                "humaneval_completion": "    return 2\n",
            }), encoding="utf-8")
            counts = EVALUATOR.rebuild_sample_files({
                "traces": traces, "tables": tables
            })
            self.assertEqual(counts, {
                "baseline": 1, "gated": 1, "always": 0, "random": 0
            })
            self.assertTrue((tables / "humaneval_baseline.jsonl").exists())
            self.assertTrue((tables / "humaneval_gated.jsonl").exists())
            self.assertFalse((tables / "humaneval_random.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
