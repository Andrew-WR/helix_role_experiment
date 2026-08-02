import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
