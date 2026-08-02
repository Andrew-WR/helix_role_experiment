import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "07e_finalize_readiness_results.py"
SPEC = importlib.util.spec_from_file_location("readiness_finalizer_07e", SCRIPT)
FINALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(FINALIZER)


class ReadinessFinalizerTests(unittest.TestCase):
    def test_status_distinguishes_within_model_failure_from_replication(self):
        failed = [{
            "candidate_method": True,
            "within_model_success": False,
            "cross_model_replication": False,
        }]
        self.assertIn("within-model", FINALIZER.commercial_status(failed))
        promising = [{
            "candidate_method": True,
            "within_model_success": True,
            "cross_model_replication": False,
        }]
        self.assertIn("second-model", FINALIZER.commercial_status(promising))

    def test_controls_are_not_selected_as_commercial_candidate(self):
        controls = [{
            "candidate_method": False,
            "within_model_success": True,
            "cross_model_replication": True,
        }]
        self.assertIn("not evaluated", FINALIZER.commercial_status(controls))


if __name__ == "__main__":
    unittest.main()
