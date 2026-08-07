import importlib.util
import json
import sys
import tempfile
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
    def test_stale_controls_can_be_ignored_for_gated_only_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gated = root / "gated.json"
            old_control = root / "always.json"
            gated.write_text(
                json.dumps({"steering_run_fingerprint": "new"}), encoding="utf-8"
            )
            old_control.write_text(
                json.dumps({"steering_run_fingerprint": "old"}), encoding="utf-8"
            )
            compatible, stale = FINALIZER.partition_steering_sources(
                [gated, old_control], "new"
            )
            self.assertEqual(compatible, [gated])
            self.assertEqual(stale, [old_control])

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
