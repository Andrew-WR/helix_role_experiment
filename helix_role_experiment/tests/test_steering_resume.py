import importlib.util
import sys
import tempfile
import unittest
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "07d_run_readiness_steering.py"
SPEC = importlib.util.spec_from_file_location("run_readiness_steering_07d", SCRIPT)
STEERING = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STEERING)


class SteeringResumeTests(unittest.TestCase):
    def test_destination_is_stable_and_progress_counts_existing_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_id, first = STEERING.steering_destination(root, "task-1", "gated", 7)
            second_id, second = STEERING.steering_destination(root, "task-1", "gated", 7)
            self.assertEqual(first_id, second_id)
            self.assertEqual(first, second)
            first.write_text(
                json.dumps({"steering_run_fingerprint": "new-probe"}),
                encoding="utf-8",
            )
            missing = root / "missing.json"
            self.assertEqual(
                STEERING.completed_expected([first, missing], "new-probe"), 1
            )
            self.assertEqual(
                STEERING.completed_expected([first, missing], "old-probe"), 0
            )


if __name__ == "__main__":
    unittest.main()
