import tempfile
import unittest
from pathlib import Path

from helix_role_experiment.steering_artifacts import steering_run_identity


class SteeringArtifactTests(unittest.TestCase):
    def test_probe_bytes_change_run_fingerprint(self):
        config = {
            "study": {"seed": 7},
            "model": {"id": "model", "device": "cuda"},
            "collection": {"max_new_tokens": 100, "temperature": 0.6},
            "intervention": {"alpha": 0.5, "conditions": ["gated"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "probe.npz"
            probe.write_bytes(b"old")
            old = steering_run_identity(config, probe, "stop")
            probe.write_bytes(b"new")
            new = steering_run_identity(config, probe, "stop")
        self.assertNotEqual(
            old["steering_run_fingerprint"], new["steering_run_fingerprint"]
        )
        self.assertNotEqual(old["probe_sha256"], new["probe_sha256"])


if __name__ == "__main__":
    unittest.main()
