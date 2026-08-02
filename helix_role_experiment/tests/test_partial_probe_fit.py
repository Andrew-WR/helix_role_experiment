import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "07c_fit_survival_probes.py"
SPEC = importlib.util.spec_from_file_location("fit_survival_probes_07c", SCRIPT)
FITTER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(FITTER)


class PartialProbeFitTests(unittest.TestCase):
    def test_coverage_report_counts_only_complete_annotation_ids(self):
        traces = [
            {"trace_id": "m1", "domain": "math", "split": "train"},
            {"trace_id": "m2", "domain": "math", "split": "val"},
            {"trace_id": "c1", "domain": "code", "split": "train"},
        ]
        report = FITTER.coverage_report(traces, {"m1", "c1"})
        self.assertEqual(report["labeled_trajectories"], 2)
        self.assertEqual(report["missing_trace_ids"], ["m2"])
        self.assertEqual(
            report["trajectory_counts_by_domain_and_split"],
            {"math": {"train": 1}, "code": {"train": 1}},
        )
        self.assertTrue(report["exploratory_partial_fit"])


if __name__ == "__main__":
    unittest.main()
