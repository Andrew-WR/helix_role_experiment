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
        annotations = {
            "m1": [{"primary_label": "forward_progress", "needs_review": False}],
            "c1": [{"primary_label": "productive_backtrack", "needs_review": True}],
        }
        report = FITTER.coverage_report(traces, annotations)
        self.assertEqual(report["labeled_trajectories"], 2)
        self.assertEqual(report["missing_trace_ids"], ["m2"])
        self.assertEqual(
            report["trajectory_counts_by_domain_and_split"],
            {"math": {"train": 1}, "code": {"train": 1}},
        )
        self.assertTrue(report["exploratory_partial_fit"])
        self.assertEqual(report["sentence_label_counts"]["forward_progress"], 1)
        self.assertEqual(report["sentence_label_counts"]["productive_backtrack"], 1)
        self.assertEqual(report["reasoning_sentences_needing_review"], 1)


if __name__ == "__main__":
    unittest.main()
