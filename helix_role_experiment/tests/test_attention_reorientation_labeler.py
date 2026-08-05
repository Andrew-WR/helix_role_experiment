import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "07b_attention_reorientation_labeler.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("attention_reorientation_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


class AttentionReorientationLabelerTests(unittest.TestCase):
    def test_percentiles_are_trace_normalized(self):
        values = LABELER.percentile_scores(
            np.asarray([np.nan, 30.0, 10.0, 20.0])
        )
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[2], 0.0)
        self.assertEqual(values[3], 0.5)

    def test_default_target_is_high_precision(self):
        values = LABELER.settings({})
        self.assertEqual(values["target_precision"], 0.5)
        self.assertIn("all_head_median", values["variants"])


if __name__ == "__main__":
    unittest.main()
