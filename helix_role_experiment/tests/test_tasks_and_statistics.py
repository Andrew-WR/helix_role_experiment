import tempfile
import unittest
from pathlib import Path

import numpy as np

from helix_role_experiment.controlled_tasks import generate_suite
from helix_role_experiment.counterfactuals import (
    build_progress_position_cross,
    validate_variant,
)
from helix_role_experiment.statistics import grouped_cross_validated_ridge
from helix_role_experiment.traces import TraceRecord, TraceStore


class TaskAndStatisticsTests(unittest.TestCase):
    def test_counterfactuals_have_exact_state_controls(self):
        problems = generate_suite(2, seed=5)
        for problem in problems:
            variants = build_progress_position_cross(problem)
            conditions = {variant.condition for variant in variants}
            self.assertTrue(
                {"teleport", "rollback", "loop", "complete_answer_forbidden"}.issubset(
                    conditions
                )
            )
            for variant in variants:
                valid, reason = validate_variant(variant, problem)
                self.assertTrue(valid, reason)

    def test_grouped_cross_validation_recovers_signal(self):
        rng = np.random.default_rng(6)
        groups = np.repeat(np.arange(30), 4)
        x = rng.normal(size=(len(groups), 3))
        y = x @ np.array([[2.0], [-1.0], [0.5]]) + rng.normal(
            scale=0.05, size=(len(groups), 1)
        )
        result = grouped_cross_validated_ridge(x, y, groups, folds=5, seed=7)
        self.assertGreater(result.r2, 0.99)

    def test_trace_store_round_trip_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(directory)
            record = TraceRecord(
                request_id="abc",
                problem_id="p1",
                task_family="test",
                condition="normal",
                split="test",
                layer=0,
                prompt_token_count=3,
                token_ids=list(range(8)),
                tokens=[str(value) for value in range(8)],
                activation_file="",
                generated_token_count=8,
                reached_eos=True,
                truncated=False,
                model_id="toy",
                model_revision="1",
                tokenizer_revision="1",
                seed=0,
            )
            array = np.arange(40, dtype=np.float32).reshape(8, 5)
            store.write(record, array)
            rows = store.read_manifest()
            self.assertEqual(len(rows), 1)
            np.testing.assert_allclose(store.load_activations(rows[0]), array)

    def test_trace_store_preserves_compact_float16_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(directory)
            record = TraceRecord(
                request_id="half",
                problem_id="p-half",
                task_family="test",
                condition="normal",
                split="test",
                layer=3,
                prompt_token_count=2,
                token_ids=list(range(8)),
                tokens=[],
                activation_file="",
                generated_token_count=8,
                reached_eos=True,
                truncated=False,
                model_id="toy",
                model_revision=None,
                tokenizer_revision=None,
                seed=0,
            )
            array = np.arange(40, dtype=np.float16).reshape(8, 5)
            store.write(record, array)
            with np.load(Path(directory) / "half.npz", allow_pickle=False) as shard:
                self.assertEqual(shard["activations"].dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
