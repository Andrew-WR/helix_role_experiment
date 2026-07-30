import re
import unittest

import numpy as np

from helix_role_experiment.subgoal_progress import (
    align_sentences_to_ordered_subgoals,
    answer_match,
    cosine_similarity,
    fit_progress_signal,
    mean_pairwise_step_similarity,
    ordered_steps,
)


class OffsetTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))
        return {"offset_mapping": [(match.start(), match.end()) for match in matches]}


class IdOffsetTokenizer(OffsetTokenizer):
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        output = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        output["input_ids"] = list(range(len(output["offset_mapping"])))
        return output


class SubgoalProgressTests(unittest.TestCase):
    def test_steps_sort_numerically(self):
        problem = {
            "steps": {
                "step_10": "ten",
                "step_2": "two",
                "step_1": "one",
            }
        }
        self.assertEqual(ordered_steps(problem), ["one", "two", "ten"])

    def test_threshold_is_mean_of_distinct_step_pairs(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]
        )
        threshold, rows = mean_pairwise_step_similarity(embeddings)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(threshold, 2 * (2**-0.5) / 3)

    def test_alignment_advances_only_next_ordered_step(self):
        text = "First calculation. Unrelated aside. Final calculation."
        sentence_embeddings = np.asarray(
            [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        )
        step_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        tokenizer = OffsetTokenizer()
        token_count = len(re.findall(r"\S+", text))
        rows, progress, summary = align_sentences_to_ordered_subgoals(
            tokenizer,
            text,
            token_count,
            sentence_embeddings,
            step_embeddings,
            0.5,
        )
        self.assertEqual([row["advanced_frontier"] for row in rows], [1, 0, 1])
        self.assertEqual(summary["ordered_subgoals_completed"], 2)
        self.assertAlmostEqual(progress[-1], 0.5)

    def test_one_sentence_cannot_cascade_across_subgoals(self):
        text = "Combined calculation."
        sentence_embeddings = np.asarray([[2**-0.5, 2**-0.5]])
        step_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        rows, progress, summary = align_sentences_to_ordered_subgoals(
            OffsetTokenizer(),
            text,
            2,
            sentence_embeddings,
            step_embeddings,
            0.7,
        )
        self.assertEqual(rows[0]["advanced_step_count"], 1)
        self.assertEqual(rows[0]["advanced_steps"], "1")
        self.assertEqual(summary["ordered_subgoals_completed"], 1)
        self.assertAlmostEqual(progress[-1], 0.0)

    def test_recomputation_requires_matching_completed_subgoal(self):
        text = "First calculation. First calculation again."
        sentence_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        step_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        rows, _, summary = align_sentences_to_ordered_subgoals(
            OffsetTokenizer(),
            text,
            5,
            sentence_embeddings,
            step_embeddings,
            0.5,
        )
        self.assertEqual(rows[1]["recomputed_subgoal_sentence"], 1)
        self.assertEqual(summary["recomputed_subgoal_sentence_count"], 1)

    def test_final_answer_does_not_receive_subgoal_credit(self):
        text = "FINAL: x = 1; y = 2."
        sentence_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        step_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        rows, progress, summary = align_sentences_to_ordered_subgoals(
            OffsetTokenizer(),
            text,
            len(re.findall(r"\S+", text)),
            sentence_embeddings,
            step_embeddings,
            0.5,
        )
        self.assertEqual(
            [row["is_final_answer_sentence"] for row in rows],
            [1, 1],
        )
        self.assertEqual(
            [row["advanced_step_count"] for row in rows],
            [0, 0],
        )
        self.assertEqual(summary["ordered_subgoals_completed"], 0)
        self.assertAlmostEqual(progress[-1], 0.0)

    def test_post_thinking_boxed_answer_cannot_advance_progress(self):
        text = (
            r"<think>First calculation.</think>"
            "\n"
            r"Therefore, \boxed{1}."
        )
        rows, _, summary = align_sentences_to_ordered_subgoals(
            OffsetTokenizer(),
            text,
            len(re.findall(r"\S+", text)),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            0.5,
        )
        self.assertEqual(rows[0]["advanced_step_count"], 1)
        self.assertEqual(rows[-1]["is_post_thinking_sentence"], 1)
        self.assertEqual(rows[-1]["is_final_answer_sentence"], 1)
        self.assertEqual(rows[-1]["advanced_step_count"], 0)
        self.assertEqual(summary["ordered_subgoals_completed"], 1)

    def test_progress_changes_only_after_sentence_boundary(self):
        text = "First calculation. FINAL: 1"
        sentence_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        step_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        rows, progress, _ = align_sentences_to_ordered_subgoals(
            OffsetTokenizer(),
            text,
            4,
            sentence_embeddings,
            step_embeddings,
            0.5,
        )
        self.assertEqual(rows[0]["token_end"], 2)
        np.testing.assert_allclose(progress, [0.0, 0.0, 0.5, 0.5])

    def test_token_id_round_trip_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "re-tokenize"):
            align_sentences_to_ordered_subgoals(
                IdOffsetTokenizer(),
                "First calculation.",
                2,
                np.asarray([[1.0, 0.0]]),
                np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                0.5,
                expected_token_ids=[99, 100],
            )

    def test_semantic_direction_is_conditional_on_controls(self):
        count = 80
        position = np.arange(count) / count
        semantic = np.repeat([0.0, 0.25, 0.5, 0.75], 20)
        activations = np.column_stack(
            (
                3.0 * position,
                np.cos(2 * np.pi * position),
                np.sin(2 * np.pi * position),
                2.0 * semantic,
                np.zeros(count),
            )
        )
        fit = fit_progress_signal(activations, semantic, fold_count=4)
        expected = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertGreater(
            cosine_similarity(fit.semantic_conditional, expected),
            0.99,
        )
        self.assertGreater(fit.in_sample_incremental_r2, 0.9)
        self.assertGreater(fit.native_token_step_norm, 0.0)

    def test_answer_match_exports_method(self):
        correct, method, answer = answer_match(
            "Work.\nFINAL: $\\lambda=1,\\lambda=3$",
            "$\\lambda = 1$ and $\\lambda = 3$.",
        )
        self.assertTrue(correct)
        self.assertEqual(method, "variable_assignment_signature")
        self.assertIsNotNone(answer)

    def test_answer_match_rejects_swapped_variable_values(self):
        correct, method, _ = answer_match(
            "Work.\nFINAL: $x=2,y=1$",
            "$x = 1, y = 2$",
        )
        self.assertFalse(correct)
        self.assertEqual(method, "manual_review_required")

    def test_answer_match_preserves_sign_and_fraction_order(self):
        equivalent, _, _ = answer_match(
            "Work.\nFINAL: $-1/6$",
            "$-\\dfrac{1}{6}$",
        )
        wrong_sign, _, _ = answer_match(
            "Work.\nFINAL: $\\dfrac{1}{6}$",
            "$-\\dfrac{1}{6}$",
        )
        reciprocal, _, _ = answer_match(
            "Work.\nFINAL: $\\dfrac{6}{1}$",
            "$\\dfrac{1}{6}$",
        )
        self.assertTrue(equivalent)
        self.assertFalse(wrong_sign)
        self.assertFalse(reciprocal)


if __name__ == "__main__":
    unittest.main()
