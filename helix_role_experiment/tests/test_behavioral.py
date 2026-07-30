import unittest

from helix_role_experiment.behavioral import (
    anchor_sentence_fraction,
    extract_final_answer,
    final_answer_is_correct,
    repeated_ngram_fraction,
    repeated_sentence_fraction,
    split_sentences,
)


class BehavioralTests(unittest.TestCase):
    def test_final_answer_requires_marker_and_matches_token(self):
        text = "I considered 12 first.\nFINAL: 2"
        self.assertEqual(extract_final_answer(text), ("2", text.index("FINAL:")))
        self.assertTrue(final_answer_is_correct(text, "2"))
        self.assertFalse(final_answer_is_correct(text, "12"))
        self.assertFalse(final_answer_is_correct("FINAL: NOT DONE", "DONE"))
        self.assertFalse(final_answer_is_correct("The result might be 2.", "2"))
        self.assertTrue(
            final_answer_is_correct(r"FINAL: \boxed{42}", "42")
        )
        self.assertTrue(
            final_answer_is_correct("FINAL: 2<|im_end|>", "2")
        )

    def test_anchor_and_repetition_metrics(self):
        text = (
            "First, I will make a plan. Wait, that was a mistake. "
            "Check the result. Check the result. FINAL: 4"
        )
        self.assertGreater(anchor_sentence_fraction(text), 0.5)
        self.assertGreater(repeated_sentence_fraction(text), 0.0)
        self.assertGreater(
            repeated_ngram_fraction([1, 2, 3, 4, 1, 2, 3, 4]),
            0.0,
        )

    def test_short_ngram_sequence_is_not_repetition(self):
        self.assertEqual(repeated_ngram_fraction([1, 2, 3]), 0.0)

    def test_sentence_splitter_protects_math_decimals_and_latex(self):
        text = (
            r"Let x=3.14 and use $f(x)=x^2;\ x>0$. "
            r"Compute it; then verify! FINAL: \boxed{9.8596}"
        )
        self.assertEqual(
            split_sentences(text),
            [
                r"Let x=3.14 and use $f(x)=x^2;\ x>0$.",
                "Compute it;",
                "then verify!",
                r"FINAL: \boxed{9.8596}",
            ],
        )


if __name__ == "__main__":
    unittest.main()
