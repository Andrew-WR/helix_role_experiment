import unittest

from helix_role_experiment.semantic_progress import lexical_scores


class SemanticProgressTests(unittest.TestCase):
    def test_sentence_can_receive_multiple_lexical_roles(self):
        scores = lexical_scores(
            "Wait, let me recalculate and verify the result."
        )
        self.assertGreater(scores["uncertainty_management"], 0)
        self.assertGreater(scores["active_computation"], 0)
        self.assertGreater(scores["self_checking"], 0)

    def test_final_answer_rule_is_high_precision(self):
        scores = lexical_scores(r"FINAL: \boxed{42}")
        self.assertGreaterEqual(scores["final_answer_emission"], 2.0)


if __name__ == "__main__":
    unittest.main()
