import unittest

import numpy as np

from helix_role_experiment.readiness import (
    ExponentialProbe, SentenceSteeringController, assign_group_splits,
    build_survival_rows, concordance_index, fit_exponential_probe,
    sentence_boundaries, validate_annotations,
)


class CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(value) for value in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, ids, skip_special_tokens=False, **_kwargs):
        return "".join(chr(value) for value in ids)


class ReadinessTests(unittest.TestCase):
    def test_sentence_alignment_protects_decimal_and_latex(self):
        text = r"<think>Use 3.14 and $x=2.5$. Then solve.</think> FINAL: 4"
        rows = sentence_boundaries(CharacterTokenizer(), text, [ord(value) for value in text])
        self.assertIn("3.14", rows[0].text)
        self.assertIn("$x=2.5$", rows[0].text)
        self.assertTrue(rows[0].is_reasoning)
        self.assertFalse(rows[-1].is_reasoning)
        self.assertTrue(all(row.activation_index == row.token_start for row in rows))

    def test_validator_rejects_hallucinated_evidence(self):
        sentences = [{"sentence_id": "S0000", "text": "Therefore x=2."}]
        payload = {"annotations": [{
            "sentence_id": "S0000", "mathematically_correct": "yes", "novel": "yes",
            "advances_valid_path": "yes", "primary_label": "forward_progress",
            "evidence": "x=3", "state_change": "sets x", "needs_review": False,
        }]}
        with self.assertRaisesRegex(ValueError, "exact substring"):
            validate_annotations(sentences, payload)

    def test_survival_event_and_right_censoring(self):
        trace = {
            "trace_id": "t", "task_id": "p", "domain": "math", "split": "train",
            "output_token_count": 30,
            "sentences": [
                {"sentence_id": "S0000", "is_reasoning": True, "token_start": 0, "token_end": 5, "activation_index": 0},
                {"sentence_id": "S0001", "is_reasoning": True, "token_start": 5, "token_end": 12, "activation_index": 5},
                {"sentence_id": "S0002", "is_reasoning": True, "token_start": 12, "token_end": 20, "activation_index": 12},
            ],
        }
        annotations = [
            {"primary_label": "neutral_support"},
            {"primary_label": "forward_progress"},
            {"primary_label": "neutral_support"},
        ]
        rows = build_survival_rows(trace, annotations, 1)
        self.assertEqual((rows[0]["event"], rows[0]["duration"]), (1, 12))
        self.assertEqual((rows[1]["event"], rows[1]["duration"]), (1, 7))
        self.assertEqual((rows[2]["event"], rows[2]["duration"]), (0, 18))

    def test_group_split_is_domain_stratified_and_has_no_leakage(self):
        rows = [
            {"task_id": f"{domain}-{index}", "domain": domain}
            for domain in ("math", "code") for index in range(10)
        ]
        split = assign_group_splits(iter(rows), 4)
        for domain in ("math", "code"):
            values = [split[row["task_id"]] for row in rows if row["domain"] == domain]
            self.assertEqual(values.count("train"), 6)
            self.assertEqual(values.count("val"), 2)
            self.assertEqual(values.count("test"), 2)
        self.assertEqual(len(split), len(rows))

    def test_exponential_probe_recovers_risk_direction(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy unavailable")
        rng = np.random.default_rng(9)
        x = rng.normal(size=(500, 4))
        true = np.asarray([1.5, -0.7, 0.0, 0.0])
        rate = np.exp(x @ true - 1.0)
        event_time = rng.exponential(1 / rate)
        censor = rng.exponential(4.0, size=len(x))
        duration = np.minimum(event_time, censor)
        event = event_time <= censor
        probe = fit_exponential_probe(x, duration, event, ridge=1e-3)
        self.assertGreater(np.dot(probe.direction, true / np.linalg.norm(true)), 0.9)
        self.assertGreater(concordance_index(duration, event, probe.score(x)), 0.7)

    def test_streaming_controller_does_not_split_decimal_or_latex(self):
        probe = ExponentialProbe(
            mean=np.zeros(2), scale=np.ones(2), weights=np.ones(2),
            intercept=0.0, duration_scale=1.0, layer=0, threshold=-99,
        )
        controller = SentenceSteeringController(CharacterTokenizer(), probe, pulse_tokens=1)
        controller._boundary_pending = False
        for character in r"Value 3.14 is $x. y$":
            controller.observe_token(ord(character))
        self.assertFalse(controller._boundary_pending)
        for character in ". ":
            controller.observe_token(ord(character))
        self.assertTrue(controller._boundary_pending)


if __name__ == "__main__":
    unittest.main()
