from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def categorical_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    midpoint = 0.5 * (left + right)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


@dataclass
class SentenceCounterfactual:
    sentence_index: int
    replacement: str
    answer_divergence: float
    correctness_effect: float
    operation_divergence: float
    subspace_transition_effect: float
    downstream_causal_influence: float


def sentence_counterfactual_analysis(
    sentences: list[str],
    replacement_sampler: Callable[[int, str], list[str]],
    continuation_evaluator: Callable[[list[str]], dict[str, np.ndarray | float]],
) -> list[SentenceCounterfactual]:
    """Evaluate present-versus-replacement sentence effects.

    `continuation_evaluator` is deliberately backend-agnostic and should return
    answer probabilities, correctness, operation probabilities, a subspace
    trajectory, and optionally downstream token-logit influence.
    """

    baseline = continuation_evaluator(sentences)
    rows: list[SentenceCounterfactual] = []
    for index, sentence in enumerate(sentences):
        replacements = replacement_sampler(index, sentence)
        if not replacements:
            raise ValueError(f"replacement sampler returned no sentence for index {index}")
        metrics = []
        for replacement in replacements:
            changed = list(sentences)
            changed[index] = replacement
            evaluated = continuation_evaluator(changed)
            baseline_trajectory = np.asarray(baseline["subspace_trajectory"])
            changed_trajectory = np.asarray(evaluated["subspace_trajectory"])
            common = min(len(baseline_trajectory), len(changed_trajectory))
            metrics.append(
                (
                    categorical_js_divergence(
                        np.asarray(baseline["answer_probabilities"]),
                        np.asarray(evaluated["answer_probabilities"]),
                    ),
                    float(evaluated["correctness"]) - float(baseline["correctness"]),
                    categorical_js_divergence(
                        np.asarray(baseline["operation_probabilities"]),
                        np.asarray(evaluated["operation_probabilities"]),
                    ),
                    float(
                        np.linalg.norm(
                            changed_trajectory[:common] - baseline_trajectory[:common], axis=-1
                        ).mean()
                    ),
                    float(evaluated.get("downstream_causal_influence", 0.0)),
                )
            )
        values = np.asarray(metrics)
        rows.append(
            SentenceCounterfactual(
                sentence_index=index,
                replacement=replacements[0],
                answer_divergence=float(values[:, 0].mean()),
                correctness_effect=float(values[:, 1].mean()),
                operation_divergence=float(values[:, 2].mean()),
                subspace_transition_effect=float(values[:, 3].mean()),
                downstream_causal_influence=float(values[:, 4].mean()),
            )
        )
    return rows


def causal_link_approximation(
    baseline_logits: list[np.ndarray],
    suppressed_logits: list[list[np.ndarray]],
    sentence_token_spans: list[tuple[int, int]],
) -> np.ndarray:
    """Aggregate token-logit KL effects into a sentence-to-sentence matrix."""

    sentence_count = len(sentence_token_spans)
    output = np.zeros((sentence_count, sentence_count), dtype=np.float64)
    for source in range(sentence_count):
        if len(suppressed_logits[source]) != len(baseline_logits):
            raise ValueError("suppressed and baseline token positions are misaligned")
        for target, (start, end) in enumerate(sentence_token_spans):
            effects = []
            for token_index in range(start, end):
                p_logits = np.asarray(baseline_logits[token_index], dtype=np.float64)
                q_logits = np.asarray(suppressed_logits[source][token_index], dtype=np.float64)
                p = np.exp(p_logits - p_logits.max())
                q = np.exp(q_logits - q_logits.max())
                p /= p.sum()
                q /= q.sum()
                effects.append(categorical_js_divergence(p, q))
            output[source, target] = float(np.mean(effects)) if effects else 0.0
    return output

