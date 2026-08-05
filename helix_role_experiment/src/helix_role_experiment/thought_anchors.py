from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def categorical_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    midpoint = 0.5 * (left + right)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(
            a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))
        ))

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
    """Evaluate present-versus-replacement sentence effects."""
    baseline = continuation_evaluator(sentences)
    rows: list[SentenceCounterfactual] = []
    for index, sentence in enumerate(sentences):
        replacements = replacement_sampler(index, sentence)
        if not replacements:
            raise ValueError(
                f"replacement sampler returned no sentence for index {index}"
            )
        metrics = []
        for replacement in replacements:
            changed = list(sentences)
            changed[index] = replacement
            evaluated = continuation_evaluator(changed)
            baseline_trajectory = np.asarray(baseline["subspace_trajectory"])
            changed_trajectory = np.asarray(evaluated["subspace_trajectory"])
            common = min(len(baseline_trajectory), len(changed_trajectory))
            metrics.append((
                categorical_js_divergence(
                    np.asarray(baseline["answer_probabilities"]),
                    np.asarray(evaluated["answer_probabilities"]),
                ),
                float(evaluated["correctness"]) - float(baseline["correctness"]),
                categorical_js_divergence(
                    np.asarray(baseline["operation_probabilities"]),
                    np.asarray(evaluated["operation_probabilities"]),
                ),
                float(np.linalg.norm(
                    changed_trajectory[:common] - baseline_trajectory[:common],
                    axis=-1,
                ).mean()),
                float(evaluated.get("downstream_causal_influence", 0.0)),
            ))
        values = np.asarray(metrics)
        rows.append(SentenceCounterfactual(
            sentence_index=index,
            replacement=replacements[0],
            answer_divergence=float(values[:, 0].mean()),
            correctness_effect=float(values[:, 1].mean()),
            operation_divergence=float(values[:, 2].mean()),
            subspace_transition_effect=float(values[:, 3].mean()),
            downstream_causal_influence=float(values[:, 4].mean()),
        ))
    return rows


def causal_link_approximation(
    baseline_logits: list[np.ndarray],
    suppressed_logits: list[list[np.ndarray]],
    sentence_token_spans: list[tuple[int, int]],
) -> np.ndarray:
    """Aggregate token-logit JS effects into a sentence-to-sentence matrix."""
    sentence_count = len(sentence_token_spans)
    output = np.zeros((sentence_count, sentence_count), dtype=np.float64)
    for source in range(sentence_count):
        if len(suppressed_logits[source]) != len(baseline_logits):
            raise ValueError("suppressed and baseline token positions are misaligned")
        for target, (start, end) in enumerate(sentence_token_spans):
            effects = []
            for token_index in range(start, end):
                p_logits = np.asarray(baseline_logits[token_index], dtype=np.float64)
                q_logits = np.asarray(
                    suppressed_logits[source][token_index], dtype=np.float64
                )
                p = np.exp(p_logits - p_logits.max())
                q = np.exp(q_logits - q_logits.max())
                p /= p.sum()
                q /= q.sum()
                effects.append(categorical_js_divergence(p, q))
            output[source, target] = (
                float(np.mean(effects)) if effects else 0.0
            )
    return output


def excess_kurtosis(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        return float("nan")
    centered = finite - finite.mean()
    variance = float(np.mean(centered * centered))
    if variance <= 0:
        return float("nan")
    return float(np.mean(centered ** 4) / (variance ** 2) - 3.0)


def rank_normalize_rows(
    sentence_attention: np.ndarray, proximity_ignore: int = 4
) -> np.ndarray:
    """Depth-control a causal sentence matrix as in Thought Anchors.

    Rows are downstream/query sentences and columns are prior/key sentences.
    Only sources at least ``proximity_ignore`` sentences earlier are retained.
    """
    values = np.asarray(sentence_attention, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("sentence attention must be a square matrix")
    if proximity_ignore < 1:
        raise ValueError("proximity_ignore must be positive")
    result = np.full_like(values, np.nan)
    for row in range(len(values)):
        stop = row - proximity_ignore + 1
        if stop <= 0:
            continue
        local = values[row, :stop]
        finite = np.isfinite(local)
        if not finite.any():
            continue
        selected = local[finite]
        order = np.argsort(selected, kind="stable")
        ranks = np.empty(len(selected), dtype=np.float64)
        ranks[order] = np.arange(1, len(selected) + 1, dtype=np.float64)
        normalized = ranks / len(selected)
        positions = np.flatnonzero(finite)
        result[row, positions] = normalized
    return result


def vertical_scores(
    sentence_attention: np.ndarray, proximity_ignore: int = 4
) -> np.ndarray:
    ranked = rank_normalize_rows(sentence_attention, proximity_ignore)
    scores = np.full(len(ranked), np.nan, dtype=np.float64)
    for source in range(len(ranked)):
        received = ranked[source + proximity_ignore :, source]
        if np.isfinite(received).any():
            scores[source] = float(np.nanmean(received))
    return scores


def receiver_head_statistics(
    matrices: np.ndarray, proximity_ignore: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-head vertical scores and kurtoses.

    ``matrices`` is ``[layers, heads, sentences, sentences]``.
    """
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
        raise ValueError("matrices must be [layers, heads, sentences, sentences]")
    scores = np.full(values.shape[:2] + (values.shape[-1],), np.nan)
    kurtosis = np.full(values.shape[:2], np.nan)
    for layer in range(values.shape[0]):
        for head in range(values.shape[1]):
            local = vertical_scores(values[layer, head], proximity_ignore)
            scores[layer, head] = local
            kurtosis[layer, head] = excess_kurtosis(local)
    return scores.astype(np.float32), kurtosis.astype(np.float32)


def top_fraction_flags(
    scores: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mark the top fraction of finite scores and return within-trace percentiles."""
    if not 0 < fraction < 1:
        raise ValueError("anchor fraction must be in (0, 1)")
    values = np.asarray(scores, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    flags = np.zeros(len(values), dtype=bool)
    percentiles = np.full(len(values), np.nan, dtype=np.float64)
    if not len(valid):
        return flags, percentiles
    order = valid[np.argsort(values[valid], kind="stable")]
    if len(order) == 1:
        percentiles[order] = 1.0
    else:
        percentiles[order] = np.arange(len(order), dtype=float) / (len(order) - 1)
    count = max(1, int(math.ceil(len(order) * fraction)))
    flags[order[-count:]] = True
    return flags, percentiles


def forward_anchor_overlap(
    anchor_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    annotations = {
        str(record["trace_id"]): record for record in annotation_records
        if record.get("source") != "modernbert_sequential_event_tagger"
    }
    totals = {
        "scored_sentences": 0, "anchors": 0, "forward_progress": 0,
        "productive_backtracks": 0, "forward_and_anchor": 0,
        "backtrack_and_anchor": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    for anchor_record in anchor_records:
        annotation_record = annotations.get(str(anchor_record["trace_id"]))
        if annotation_record is None:
            continue
        rows = annotation_record["annotations"]
        source = str(annotation_record.get("source", "unknown"))
        local = by_source.setdefault(source, {key: 0 for key in totals})
        index = {str(row["sentence_id"]): row for row in rows}
        for anchor in anchor_record["sentences"]:
            annotation = index.get(str(anchor["sentence_id"]))
            if annotation is None or anchor.get("score") is None:
                continue
            is_anchor = bool(anchor["thought_anchor"])
            label = str(annotation["primary_label"])
            increments = {
                "scored_sentences": 1,
                "anchors": int(is_anchor),
                "forward_progress": int(label == "forward_progress"),
                "productive_backtracks": int(label == "productive_backtrack"),
                "forward_and_anchor": int(is_anchor and label == "forward_progress"),
                "backtrack_and_anchor": int(
                    is_anchor and label == "productive_backtrack"
                ),
            }
            for key, value in increments.items():
                totals[key] += value
                local[key] += value

    def rates(counts: dict[str, int]) -> dict[str, Any]:
        forward = counts["forward_progress"]
        anchors = counts["anchors"]
        backtracks = counts["productive_backtracks"]
        return {
            **counts,
            "percent_llm_forward_that_are_anchors": (
                100.0 * counts["forward_and_anchor"] / forward if forward else None
            ),
            "percent_anchors_labeled_llm_forward": (
                100.0 * counts["forward_and_anchor"] / anchors if anchors else None
            ),
            "percent_llm_backtracks_that_are_anchors": (
                100.0 * counts["backtrack_and_anchor"] / backtracks
                if backtracks else None
            ),
        }

    return {
        "definition": (
            "Thought anchors are the configured top within-trace fraction of "
            "receiver-head scores. LLM rows exclude ModernBERT pseudo-labels."
        ),
        "overall": rates(totals),
        "by_llm_source": {key: rates(value) for key, value in by_source.items()},
    }
