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


def attention_reorientation_scores(
    sentence_attention: np.ndarray,
    aggregation: str = "median",
) -> np.ndarray:
    """Measure query-row redistribution at each sentence boundary.

    Input is ``[heads, query_sentence, key_sentence]``. Sentence ``i`` and
    sentence ``i-1`` are compared only over their shared causal history
    ``[:i-1]`` so the score is not merely attention to the immediately previous
    sentence. Scores for the first two sentences are undefined.
    """
    values = np.asarray(sentence_attention, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError(
            "sentence attention must be [heads, sentences, sentences]"
        )
    if aggregation not in {"median", "upper_quartile"}:
        raise ValueError("aggregation must be median or upper_quartile")
    scores = np.full(values.shape[-1], np.nan, dtype=np.float64)
    for sentence_index in range(2, values.shape[-1]):
        current = values[:, sentence_index, :sentence_index - 1]
        previous = values[:, sentence_index - 1, :sentence_index - 1]
        divergences = []
        for left, right in zip(current, previous):
            finite = np.isfinite(left) & np.isfinite(right)
            if not finite.any():
                continue
            left_values = np.maximum(left[finite], 0.0)
            right_values = np.maximum(right[finite], 0.0)
            if left_values.sum() <= 0 or right_values.sum() <= 0:
                continue
            divergences.append(
                categorical_js_divergence(left_values, right_values)
            )
        if divergences:
            quantile = 0.5 if aggregation == "median" else 0.75
            scores[sentence_index] = float(np.quantile(divergences, quantile))
    return scores


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


def attended_anchor_flags(
    sentence_attention: np.ndarray,
    primary_anchors: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one-hop ancestors most attended by each primary anchor.

    ``sentence_attention[query, key]`` must already be aggregated over the
    selected receiver heads. Only earlier key sentences are eligible. The
    returned percentile and score are the maximum obtained across primary
    anchor queries; this is deliberately not a recursive transitive closure.
    """
    values = np.asarray(sentence_attention, dtype=np.float64)
    primary = np.asarray(primary_anchors, dtype=bool)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("sentence attention must be a square matrix")
    if primary.shape != (len(values),):
        raise ValueError("primary anchor flags must match sentence attention")
    if not 0 < fraction < 1:
        raise ValueError("anchor fraction must be in (0, 1)")
    flags = np.zeros(len(values), dtype=bool)
    percentiles = np.full(len(values), np.nan, dtype=np.float64)
    scores = np.full(len(values), np.nan, dtype=np.float64)
    for query_index in np.flatnonzero(primary):
        candidates = np.arange(query_index, dtype=np.int64)
        finite = np.isfinite(values[query_index, :query_index])
        candidates = candidates[finite]
        if not len(candidates):
            continue
        order = candidates[np.argsort(
            values[query_index, candidates], kind="stable"
        )]
        if len(order) == 1:
            local_percentiles = np.asarray([1.0])
        else:
            local_percentiles = (
                np.arange(len(order), dtype=np.float64) / (len(order) - 1)
            )
        for sentence_index, percentile in zip(order, local_percentiles):
            score = values[query_index, sentence_index]
            if (
                not np.isfinite(percentiles[sentence_index])
                or percentile > percentiles[sentence_index]
            ):
                percentiles[sentence_index] = percentile
            if not np.isfinite(scores[sentence_index]) or score > scores[sentence_index]:
                scores[sentence_index] = score
        count = max(1, int(math.ceil(len(order) * fraction)))
        chosen = order[-count:]
        flags[chosen] = True
    return flags, percentiles, scores


def combined_anchor_scores(
    receiver_percentiles: np.ndarray,
    ancestor_percentiles: np.ndarray,
    receiver_weight: float,
) -> np.ndarray:
    """Blend receiver and ancestor ranks on the receiver-score support."""
    if not 0 <= receiver_weight <= 1:
        raise ValueError("receiver weight must be in [0, 1]")
    receiver = np.asarray(receiver_percentiles, dtype=np.float64)
    ancestor = np.asarray(ancestor_percentiles, dtype=np.float64)
    if receiver.shape != ancestor.shape:
        raise ValueError("receiver and ancestor percentiles must have equal shape")
    valid = np.isfinite(receiver)
    result = np.full(receiver.shape, np.nan, dtype=np.float64)
    result[valid] = (
        receiver_weight * receiver[valid]
        + (1.0 - receiver_weight)
        * np.nan_to_num(ancestor[valid], nan=0.0)
    )
    return result


def calibrate_anchor_selector(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    minimum_fraction: float,
    maximum_fraction: float,
    weight_steps: int = 11,
    fraction_steps: int = 7,
) -> dict[str, float | int]:
    """Tune a bounded selector using only supplied labeled train examples."""
    if not examples:
        raise ValueError("anchor calibration requires labeled train examples")
    if not 0 < minimum_fraction <= maximum_fraction < 1:
        raise ValueError("invalid final anchor fraction bounds")
    if weight_steps < 2 or fraction_steps < 2:
        raise ValueError("selector grids require at least two steps")
    best: dict[str, float | int] | None = None
    for weight in np.linspace(0.0, 1.0, weight_steps):
        for fraction in np.linspace(
            minimum_fraction, maximum_fraction, fraction_steps
        ):
            tp = fp = fn = 0
            for receiver, ancestor, labels in examples:
                scores = combined_anchor_scores(receiver, ancestor, float(weight))
                predictions, _ = top_fraction_flags(scores, float(fraction))
                truth = np.asarray(labels, dtype=bool)
                tp += int(np.sum(predictions & truth))
                fp += int(np.sum(predictions & ~truth))
                fn += int(np.sum(~predictions & truth))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            candidate: dict[str, float | int] = {
                "receiver_weight": float(weight),
                "final_anchor_fraction": float(fraction),
                "train_precision": float(precision),
                "train_recall": float(recall),
                "train_f1": float(f1),
                "train_tp": tp,
                "train_fp": fp,
                "train_fn": fn,
            }
            key = (f1, precision, -float(fraction))
            best_key = (
                float(best["train_f1"]),
                float(best["train_precision"]),
                -float(best["final_anchor_fraction"]),
            ) if best is not None else None
            if best_key is None or key > best_key:
                best = candidate
    assert best is not None
    return best


def forward_anchor_overlap(
    anchor_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    annotations = {
        str(record["trace_id"]): record for record in annotation_records
        if record.get("source") != "modernbert_sequential_event_tagger"
    }
    totals = {
        "scored_sentences": 0, "anchors": 0, "primary_anchors": 0,
        "anchor_of_anchor_sentences": 0, "forward_progress": 0,
        "productive_backtracks": 0, "forward_and_anchor": 0,
        "backtrack_and_anchor": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    by_split: dict[str, dict[str, int]] = {}
    for anchor_record in anchor_records:
        annotation_record = annotations.get(str(anchor_record["trace_id"]))
        if annotation_record is None:
            continue
        rows = annotation_record["annotations"]
        source = str(annotation_record.get("source", "unknown"))
        local = by_source.setdefault(source, {key: 0 for key in totals})
        split = str(anchor_record.get("split", "unknown"))
        local_split = by_split.setdefault(split, {key: 0 for key in totals})
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
                "primary_anchors": int(anchor.get("primary_thought_anchor", is_anchor)),
                "anchor_of_anchor_sentences": int(
                    anchor.get("anchor_of_anchor", False)
                ),
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
                local_split[key] += value

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
            "Thought anchors are selected under a bounded per-trajectory budget "
            "from receiver-score and one-hop ancestor-attention ranks. The "
            "selector is calibrated only on labeled training trajectories. "
            "LLM rows exclude ModernBERT pseudo-labels."
        ),
        "overall": rates(totals),
        "by_llm_source": {key: rates(value) for key, value in by_source.items()},
        "by_split": {key: rates(value) for key, value in by_split.items()},
    }
