from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .event_tagger import binary_metrics, select_event_threshold


POSITIVE_LABELS = frozenset({"forward_progress", "productive_backtrack"})


@dataclass(frozen=True)
class LabeledTraceFeatures:
    trace_id: str
    domain: str
    split: str
    sentence_ids: tuple[str, ...]
    labels: np.ndarray
    features: np.ndarray


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array), array, 0.0)


def direct_prm_features(scores: np.ndarray) -> np.ndarray:
    """Turn step rewards into local trajectory features.

    A PRM estimates correctness/quality, not novelty.  The level alone is
    therefore retained as a baseline, while changes and local contrasts let a
    fitted detector test whether a new productive state is marked by a reward
    transition.
    """
    reward = _finite(np.asarray(scores).reshape(-1))
    if not len(reward):
        return np.empty((0, 7), dtype=np.float64)
    previous = np.r_[reward[0], reward[:-1]]
    delta = reward - previous
    prior_delta = np.r_[delta[0], delta[:-1]]
    second_delta = delta - prior_delta
    prior_mean = np.asarray([
        np.mean(reward[max(0, index - 3):index]) if index else reward[0]
        for index in range(len(reward))
    ])
    future_mean = np.asarray([
        np.mean(reward[index + 1:min(len(reward), index + 4)])
        if index + 1 < len(reward) else reward[index]
        for index in range(len(reward))
    ])
    prefix_min = np.minimum.accumulate(reward)
    prefix_max = np.maximum.accumulate(reward)
    return np.column_stack((
        reward,
        delta,
        second_delta,
        reward - prior_mean,
        future_mean - reward,
        reward - prefix_min,
        prefix_max - reward,
    ))


def prm_graph_node_vectors(scores: np.ndarray) -> np.ndarray:
    """Construct a non-semantic node state from a scalar PRM trajectory."""
    base = direct_prm_features(scores)
    if not len(base):
        return np.empty((0, 4), dtype=np.float64)
    # These components describe reward state, velocity, curvature, and its
    # short-horizon continuation.  They are deliberately distinct from the
    # direct PRM baseline; the graph is built from similarity between states.
    return base[:, [0, 1, 2, 4]]


def temporal_graph_features(node_vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute directed past/future support features on a temporal kNN graph.

    The graph uses only within-trajectory nodes.  Future support is legitimate
    here because this is an offline annotation experiment, not an online gate.
    """
    vectors = _finite(np.asarray(node_vectors, dtype=np.float64))
    if vectors.ndim != 2:
        raise ValueError("node vectors must be a two-dimensional array")
    if k < 1:
        raise ValueError("graph k must be positive")
    count = len(vectors)
    if not count:
        return np.empty((0, 10), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)

    past_max = np.zeros(count)
    past_mean = np.zeros(count)
    future_max = np.zeros(count)
    future_mean = np.zeros(count)
    local_change = np.zeros(count)
    future_indegree = np.zeros(count)
    future_inweight = np.zeros(count)

    for index in range(count):
        if index:
            values = similarity[index, :index]
            chosen = np.argsort(values)[-min(k, len(values)):]
            past_max[index] = float(np.max(values))
            past_mean[index] = float(np.mean(values[chosen]))
            local_change[index] = 1.0 - float(similarity[index, index - 1])
        if index + 1 < count:
            values = similarity[index, index + 1:]
            chosen = np.argsort(values)[-min(k, len(values)):]
            future_max[index] = float(np.max(values))
            future_mean[index] = float(np.mean(values[chosen]))

    # A sentence receives support when later sentences retrieve it among their
    # strongest prior neighbors.  This is the graph analogue of later uptake.
    for later in range(1, count):
        values = similarity[later, :later]
        chosen = np.argsort(values)[-min(k, len(values)):]
        for earlier in chosen:
            weight = max(float(values[earlier]), 0.0)
            future_indegree[earlier] += 1.0
            future_inweight[earlier] += weight
    remaining = np.maximum(count - np.arange(count) - 1, 1)
    future_indegree /= remaining
    future_inweight /= remaining

    novelty = 1.0 - past_max
    support_asymmetry = future_mean - past_mean
    branch_score = novelty * np.maximum(future_mean, 0.0)
    return np.column_stack((
        past_max,
        past_mean,
        future_max,
        future_mean,
        support_asymmetry,
        novelty,
        local_change,
        future_indegree,
        future_inweight,
        branch_score,
    ))


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, ridge: float = 1.0,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("invalid logistic training arrays")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("logistic training requires both classes")
    mean = np.nanmean(x, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (np.where(np.isfinite(x), x, mean) - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    positive_weight = len(y) / (2.0 * np.sum(y == 1))
    negative_weight = len(y) / (2.0 * np.sum(y == 0))
    sample_weight = np.where(y == 1, positive_weight, negative_weight)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    for _ in range(100):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probabilities - y))
        gradient += penalty @ coefficients
        curvature = sample_weight * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(
            hessian + np.eye(len(coefficients)) * 1e-8, gradient
        )
        coefficients -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break
    return {
        "mean": mean,
        "scale": scale,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
    }


def predict_logistic(
    model: dict[str, np.ndarray | float], features: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    standardized = (np.where(np.isfinite(x), x, mean) - mean) / scale
    logits = float(model["intercept"]) + standardized @ np.asarray(
        model["coefficients"], dtype=np.float64
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def fit_oof(
    groups: list[LabeledTraceFeatures], ridge: float,
    minimum_threshold: float, target_precision: float,
) -> tuple[dict[str, np.ndarray | float], float, dict[str, float]]:
    if len(groups) < 2:
        raise ValueError("at least two training trajectories are required")
    labels: list[int] = []
    probabilities: list[float] = []
    for held_index, held in enumerate(groups):
        fit_groups = [group for i, group in enumerate(groups) if i != held_index]
        model = fit_logistic(
            np.concatenate([group.features for group in fit_groups]),
            np.concatenate([group.labels for group in fit_groups]),
            ridge,
        )
        probabilities.extend(predict_logistic(model, held.features).tolist())
        labels.extend(held.labels.tolist())
    threshold, metrics = select_event_threshold(
        np.asarray(labels), np.asarray(probabilities),
        minimum_threshold=minimum_threshold,
        target_precision=target_precision,
    )
    final_model = fit_logistic(
        np.concatenate([group.features for group in groups]),
        np.concatenate([group.labels for group in groups]),
        ridge,
    )
    return final_model, float(threshold), metrics


def tolerant_event_metrics(
    groups: list[tuple[np.ndarray, np.ndarray]], tolerance: int = 1,
) -> dict[str, float]:
    """One-to-one event matching within each trajectory."""
    tp = fp = fn = 0
    for labels, predictions in groups:
        truth = np.flatnonzero(np.asarray(labels, dtype=bool))
        predicted = np.flatnonzero(np.asarray(predictions, dtype=bool))
        truth_index = predicted_index = matched = 0
        while truth_index < len(truth) and predicted_index < len(predicted):
            left = int(truth[truth_index])
            right = int(predicted[predicted_index])
            if abs(left - right) <= tolerance:
                matched += 1
                truth_index += 1
                predicted_index += 1
            elif right < left - tolerance:
                predicted_index += 1
            else:
                truth_index += 1
        tp += matched
        fp += len(predicted) - matched
        fn += len(truth) - matched
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "tp": float(tp), "fp": float(fp), "fn": float(fn),
    }


def evaluate_groups(
    groups: list[LabeledTraceFeatures], model: dict[str, np.ndarray | float],
    threshold: float,
) -> dict[str, Any]:
    if not groups:
        return {"trajectory_count": 0, "unavailable": True}
    scored = []
    for group in groups:
        probabilities = predict_logistic(model, group.features)
        scored.append((group, probabilities, probabilities >= threshold))
    labels = np.concatenate([item[0].labels for item in scored])
    predictions = np.concatenate([item[2] for item in scored])
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    exact = binary_metrics(labels, predictions)
    tolerant = tolerant_event_metrics([
        (item[0].labels, item[2]) for item in scored
    ], tolerance=1)
    domains: dict[str, Any] = {}
    for domain in sorted({item[0].domain for item in scored}):
        selected = [item for item in scored if item[0].domain == domain]
        domain_labels = np.concatenate([item[0].labels for item in selected])
        domain_predictions = np.concatenate([item[2] for item in selected])
        domains[domain] = {
            "trajectory_count": len(selected),
            "prevalence": float(np.mean(domain_labels)),
            "exact": binary_metrics(domain_labels, domain_predictions),
            "tolerant_1": tolerant_event_metrics([
                (item[0].labels, item[2]) for item in selected
            ]),
        }
    return {
        "trajectory_count": len(groups),
        "sentence_count": int(len(labels)),
        "prevalence": prevalence,
        "precision_lift": exact["precision"] / prevalence if prevalence else None,
        "exact": exact,
        "tolerant_1": tolerant,
        "by_domain": domains,
    }


def validation_gate(metrics: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if metrics.get("unavailable"):
        return {"passed": False, "reasons": ["validation labels unavailable"]}
    required_precision = max(
        float(settings["minimum_validation_precision"]),
        float(settings["minimum_validation_lift"]) * float(metrics["prevalence"]),
    )
    exact = metrics["exact"]
    reasons = []
    if exact["precision"] < required_precision:
        reasons.append(
            f"exact precision {exact['precision']:.3f} < {required_precision:.3f}"
        )
    if exact["recall"] < float(settings["minimum_validation_recall"]):
        reasons.append(
            f"exact recall {exact['recall']:.3f} < "
            f"{float(settings['minimum_validation_recall']):.3f}"
        )
    if metrics["tolerant_1"]["f1"] < float(settings["minimum_tolerant_f1"]):
        reasons.append(
            f"tolerant F1 {metrics['tolerant_1']['f1']:.3f} < "
            f"{float(settings['minimum_tolerant_f1']):.3f}"
        )
    return {
        "passed": not reasons,
        "required_exact_precision": required_precision,
        "required_exact_recall": float(settings["minimum_validation_recall"]),
        "required_tolerant_f1": float(settings["minimum_tolerant_f1"]),
        "reasons": reasons,
    }


def serializable_model(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        "mean": np.asarray(model["mean"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "intercept": float(model["intercept"]),
        "coefficients": np.asarray(model["coefficients"]).tolist(),
    }
