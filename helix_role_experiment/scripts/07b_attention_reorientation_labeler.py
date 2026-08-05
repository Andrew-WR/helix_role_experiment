from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    read_jsonl,
    write_jsonl,
)
from helix_role_experiment.event_tagger import (
    annotations_from_event_probabilities,
    binary_metrics,
    select_event_threshold,
)
from helix_role_experiment.thought_anchors import (
    attention_burst_features,
    top_fraction_flags,
)


PSEUDO_SOURCE = "attention_burst_pseudo_labeler"
EXCLUDED_SEED_SOURCES = {
    PSEUDO_SOURCE,
    "attention_reorientation_pseudo_labeler",
    "modernbert_sequential_event_tagger",
}
POSITIVE_LABELS = {"forward_progress", "productive_backtrack"}
FEATURE_NAMES = (
    "stability",
    "recent_one",
    "recent_two",
    "local_echo",
    "transition_then_stabilization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a burst-progress attention labeler on strong train labels and "
            "pseudo-label only missing training trajectories"
        )
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def settings(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "target_precision": 0.25,
        "minimum_threshold": 0.05,
        "review_margin": 0.05,
        "echo_horizon": 3,
        "ridge": 1.0,
        "minimum_validation_precision": 0.13,
        "minimum_validation_recall": 0.25,
        "minimum_validation_lift": 2.0,
        "require_validation_gate": True,
    }
    values.update(config.get("attention_reorientation", {}))
    return values


def trace_files(paths: dict[str, Path]) -> list[Path]:
    return sorted((paths["traces"] / "readiness_baseline").glob("*.json"))


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def strong_annotations(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    path = paths["tables"] / "sentence_annotations.jsonl"
    if not path.exists():
        raise RuntimeError("sentence_annotations.jsonl is missing; restore the 15 labels")
    records = read_jsonl(path)
    return {
        str(record["trace_id"]): record
        for record in records
        if str(record.get("source", "unknown")) not in EXCLUDED_SEED_SOURCES
    }


def percentile_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    columns = []
    for name in FEATURE_NAMES:
        _, percentiles = top_fraction_flags(raw[name], 0.5)
        columns.append(percentiles)
    return np.column_stack(columns)


def trace_features(
    trace: dict[str, Any], paths: dict[str, Path], echo_horizon: int,
) -> np.ndarray:
    artifact = (
        paths["traces"] / "thought_anchor_attention"
        / f"{trace['trace_id']}.npz"
    )
    if not artifact.exists():
        raise RuntimeError(f"missing attention artifact {artifact}")
    with np.load(artifact) as data:
        if "sentence_attention" not in data.files:
            raise RuntimeError(
                f"{artifact} lacks sentence_attention; rerun Thought Anchor "
                "collection once with the current code"
            )
        attention = data["sentence_attention"].astype(np.float32)
        sentence_ids = data["sentence_ids"].astype(str).tolist()
    reasoning_ids = [
        str(row["sentence_id"]) for row in trace["sentences"]
        if row.get("is_reasoning", False)
    ]
    if sentence_ids != reasoning_ids:
        raise RuntimeError(f"attention/sentence alignment mismatch for {trace['trace_id']}")
    all_heads = attention.reshape(-1, attention.shape[-2], attention.shape[-1])
    return percentile_features(
        attention_burst_features(all_heads, echo_horizon=echo_horizon)
    )


def labels_for_trace(
    trace: dict[str, Any], record: dict[str, Any]
) -> np.ndarray:
    labels = {
        str(row["sentence_id"]): str(row["primary_label"])
        for row in record["annotations"]
    }
    reasoning = [row for row in trace["sentences"] if row.get("is_reasoning", False)]
    if any(str(row["sentence_id"]) not in labels for row in reasoning):
        raise RuntimeError(f"incomplete strong labels for {trace['trace_id']}")
    return np.asarray([
        labels[str(row["sentence_id"])] in POSITIVE_LABELS for row in reasoning
    ], dtype=np.int64)


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, ridge: float,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(x) != len(y) or x.ndim != 2:
        raise ValueError("invalid logistic training arrays")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("logistic training requires both classes")
    mean = np.nanmean(x, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale = np.nanstd(x, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (np.where(np.isfinite(x), x, mean) - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
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
        step = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-8, gradient)
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
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def labeled_groups(
    traces: list[dict[str, Any]], features: dict[str, np.ndarray],
    seeds: dict[str, dict[str, Any]], split: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = []
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace["split"] == split and trace_id in seeds:
            groups.append((
                trace_id, features[trace_id], labels_for_trace(trace, seeds[trace_id])
            ))
    return groups


def fit_oof_detector(
    train_groups: list[tuple[str, np.ndarray, np.ndarray]],
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    if len(train_groups) < 2:
        raise RuntimeError("attention burst calibration requires two train trajectories")
    oof_labels, oof_probabilities = [], []
    ridge = float(values["ridge"])
    for held_out, (_, held_features, held_labels) in enumerate(train_groups):
        fit_features = np.concatenate([
            group[1] for index, group in enumerate(train_groups)
            if index != held_out
        ])
        fit_labels = np.concatenate([
            group[2] for index, group in enumerate(train_groups)
            if index != held_out
        ])
        model = fit_logistic(fit_features, fit_labels, ridge)
        oof_probabilities.extend(predict_logistic(model, held_features).tolist())
        oof_labels.extend(held_labels.tolist())
    threshold, metrics = select_event_threshold(
        np.asarray(oof_labels),
        np.asarray(oof_probabilities),
        minimum_threshold=float(values["minimum_threshold"]),
        target_precision=float(values["target_precision"]),
    )
    all_features = np.concatenate([group[1] for group in train_groups])
    all_labels = np.concatenate([group[2] for group in train_groups])
    final_model = fit_logistic(all_features, all_labels, ridge)
    report = {
        "threshold": float(threshold),
        "oof_metrics": metrics,
        "precision_target_met": bool(
            metrics["tp"] > 0
            and metrics["precision"] >= float(values["target_precision"])
        ),
        "train_trajectory_count": len(train_groups),
        "feature_names": list(FEATURE_NAMES),
        "coefficients": {
            name: float(value) for name, value in zip(
                FEATURE_NAMES, np.asarray(final_model["coefficients"])
            )
        },
        "intercept": float(final_model["intercept"]),
        "imputation_mean": np.asarray(final_model["mean"]).tolist(),
        "standardization_scale": np.asarray(final_model["scale"]).tolist(),
    }
    return report, final_model


def evaluate_groups(
    groups: list[tuple[str, np.ndarray, np.ndarray]],
    model: dict[str, np.ndarray | float], threshold: float,
) -> dict[str, Any]:
    if not groups:
        return {"trajectory_count": 0, "unavailable": True}
    labels = np.concatenate([group[2] for group in groups])
    probabilities = np.concatenate([
        predict_logistic(model, group[1]) for group in groups
    ])
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    metrics = binary_metrics(labels, probabilities >= threshold)
    return {
        "trajectory_count": len(groups),
        "prevalence": prevalence,
        "precision_lift": (
            metrics["precision"] / prevalence if prevalence else None
        ),
        **metrics,
    }


def validation_gate(metrics: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    if metrics.get("unavailable"):
        return {"passed": False, "reasons": ["validation labels unavailable"]}
    required_precision = max(
        float(values["minimum_validation_precision"]),
        float(values["minimum_validation_lift"]) * float(metrics["prevalence"]),
    )
    reasons = []
    if float(metrics["precision"]) < required_precision:
        reasons.append(
            f"precision {metrics['precision']:.3f} < {required_precision:.3f}"
        )
    if float(metrics["recall"]) < float(values["minimum_validation_recall"]):
        reasons.append(
            f"recall {metrics['recall']:.3f} < "
            f"{float(values['minimum_validation_recall']):.3f}"
        )
    return {
        "passed": not reasons,
        "required_precision": required_precision,
        "required_recall": float(values["minimum_validation_recall"]),
        "reasons": reasons,
    }


def model_payload(model: dict[str, np.ndarray | float]) -> dict[str, Any]:
    return {
        "mean": np.asarray(model["mean"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "intercept": float(model["intercept"]),
        "coefficients": np.asarray(model["coefficients"]).tolist(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    values = settings(config)
    sources = trace_files(paths)
    if not sources:
        raise RuntimeError("run 07a collection first")
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    seeds = strong_annotations(paths)
    eligible = [
        trace for trace in traces
        if trace["split"] == "train" or str(trace["trace_id"]) in seeds
    ]
    features: dict[str, np.ndarray] = {}
    for index, trace in enumerate(eligible, start=1):
        features[str(trace["trace_id"])] = trace_features(
            trace, paths, int(values["echo_horizon"])
        )
        print(
            f"[attention burst {index}/{len(eligible)}] {trace['trace_id']}",
            flush=True,
        )

    train_groups = labeled_groups(traces, features, seeds, "train")
    val_groups = labeled_groups(traces, features, seeds, "val")
    test_groups = labeled_groups(traces, features, seeds, "test")
    detector, model = fit_oof_detector(train_groups, values)
    threshold = float(detector["threshold"])
    validation = evaluate_groups(val_groups, model, threshold)
    test = evaluate_groups(test_groups, model, threshold)
    gate = validation_gate(validation, values)
    calibration = {
        "detector": detector,
        "validation_metrics": validation,
        "test_metrics": test,
        "validation_gate": gate,
        "positive_labels": sorted(POSITIVE_LABELS),
        "pseudo_label_scope": "missing_train_trajectories_only",
        "echo_horizon": int(values["echo_horizon"]),
        "target_precision": float(values["target_precision"]),
        "model": model_payload(model),
    }
    atomic_json(
        paths["tables"] / "attention_reorientation_calibration.json",
        calibration,
    )

    order = {str(trace["trace_id"]): index for index, trace in enumerate(traces)}
    strong_rows = sorted(
        seeds.values(), key=lambda row: order.get(str(row["trace_id"]), 10**9)
    )
    atomic_write_jsonl(
        paths["tables"] / "sentence_annotations_strong_seed.jsonl", strong_rows
    )
    gate_required = bool(values["require_validation_gate"])
    if gate_required and not gate["passed"]:
        atomic_write_jsonl(paths["tables"] / "sentence_annotations.jsonl", strong_rows)
        print(json.dumps(calibration, indent=2), flush=True)
        print(
            "Validation gate failed; restored strong labels and wrote no "
            f"pseudo-labels. Reasons: {gate['reasons']}", flush=True,
        )
        return

    output = dict(seeds)
    score_records = []
    pseudo_count = 0
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id in seeds or trace["split"] != "train":
            continue
        local_features = features[trace_id]
        reasoning_probabilities = predict_logistic(model, local_features)
        probabilities: list[float | None] = []
        reasoning_index = 0
        sentence_scores = []
        for sentence in trace["sentences"]:
            if sentence.get("is_reasoning", False):
                probability = float(reasoning_probabilities[reasoning_index])
                feature_values = {
                    name: (
                        float(local_features[reasoning_index, feature_index])
                        if np.isfinite(local_features[reasoning_index, feature_index])
                        else None
                    )
                    for feature_index, name in enumerate(FEATURE_NAMES)
                }
                reasoning_index += 1
                probabilities.append(probability)
                sentence_scores.append({
                    "sentence_id": sentence["sentence_id"],
                    "progress_probability": probability,
                    "predicted_progress": bool(probability >= threshold),
                    "features": feature_values,
                })
            else:
                probabilities.append(None)
        annotations = annotations_from_event_probabilities(
            trace, probabilities, threshold, float(values["review_margin"])
        )
        output[trace_id] = {
            "trace_id": trace_id,
            "task_id": trace["task_id"],
            "domain": trace["domain"],
            "split": trace["split"],
            "source": PSEUDO_SOURCE,
            "detector_threshold": threshold,
            "annotations": annotations,
        }
        score_records.append({
            "trace_id": trace_id,
            "threshold": threshold,
            "sentences": sentence_scores,
        })
        pseudo_count += 1
    records = sorted(
        output.values(), key=lambda row: order.get(str(row["trace_id"]), 10**9)
    )
    atomic_write_jsonl(paths["tables"] / "sentence_annotations.jsonl", records)
    atomic_write_jsonl(
        paths["tables"] / "attention_reorientation_scores.jsonl", score_records
    )
    print(json.dumps(calibration, indent=2), flush=True)
    print(
        f"Validation gate passed. Preserved {len(seeds)} strong trajectories "
        f"and added {pseudo_count} burst-labeled training trajectories. "
        "Validation/test remain strong-only.", flush=True,
    )


if __name__ == "__main__":
    main()
