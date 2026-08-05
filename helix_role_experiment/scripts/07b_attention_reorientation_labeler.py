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
    attention_reorientation_scores,
    top_fraction_flags,
)


PSEUDO_SOURCE = "attention_reorientation_pseudo_labeler"
EXCLUDED_SEED_SOURCES = {
    PSEUDO_SOURCE,
    "modernbert_sequential_event_tagger",
}
POSITIVE_LABELS = {"forward_progress", "productive_backtrack"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate attention reorientation on strong labels and pseudo-label "
            "only missing training trajectories"
        )
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def settings(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "target_precision": 0.50,
        "minimum_threshold": 0.50,
        "review_margin": 0.05,
        "variants": [
            "all_head_median",
            "all_head_upper_quartile",
            "receiver_head_median",
        ],
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


def receiver_head_indices(
    paths: dict[str, Path], layers: list[int], head_count: int,
) -> list[tuple[int, int]]:
    path = paths["tables"] / "thought_anchor_receiver_heads.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = []
    for row in payload.get("heads", []):
        layer = int(row["layer"])
        head = int(row["head"])
        if layer in layers and 0 <= head < head_count:
            indices.append((layers.index(layer), head))
    return indices


def percentile_scores(raw_scores: np.ndarray) -> np.ndarray:
    _, percentiles = top_fraction_flags(raw_scores, 0.5)
    return percentiles


def trace_scores(
    trace: dict[str, Any], paths: dict[str, Path]
) -> dict[str, np.ndarray]:
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
        layers = data["layers"].astype(int).tolist()
        sentence_ids = data["sentence_ids"].astype(str).tolist()
    reasoning_ids = [
        str(row["sentence_id"]) for row in trace["sentences"]
        if row.get("is_reasoning", False)
    ]
    if sentence_ids != reasoning_ids:
        raise RuntimeError(f"attention/sentence alignment mismatch for {trace['trace_id']}")
    all_heads = attention.reshape(-1, attention.shape[-2], attention.shape[-1])
    selected = receiver_head_indices(paths, layers, attention.shape[1])
    receiver = np.asarray(
        [attention[layer, head] for layer, head in selected], dtype=np.float32
    ) if selected else all_heads
    variants = {
        "all_head_median": attention_reorientation_scores(all_heads, "median"),
        "all_head_upper_quartile": attention_reorientation_scores(
            all_heads, "upper_quartile"
        ),
        "receiver_head_median": attention_reorientation_scores(
            receiver, "median"
        ),
    }
    return {name: percentile_scores(score) for name, score in variants.items()}


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


def choose_detector(
    traces: list[dict[str, Any]],
    scores: dict[str, dict[str, np.ndarray]],
    seeds: dict[str, dict[str, Any]],
    values: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for variant in values["variants"]:
        train_labels, train_scores = [], []
        for trace in traces:
            trace_id = str(trace["trace_id"])
            if trace["split"] != "train" or trace_id not in seeds:
                continue
            labels = labels_for_trace(trace, seeds[trace_id])
            local_scores = scores[trace_id][variant]
            finite = np.isfinite(local_scores)
            train_labels.extend(labels[finite].tolist())
            train_scores.extend(local_scores[finite].tolist())
        if not train_labels:
            raise RuntimeError("no labeled training sentences for calibration")
        threshold, metrics = select_event_threshold(
            np.asarray(train_labels),
            np.asarray(train_scores),
            minimum_threshold=float(values["minimum_threshold"]),
            target_precision=float(values["target_precision"]),
        )
        candidates.append({
            "variant": variant,
            "threshold": float(threshold),
            **{f"train_{key}": value for key, value in metrics.items()},
            "precision_target_met": bool(
                metrics["tp"] > 0
                and metrics["precision"] >= float(values["target_precision"])
            ),
        })
    selected = max(candidates, key=lambda row: (
        int(row["precision_target_met"]),
        row["train_recall"] if row["precision_target_met"] else row["train_precision"],
        row["train_precision"], row["train_f1"], row["threshold"],
    ))
    return {"selected": selected, "candidates": candidates}


def split_metrics(
    traces: list[dict[str, Any]], scores: dict[str, dict[str, np.ndarray]],
    seeds: dict[str, dict[str, Any]], detector: dict[str, Any],
) -> dict[str, Any]:
    variant = str(detector["variant"])
    threshold = float(detector["threshold"])
    output = {}
    for split in ("train", "val", "test"):
        labels, predictions = [], []
        trace_count = 0
        for trace in traces:
            trace_id = str(trace["trace_id"])
            if trace["split"] != split or trace_id not in seeds:
                continue
            truth = labels_for_trace(trace, seeds[trace_id])
            local = scores[trace_id][variant]
            finite = np.isfinite(local)
            labels.extend(truth[finite].tolist())
            predictions.extend((local[finite] >= threshold).tolist())
            trace_count += 1
        output[split] = {
            "trajectory_count": trace_count,
            **binary_metrics(np.asarray(labels), np.asarray(predictions)),
        } if labels else {"trajectory_count": trace_count, "unavailable": True}
    return output


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
    scores: dict[str, dict[str, np.ndarray]] = {}
    for index, trace in enumerate(eligible, start=1):
        scores[str(trace["trace_id"])] = trace_scores(trace, paths)
        print(
            f"[attention reorientation {index}/{len(eligible)}] "
            f"{trace['trace_id']}", flush=True,
        )
    calibration = choose_detector(eligible, scores, seeds, values)
    detector = calibration["selected"]
    calibration["held_out_metrics"] = split_metrics(
        eligible, scores, seeds, detector
    )
    calibration["positive_labels"] = sorted(POSITIVE_LABELS)
    calibration["pseudo_label_scope"] = "missing_train_trajectories_only"
    calibration["target_precision"] = float(values["target_precision"])
    atomic_json(
        paths["tables"] / "attention_reorientation_calibration.json",
        calibration,
    )

    output = dict(seeds)
    score_records = []
    pseudo_count = 0
    variant = str(detector["variant"])
    threshold = float(detector["threshold"])
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id in seeds or trace["split"] != "train":
            continue
        reasoning_scores = scores[trace_id][variant]
        probabilities: list[float | None] = []
        reasoning_index = 0
        sentence_scores = []
        for sentence in trace["sentences"]:
            if sentence.get("is_reasoning", False):
                probability = float(reasoning_scores[reasoning_index])
                if not np.isfinite(probability):
                    probability = 0.0
                reasoning_index += 1
                probabilities.append(probability)
                sentence_scores.append({
                    "sentence_id": sentence["sentence_id"],
                    "reorientation_percentile": probability,
                    "predicted_progress": bool(probability >= threshold),
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
            "detector_variant": variant,
            "detector_threshold": threshold,
            "annotations": annotations,
        }
        score_records.append({
            "trace_id": trace_id,
            "variant": variant,
            "threshold": threshold,
            "sentences": sentence_scores,
        })
        pseudo_count += 1
    order = {str(trace["trace_id"]): index for index, trace in enumerate(traces)}
    records = sorted(output.values(), key=lambda row: order.get(str(row["trace_id"]), 10**9))
    atomic_write_jsonl(
        paths["tables"] / "sentence_annotations_strong_seed.jsonl",
        sorted(
            seeds.values(),
            key=lambda row: order.get(str(row["trace_id"]), 10**9),
        ),
    )
    atomic_write_jsonl(paths["tables"] / "sentence_annotations.jsonl", records)
    atomic_write_jsonl(
        paths["tables"] / "attention_reorientation_scores.jsonl", score_records
    )
    print(json.dumps(calibration, indent=2), flush=True)
    print(
        f"Preserved {len(seeds)} strong trajectories and added {pseudo_count} "
        "attention-labeled training trajectories. Validation/test remain strong-only.",
        flush=True,
    )


if __name__ == "__main__":
    main()
