from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from _common import write_csv
from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    read_jsonl,
    write_jsonl,
)
from helix_role_experiment.event_tagger import (
    EventExample,
    PROGRESS_LABELS,
    annotations_from_event_probabilities,
    binary_metrics,
    encode_event_context,
    inferred_event_label,
    is_progress_label,
    prior_event_memory,
    select_event_threshold,
)


SCRIPT_DIR = Path(__file__).resolve().parent
API_LABELER_PATH = SCRIPT_DIR / "07b_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location(
    "label_subgoal_events_07b_for_modernbert", API_LABELER_PATH
)
API_LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(API_LABELER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and apply a sequential ModernBERT progress-event tagger"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["prepare", "train", "apply"])
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Apply to only the first N missing trajectories for a smoke test",
    )
    return parser.parse_args()


def tagger_settings(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "model_id": "answerdotai/ModernBERT-base",
        "max_length": 2048,
        "recent_sentences": 8,
        "memory_dropout": 0.20,
        "negative_ratio": 6.0,
        "epochs": 100,
        "train_batch_size": 4,
        "inference_batch_size": 32,
        "gradient_accumulation": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "trainable_top_layers": 4,
        "review_margin": 0.08,
        "minimum_event_threshold": 0.50,
        "target_event_precision": 0.50,
        "memory_min_confidence": 0.80,
        "seed": int(config["study"].get("seed", 0)),
    }
    defaults.update(config.get("event_tagger", {}))
    return defaults


def model_directory(paths: dict[str, Path]) -> Path:
    return paths["models"] / "modernbert_event_tagger"


def load_traces(paths: dict[str, Path]) -> list[dict[str, Any]]:
    sources = sorted((paths["traces"] / "readiness_baseline").glob("*.json"))
    if not sources:
        raise RuntimeError("run 07a collection before training the event tagger")
    return [json.loads(source.read_text(encoding="utf-8")) for source in sources]


def load_labeled_data(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    API_LABELER.validate_cached(config, paths, allow_missing=True)
    traces = load_traces(paths)
    table = paths["tables"] / "sentence_annotations.jsonl"
    annotations = {
        str(row["trace_id"]): row["annotations"] for row in read_jsonl(table)
    }
    return traces, annotations


def apply_human_overrides(
    annotations: dict[str, list[dict[str, Any]]], override_path: Path
) -> int:
    if not override_path.exists():
        return 0
    changed = 0
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("human_label") or "").strip().casefold()
            if not value:
                continue
            trace_id = str(row["trace_id"])
            index = int(row["sentence_index"])
            if trace_id not in annotations or not 0 <= index < len(annotations[trace_id]):
                raise ValueError(f"override points to unknown sentence {trace_id}:{index}")
            if value in {"progress", "event", "forward", "forward_progress"}:
                label = "forward_progress"
            elif value in {"backtrack", "productive_backtrack"}:
                label = "productive_backtrack"
            elif value in {"other", "neutral", "neutral_support", "no"}:
                label = "neutral_support"
            else:
                raise ValueError(f"unknown human_label {value!r} at {trace_id}:{index}")
            annotations[trace_id][index]["primary_label"] = label
            annotations[trace_id][index]["needs_review"] = False
            changed += 1
    return changed


def prepare_seed_review(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    traces, annotations = load_labeled_data(config, paths)
    rng = random.Random(int(config["study"].get("seed", 0)))
    rows = []
    counts = {"train": {"progress": 0, "other": 0},
              "val": {"progress": 0, "other": 0},
              "test": {"progress": 0, "other": 0}}
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id not in annotations:
            continue
        candidates = []
        positives = []
        for index, (sentence, annotation) in enumerate(
            zip(trace["sentences"], annotations[trace_id], strict=True)
        ):
            if not sentence.get("is_reasoning", False):
                continue
            event = is_progress_label(annotation["primary_label"])
            counts[trace["split"]]["progress" if event else "other"] += 1
            row = {
                "trace_id": trace_id,
                "task_id": trace["task_id"],
                "domain": trace["domain"],
                "split": trace["split"],
                "sentence_index": index,
                "sentence_id": sentence["sentence_id"],
                "weak_label": annotation["primary_label"],
                "human_label": "",
                "previous_sentences": "\n".join(
                    value["text"] for value in trace["sentences"][max(0, index - 3):index]
                ),
                "text": sentence["text"],
            }
            candidates.append(row)
            if event:
                positives.append(row)
        rows.extend(positives)
        negatives = [row for row in candidates if row not in positives]
        rng.shuffle(negatives)
        rows.extend(negatives[:max(5, len(positives))])
    rows.sort(key=lambda row: (row["split"], row["domain"], row["trace_id"], row["sentence_index"]))
    destination = paths["tables"] / "modernbert_seed_review.csv"
    write_csv(destination, rows)
    report = {
        "labeled_trajectories": len(annotations),
        "total_trajectories": len(traces),
        "sentence_counts": counts,
        "review_rows": len(rows),
        "review_file": str(destination),
        "optional_override_file": str(
            paths["tables"] / "modernbert_seed_overrides.csv"
        ),
    }
    atomic_json(paths["tables"] / "modernbert_seed_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def selected_training_indices(
    trace: dict[str, Any], annotations: list[dict[str, Any]],
    negative_ratio: float, seed: int,
) -> set[int]:
    reasoning = [
        index for index, sentence in enumerate(trace["sentences"])
        if sentence.get("is_reasoning", False)
    ]
    positives = {
        index for index in reasoning
        if is_progress_label(annotations[index]["primary_label"])
    }
    if not positives:
        rng = random.Random(seed)
        sample = reasoning[:]
        rng.shuffle(sample)
        return set(sample[:min(len(sample), 32)])
    hard = {
        index for index in reasoning
        if any(abs(index - event) <= 2 for event in positives)
    }
    negatives = [index for index in reasoning if index not in positives and index not in hard]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    target_negatives = max(0, int(math.ceil(len(positives) * negative_ratio)) - len(hard - positives))
    return positives | hard | set(negatives[:target_negatives])


def build_examples(
    tokenizer: Any,
    traces: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    split: str,
    settings: dict[str, Any],
    epoch: int = 0,
    training: bool = False,
) -> list[EventExample]:
    result = []
    seed = int(settings["seed"]) + 1009 * epoch
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace["split"] != split or trace_id not in annotations:
            continue
        rows = annotations[trace_id]
        selected = (
            selected_training_indices(
                trace, rows, float(settings["negative_ratio"]),
                seed + int(trace_id[:8], 16) if all(c in "0123456789abcdef" for c in trace_id[:8].lower()) else seed,
            ) if training else {
                index for index, sentence in enumerate(trace["sentences"])
                if sentence.get("is_reasoning", False)
            }
        )
        for index in sorted(selected):
            rng = random.Random(seed + index * 7919 + len(result))
            memory = prior_event_memory(
                trace["sentences"], rows, index,
                dropout=float(settings["memory_dropout"]) if training else 0.0,
                rng=rng,
            )
            input_ids, attention_mask = encode_event_context(
                tokenizer, trace, index, memory,
                recent_sentences=int(settings["recent_sentences"]),
                max_length=int(settings["max_length"]),
            )
            result.append(EventExample(
                trace_id=trace_id,
                task_id=str(trace["task_id"]),
                domain=str(trace["domain"]),
                split=split,
                sentence_index=index,
                sentence_id=str(trace["sentences"][index]["sentence_id"]),
                label=int(is_progress_label(rows[index]["primary_label"])),
                input_ids=input_ids,
                attention_mask=attention_mask,
            ))
    return result


class ExampleDataset:
    def __init__(self, examples: list[EventExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EventExample:
        return self.examples[index]


def collator(tokenizer: Any):
    import torch

    def collate(examples: list[EventExample]) -> dict[str, Any]:
        encoded = tokenizer.pad(
            [{"input_ids": row.input_ids, "attention_mask": row.attention_mask}
             for row in examples],
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row.label for row in examples], dtype=torch.long)
        encoded["examples"] = examples
        return encoded

    return collate


def find_layers(model: Any) -> list[Any]:
    for candidate in (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "modernbert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "base_model", None), "encoder", None), "layer", None),
    ):
        if candidate is not None:
            return list(candidate)
    raise RuntimeError("could not locate ModernBERT transformer layers")


def freeze_for_small_data(model: Any, top_layers: int) -> dict[str, int]:
    base = model.base_model
    for parameter in base.parameters():
        parameter.requires_grad = False
    layers = find_layers(model)
    if top_layers <= 0 or top_layers > len(layers):
        raise ValueError(f"trainable_top_layers must be in [1, {len(layers)}]")
    for layer in layers[-top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    # The task head is outside base_model and remains trainable. Unfreeze final
    # normalization when its name is discoverable.
    for name, parameter in base.named_parameters():
        if name.startswith("final_norm."):
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total,
            "transformer_layers": len(layers), "trainable_top_layers": top_layers}


class LogitModule:
    @staticmethod
    def wrap(model: Any) -> Any:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

        return Wrapper(model)


def evaluate_model(
    wrapped: Any, examples: list[EventExample], tokenizer: Any,
    device: Any, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    if not examples:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    loader = DataLoader(
        ExampleDataset(examples), batch_size=batch_size, shuffle=False,
        collate_fn=collator(tokenizer),
    )
    wrapped.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch.pop("labels").tolist())
            batch.pop("examples")
            values = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = wrapped(**values)
            probabilities.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def evaluate_rollout(
    wrapped: Any, tokenizer: Any, traces: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]], split: str,
    settings: dict[str, Any], device: Any, memory_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate with predicted, not gold, event memory."""
    import torch

    states = {}
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace["split"] == split and trace_id in annotations:
            states[trace_id] = {"trace": trace, "index": 0, "memory": []}
    labels: list[int] = []
    probabilities: list[float] = []
    batch_size = int(settings["inference_batch_size"])
    wrapped.eval()
    with torch.no_grad():
        while states:
            ready = []
            for trace_id in list(states):
                state = states[trace_id]
                sentences = state["trace"]["sentences"]
                while state["index"] < len(sentences) and not sentences[state["index"]].get("is_reasoning", False):
                    state["index"] += 1
                if state["index"] >= len(sentences):
                    states.pop(trace_id)
                    continue
                ready.append((trace_id, state))
                if len(ready) >= batch_size:
                    break
            if not ready:
                continue
            encoded = []
            for _, state in ready:
                ids, mask = encode_event_context(
                    tokenizer, state["trace"], state["index"], state["memory"],
                    recent_sentences=int(settings["recent_sentences"]),
                    max_length=int(settings["max_length"]),
                )
                encoded.append({"input_ids": ids, "attention_mask": mask})
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = wrapped(**{key: value.to(device) for key, value in batch.items()})
            batch_probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
            for (trace_id, state), probability in zip(
                ready, batch_probabilities, strict=True
            ):
                index = state["index"]
                labels.append(int(is_progress_label(
                    annotations[trace_id][index]["primary_label"]
                )))
                probabilities.append(float(probability))
                if float(probability) >= memory_threshold:
                    text = state["trace"]["sentences"][index]["text"]
                    state["memory"].append((inferred_event_label(text), text))
                state["index"] += 1
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def train(config: dict[str, Any], paths: dict[str, Path]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT training requires a CUDA GPU in this experiment")
    settings = tagger_settings(config)
    random.seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    traces, annotations = load_labeled_data(config, paths)
    overrides = apply_human_overrides(
        annotations, paths["tables"] / "modernbert_seed_overrides.csv"
    )
    model_id = str(settings["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "other", 1: "progress_event"},
        label2id={"other": 0, "progress_event": 1},
        attn_implementation="sdpa",
    )
    parameter_report = freeze_for_small_data(
        model, int(settings["trainable_top_layers"])
    )
    device = torch.device("cuda:0")
    model.to(device)
    wrapped = LogitModule.wrap(model)
    if torch.cuda.device_count() > 1 and int(settings["train_batch_size"]) >= 2:
        wrapped = torch.nn.DataParallel(wrapped, device_ids=list(range(torch.cuda.device_count())))
    wrapped.to(device)

    validation = build_examples(
        tokenizer, traces, annotations, "val", settings, training=False
    )
    test_examples = build_examples(
        tokenizer, traces, annotations, "test", settings, training=False
    )
    if not validation:
        raise RuntimeError("at least one fully labeled validation trajectory is required")
    first_train = build_examples(
        tokenizer, traces, annotations, "train", settings, epoch=0, training=True
    )
    if not first_train or not any(row.label for row in first_train):
        raise RuntimeError("training labels contain no progress events")
    label_counts = np.bincount([row.label for row in first_train], minlength=2)
    weights = np.sqrt(len(first_train) / np.maximum(2 * label_counts, 1))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    accumulation = int(settings["gradient_accumulation"])
    steps_per_epoch = math.ceil(
        math.ceil(len(first_train) / int(settings["train_batch_size"])) / accumulation
    )
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(settings["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    destination = model_directory(paths)
    destination.mkdir(parents=True, exist_ok=True)
    best_key = (-1, -1.0, -1.0, -1.0)
    history = []
    global_step = 0

    for epoch in range(epochs):
        examples = build_examples(
            tokenizer, traces, annotations, "train", settings,
            epoch=epoch, training=True,
        )
        loader = DataLoader(
            ExampleDataset(examples),
            batch_size=int(settings["train_batch_size"]),
            shuffle=True,
            collate_fn=collator(tokenizer),
        )
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_index, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            batch.pop("examples")
            values = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = wrapped(**values)
                loss = criterion(logits.float(), labels) / accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * accumulation
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        teacher_labels, teacher_probabilities = evaluate_model(
            wrapped, validation, tokenizer, device,
            int(settings["inference_batch_size"]),
        )
        teacher_threshold, teacher_metrics = select_event_threshold(
            teacher_labels,
            teacher_probabilities,
            minimum_threshold=float(settings["minimum_event_threshold"]),
            target_precision=float(settings["target_event_precision"]),
        )
        val_labels, val_probabilities = evaluate_rollout(
            wrapped, tokenizer, traces, annotations, "val", settings, device,
            memory_threshold=max(
                teacher_threshold, float(settings["memory_min_confidence"])
            ),
        )
        threshold, metrics = select_event_threshold(
            val_labels,
            val_probabilities,
            minimum_threshold=float(settings["minimum_event_threshold"]),
            target_precision=float(settings["target_event_precision"]),
        )
        precision_target_met = (
            metrics["tp"] > 0
            and metrics["precision"] >= float(settings["target_event_precision"])
        )
        row = {
            "epoch": epoch + 1,
            "train_examples": len(examples),
            "mean_train_loss": running_loss / max(len(loader), 1),
            "threshold": threshold,
            "teacher_forced_threshold": teacher_threshold,
            "teacher_forced_val_f1": teacher_metrics["f1"],
            "precision_target_met": precision_target_met,
            "val_predicted_events": int(np.sum(val_probabilities >= threshold)),
            "val_predicted_event_rate": float(np.mean(val_probabilities >= threshold)),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (
            int(precision_target_met),
            metrics["recall"] if precision_target_met else metrics["precision"],
            metrics["precision"] if precision_target_met else metrics["f1"],
            metrics["f1"] if precision_target_met else metrics["recall"],
        )
        if key > best_key:
            best_key = key
            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
            atomic_json(destination / "calibration.json", {
                "event_threshold": threshold,
                "memory_threshold": max(
                    threshold, float(settings["memory_min_confidence"])
                ),
                "review_margin": float(settings["review_margin"]),
                "target_event_precision": float(settings["target_event_precision"]),
                "precision_target_met": precision_target_met,
                "validation_metrics": metrics,
                "epoch": epoch + 1,
            })

    best_model = AutoModelForSequenceClassification.from_pretrained(
        destination, attn_implementation="sdpa"
    ).to(device)
    best_wrapped = LogitModule.wrap(best_model).to(device)
    calibration = json.loads(
        (destination / "calibration.json").read_text(encoding="utf-8")
    )
    test_labels, test_probabilities = evaluate_rollout(
        best_wrapped, tokenizer, traces, annotations, "test", settings, device,
        memory_threshold=float(calibration["memory_threshold"]),
    )
    test_metrics = (
        binary_metrics(test_labels, test_probabilities >= calibration["event_threshold"])
        if len(test_labels) else {}
    )
    calibration["held_out_test_metrics"] = test_metrics
    atomic_json(destination / "calibration.json", calibration)
    manifest = {
        "model_id": model_id,
        "settings": settings,
        "human_overrides": overrides,
        "labeled_trajectories": len(annotations),
        "parameter_report": parameter_report,
        "initial_train_label_counts": label_counts.tolist(),
        "validation_examples": len(validation),
        "test_examples": len(test_examples),
        "history": history,
        "weak_supervision_warning": (
            "Labels not replaced in modernbert_seed_overrides.csv remain weak "
            "Luna/Qwen supervision; held-out metrics measure agreement, not truth."
        ),
    }
    atomic_json(destination / "training_manifest.json", manifest)
    print(
        f"Saved best ModernBERT event tagger to {destination}; "
        f"test metrics={test_metrics}",
        flush=True,
    )


def load_inference_model(paths: dict[str, Path]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = model_directory(paths)
    calibration_path = source / "calibration.json"
    if not calibration_path.exists():
        raise RuntimeError("train the ModernBERT event tagger before apply")
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, attn_implementation="sdpa"
    ).to("cuda:0")
    wrapped = LogitModule.wrap(model)
    if torch.cuda.device_count() > 1:
        wrapped = torch.nn.DataParallel(wrapped, device_ids=list(range(torch.cuda.device_count())))
    wrapped.to("cuda:0").eval()
    return tokenizer, wrapped, json.loads(calibration_path.read_text(encoding="utf-8"))


def pending_traces(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], int]:
    pending = []
    preserved = 0
    for source in API_LABELER.trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        if API_LABELER.materialize_chunked_result(trace, config, paths) is not None:
            preserved += 1
        else:
            pending.append(trace)
    return pending, preserved


def save_predicted_trace(
    trace: dict[str, Any], probabilities: list[float | None],
    calibration: dict[str, Any], paths: dict[str, Path], model_path: Path,
) -> None:
    threshold = float(calibration["event_threshold"])
    annotations = annotations_from_event_probabilities(
        trace, probabilities, threshold, float(calibration["review_margin"])
    )
    destination = API_LABELER.full_result_path(paths, str(trace["trace_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, {
        "trace_id": trace["trace_id"],
        "source": "modernbert_sequential_event_tagger",
        "model": str(model_path),
        "calibration": calibration,
        "event_probabilities": probabilities,
        "annotations": annotations,
    })


def write_prediction_review_queue(paths: dict[str, Path]) -> list[dict[str, Any]]:
    traces = {source.stem: json.loads(source.read_text(encoding="utf-8"))
              for source in API_LABELER.trace_files(paths)}
    rows = []
    directory = API_LABELER.result_directory(paths)
    for source in sorted(directory.glob("*.json")) if directory.exists() else []:
        record = json.loads(source.read_text(encoding="utf-8"))
        if record.get("source") != "modernbert_sequential_event_tagger":
            continue
        trace = traces.get(str(record["trace_id"]))
        if trace is None:
            continue
        threshold = float(record["calibration"]["event_threshold"])
        margin = float(record["calibration"]["review_margin"])
        for index, probability in enumerate(record["event_probabilities"]):
            if probability is None or not trace["sentences"][index].get("is_reasoning", False):
                continue
            event = float(probability) >= threshold
            uncertain = abs(float(probability) - threshold) <= margin
            if not event and not uncertain:
                continue
            rows.append({
                "trace_id": trace["trace_id"], "task_id": trace["task_id"],
                "domain": trace["domain"], "split": trace["split"],
                "sentence_index": index,
                "sentence_id": trace["sentences"][index]["sentence_id"],
                "probability": probability, "threshold": threshold,
                "suggested_label": (
                    inferred_event_label(trace["sentences"][index]["text"])
                    if event else "neutral_support"
                ),
                "review_reason": "predicted_event" if event else "near_threshold",
                "previous_sentences": "\n".join(
                    value["text"] for value in trace["sentences"][max(0, index - 3):index]
                ),
                "text": trace["sentences"][index]["text"],
            })
    write_jsonl(paths["tables"] / "modernbert_prediction_review.jsonl", rows)
    return rows


def apply(config: dict[str, Any], paths: dict[str, Path], limit: int | None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT inference requires a CUDA GPU in this experiment")
    settings = tagger_settings(config)
    pending, preserved = pending_traces(config, paths)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        pending = pending[:limit]
    print(
        f"Preserved {preserved} complete trajectories; applying ModernBERT to "
        f"{len(pending)} missing trajectories.", flush=True,
    )
    if not pending:
        API_LABELER.validate_cached(config, paths)
        return
    tokenizer, wrapped, calibration = load_inference_model(paths)
    device = torch.device("cuda:0")
    batch_size = int(settings["inference_batch_size"])
    states = {
        str(trace["trace_id"]): {
            "trace": trace,
            "index": 0,
            "memory": [],
            "probabilities": [None] * len(trace["sentences"]),
        } for trace in pending
    }
    total = len(states)
    completed = 0
    try:
        while states:
            ready = []
            for trace_id in list(states):
                state = states[trace_id]
                sentences = state["trace"]["sentences"]
                while state["index"] < len(sentences) and not sentences[state["index"]].get("is_reasoning", False):
                    state["index"] += 1
                if state["index"] >= len(sentences):
                    save_predicted_trace(
                        state["trace"], state["probabilities"], calibration,
                        paths, model_directory(paths),
                    )
                    completed += 1
                    print(f"[{completed}/{total}] saved {trace_id}", flush=True)
                    states.pop(trace_id)
                    continue
                ready.append((trace_id, state))
                if len(ready) >= batch_size:
                    break
            if not ready:
                continue
            encoded = []
            for _, state in ready:
                ids, mask = encode_event_context(
                    tokenizer, state["trace"], state["index"], state["memory"],
                    recent_sentences=int(settings["recent_sentences"]),
                    max_length=int(settings["max_length"]),
                )
                encoded.append({"input_ids": ids, "attention_mask": mask})
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits = wrapped(**{key: value.to(device) for key, value in batch.items()})
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
            for (trace_id, state), probability in zip(ready, probabilities, strict=True):
                index = state["index"]
                state["probabilities"][index] = float(probability)
                if float(probability) >= float(calibration["memory_threshold"]):
                    text = state["trace"]["sentences"][index]["text"]
                    state["memory"].append((inferred_event_label(text), text))
                state["index"] += 1
    except KeyboardInterrupt:
        print(
            "Paused safely. Completed trajectories are atomic; rerun apply to resume.",
            flush=True,
        )
    API_LABELER.validate_cached(
        config, paths, allow_missing=limit is not None or completed < total
    )
    review_rows = write_prediction_review_queue(paths)
    print(
        f"Prediction review queue contains {len(review_rows)} event/uncertain "
        f"sentences: {paths['tables'] / 'modernbert_prediction_review.jsonl'}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.command == "prepare":
        prepare_seed_review(config, paths)
    elif args.command == "train":
        train(config, paths)
    else:
        apply(config, paths, args.limit)


if __name__ == "__main__":
    main()
