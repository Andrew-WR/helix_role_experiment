from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .readiness import validate_annotations


PROGRESS_LABELS = frozenset({"forward_progress", "productive_backtrack"})
BACKTRACK_PATTERN = re.compile(
    r"\b(?:but|however|instead|mistake|incorrect|wrong|backtrack|reconsider|"
    r"return(?:ing)?\s+to|go\s+back|revise|correction|actually|scrap|abandon)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EventExample:
    trace_id: str
    task_id: str
    domain: str
    split: str
    sentence_index: int
    sentence_id: str
    label: int
    input_ids: list[int]
    attention_mask: list[int]


def is_progress_label(label: str) -> bool:
    return str(label) in PROGRESS_LABELS


def inferred_event_label(text: str) -> str:
    return (
        "productive_backtrack"
        if BACKTRACK_PATTERN.search(text)
        else "forward_progress"
    )


def prior_event_memory(
    sentences: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    sentence_index: int,
    dropout: float = 0.0,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    if len(sentences) != len(annotations):
        raise ValueError("sentence and annotation lengths differ")
    if not 0 <= dropout < 1:
        raise ValueError("memory dropout must be in [0, 1)")
    generator = rng or random.Random(0)
    result = []
    for sentence, annotation in zip(
        sentences[:sentence_index], annotations[:sentence_index], strict=True
    ):
        label = str(annotation["primary_label"])
        if label not in PROGRESS_LABELS or generator.random() < dropout:
            continue
        result.append((label, str(sentence["text"])))
    return result


def _head_tail(values: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    head = int(math.ceil(limit * 0.6))
    return values[:head] + values[-(limit - head):]


def _tail(values: list[int], limit: int) -> list[int]:
    return values if len(values) <= limit else values[-limit:]


def encode_event_context(
    tokenizer: Any,
    trace: dict[str, Any],
    sentence_index: int,
    event_memory: Iterable[tuple[str, str]],
    recent_sentences: int = 8,
    max_length: int = 2048,
) -> tuple[list[int], list[int]]:
    """Build a bounded state representation without rereading the trajectory.

    Task/reference retain their beginning and conclusion. Event memory and the
    recent context retain their newest tokens. The target always receives a
    dedicated budget and is placed last.
    """
    if max_length < 128:
        raise ValueError("max_length must be at least 128")
    if recent_sentences < 0:
        raise ValueError("recent_sentences cannot be negative")
    sentences = trace["sentences"]
    if not 0 <= sentence_index < len(sentences):
        raise IndexError("sentence index outside trajectory")
    memory = list(event_memory)
    memory_text = "\n".join(
        f"[{label}] {text}" for label, text in memory
    ) or "(none yet)"
    recent_start = max(0, sentence_index - recent_sentences)
    recent_text = "\n".join(
        f"[{index}] {sentences[index]['text']}"
        for index in range(recent_start, sentence_index)
    ) or "(start of reasoning)"
    sections = {
        "task": f"[TASK]\n{trace['prompt']}",
        "reference": f"[REFERENCE]\n{trace['reference_answer']}",
        "memory": f"[ACCEPTED PROGRESS EVENTS]\n{memory_text}",
        "recent": f"[RECENT SENTENCES]\n{recent_text}",
        "target": f"[TARGET SENTENCE]\n{sentences[sentence_index]['text']}",
    }
    encoded = {
        name: list(tokenizer.encode(text, add_special_tokens=False))
        for name, text in sections.items()
    }
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    budget = max_length - special - len(separator) * 4
    if budget <= 0:
        raise ValueError("max_length leaves no room after special tokens")

    # Fixed shares make runtime predictable. Any unused share is redistributed
    # to the most informative variable-length sections.
    fractions = {
        "task": 0.15,
        "reference": 0.20,
        "memory": 0.20,
        "recent": 0.20,
        "target": 0.25,
    }
    caps = {name: max(8, int(budget * value)) for name, value in fractions.items()}
    used = {name: min(len(encoded[name]), caps[name]) for name in encoded}
    spare = max(0, budget - sum(used.values()))
    for name in ("memory", "recent", "reference", "task", "target"):
        growth = min(spare, len(encoded[name]) - used[name])
        used[name] += growth
        spare -= growth
    selected = {
        "task": _head_tail(encoded["task"], used["task"]),
        "reference": _head_tail(encoded["reference"], used["reference"]),
        "memory": _tail(encoded["memory"], used["memory"]),
        "recent": _tail(encoded["recent"], used["recent"]),
        "target": _head_tail(encoded["target"], used["target"]),
    }
    combined: list[int] = []
    for name in ("task", "reference", "memory", "recent", "target"):
        if combined:
            combined.extend(separator)
        combined.extend(selected[name])
    input_ids = list(tokenizer.build_inputs_with_special_tokens(combined))
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    return input_ids, [1] * len(input_ids)


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(len(truth), 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


def select_event_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(truth) != len(scores):
        raise ValueError("labels and probabilities differ in length")
    if not np.any(truth == 1):
        return 0.5, binary_metrics(truth, scores >= 0.5)
    candidates = sorted(set(np.linspace(0.05, 0.95, 37).tolist() + scores.tolist()))
    best_threshold = 0.5
    best_metrics = binary_metrics(truth, scores >= best_threshold)
    for threshold in candidates:
        metrics = binary_metrics(truth, scores >= threshold)
        key = (metrics["f1"], metrics["precision"], metrics["recall"], threshold)
        best_key = (
            best_metrics["f1"], best_metrics["precision"],
            best_metrics["recall"], best_threshold,
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def annotations_from_event_probabilities(
    trace: dict[str, Any],
    probabilities: list[float | None],
    threshold: float,
    review_margin: float,
) -> list[dict[str, Any]]:
    if len(probabilities) != len(trace["sentences"]):
        raise ValueError("probability count must match sentence count")
    annotations = []
    for sentence, probability in zip(trace["sentences"], probabilities, strict=True):
        if not sentence.get("is_reasoning", False):
            label = "final_answer"
            correct, novel, advances = "uncertain", "no", "no"
            needs_review = True
        else:
            if probability is None or not np.isfinite(float(probability)):
                raise ValueError("reasoning sentence is missing an event probability")
            is_event = float(probability) >= threshold
            label = inferred_event_label(sentence["text"]) if is_event else "neutral_support"
            correct = "yes"
            novel = "yes" if is_event else "no"
            advances = "yes" if is_event else "no"
            needs_review = abs(float(probability) - threshold) <= review_margin
        annotations.append({
            "sentence_id": sentence["sentence_id"],
            "mathematically_correct": correct,
            "novel": novel,
            "advances_valid_path": advances,
            "primary_label": label,
            "evidence": sentence["text"] if label == "forward_progress" else "",
            "state_change": label if label in PROGRESS_LABELS else "",
            "needs_review": needs_review,
        })
    return validate_annotations(trace["sentences"], {"annotations": annotations})
