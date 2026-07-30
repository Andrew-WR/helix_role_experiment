from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .behavioral import (
    extract_final_answer,
    normalize_answer,
    split_sentence_spans,
)
from .semantic_progress import token_span_for_char_span


EPS = 1e-12
ANSWER_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])(?P<sign>[-+]?)\s*"
    r"(?:\\(?:d?frac)\{(?P<numerator>\d+)\}\{(?P<denominator>\d+)\}"
    r"|(?P<slash_numerator>\d+)\s*/\s*(?P<slash_denominator>\d+)"
    r"|(?P<number>\d+(?:\.\d+)?))"
)
ASSIGNMENT_PATTERN = re.compile(
    r"\\?([A-Za-z]+)\s*=\s*([-+]?\d+(?:\.\d+)?)"
)
LATEX_COMMANDS = {
    "and",
    "at",
    "dfrac",
    "frac",
    "lambda",
    "left",
    "local",
    "maximum",
    "minimum",
    "or",
    "right",
    "text",
}


@dataclass(frozen=True)
class ProgressSignalFit:
    semantic_conditional: np.ndarray
    linear: np.ndarray
    k1_cosine: np.ndarray
    k1_sine: np.ndarray
    in_sample_incremental_r2: float
    blocked_cv_incremental_r2: float
    native_token_step_norm: float


def ordered_steps(problem: dict[str, Any]) -> list[str]:
    steps = problem.get("steps", {})
    if not isinstance(steps, dict) or not steps:
        raise ValueError("problem must contain a non-empty steps mapping")

    def key(value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", value)
        return (int(match.group(1)) if match else 10**9, value)

    return [str(steps[name]) for name in sorted(steps, key=key)]


def mean_pairwise_step_similarity(
    step_embeddings: np.ndarray,
) -> tuple[float, list[dict[str, float | int]]]:
    values = np.asarray(step_embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two step embeddings are required")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("step embeddings must have non-zero norm")
    values = values / norms
    similarity = values @ values.T
    rows: list[dict[str, float | int]] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            rows.append(
                {
                    "step_i": left + 1,
                    "step_j": right + 1,
                    "cosine_similarity": float(similarity[left, right]),
                }
            )
    return float(np.mean([row["cosine_similarity"] for row in rows])), rows


def _token_offsets(
    tokenizer,
    text: str,
    expected_token_ids: list[int] | None = None,
) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if expected_token_ids is not None:
        observed_ids = encoded.get("input_ids")
        if observed_ids is None:
            raise RuntimeError(
                "tokenizer did not return input_ids for alignment validation"
            )
        if observed_ids and isinstance(observed_ids[0], list):
            observed_ids = observed_ids[0]
        if [int(value) for value in observed_ids] != [
            int(value) for value in expected_token_ids
        ]:
            raise RuntimeError(
                "decoded generation does not re-tokenize to the captured token "
                "IDs; semantic token boundaries would be invalid"
            )
    return [tuple(int(item) for item in value) for value in encoded["offset_mapping"]]


def align_sentences_to_ordered_subgoals(
    tokenizer,
    text: str,
    output_tokens: int,
    sentence_embeddings: np.ndarray,
    step_embeddings: np.ndarray,
    threshold: float,
    expected_token_ids: list[int] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Align each generated sentence to the next authored subgoal.

    Progress advances only when the next *ordered* step exceeds the
    calibration-derived threshold. This prevents a later but similar sentence
    from receiving credit for skipped prerequisites.
    """

    spans = split_sentence_spans(text)
    sentence_values = np.asarray(sentence_embeddings, dtype=np.float64)
    step_values = np.asarray(step_embeddings, dtype=np.float64)
    if len(spans) != len(sentence_values):
        raise ValueError("one embedding is required for every generated sentence")
    if not len(step_values):
        raise ValueError("at least one subgoal is required")
    similarities = (
        sentence_values @ step_values.T
        if len(sentence_values)
        else np.empty((0, len(step_values)))
    )
    sentence_similarity = (
        sentence_values @ sentence_values.T
        if len(sentence_values)
        else np.empty((0, 0))
    )
    offsets = _token_offsets(tokenizer, text, expected_token_ids)
    if expected_token_ids is not None and len(expected_token_ids) != int(
        output_tokens
    ):
        raise ValueError("expected_token_ids length must equal output_tokens")
    token_progress = np.zeros(max(int(output_tokens), 1), dtype=np.float64)
    frontier = 0
    previous_token_end = 0
    recomputed_subgoal_count = 0
    pairwise_recurrence_count = 0
    _, final_answer_start = extract_final_answer(text)
    rows: list[dict[str, Any]] = []
    for sentence_index, span in enumerate(spans):
        token_start, token_end = token_span_for_char_span(offsets, span)
        token_start = min(token_start, len(token_progress))
        token_end = min(max(token_end, token_start), len(token_progress))
        if previous_token_end < token_start:
            token_progress[previous_token_end:token_start] = (
                frontier / len(step_values)
            )
        next_similarity = (
            float(similarities[sentence_index, frontier])
            if frontier < len(step_values)
            else float("nan")
        )
        prior_frontier = frontier
        advanced_steps: list[int] = []
        is_final_answer_sentence = bool(
            final_answer_start is not None and span.start >= final_answer_start
        )
        if (
            not is_final_answer_sentence
            and frontier < len(step_values)
            and float(similarities[sentence_index, frontier])
            >= float(threshold)
        ):
            advanced_steps.append(frontier + 1)
            frontier += 1
        advances = bool(advanced_steps)
        start_progress = prior_frontier / len(step_values)
        end_progress = frontier / len(step_values)
        if token_end > token_start:
            # The complete sentence establishes its subgoal only after its last
            # token. Earlier activations must not receive a retrospective label.
            token_progress[token_start:token_end] = start_progress
        prior_sentence_similarity = (
            float(np.max(sentence_similarity[sentence_index, :sentence_index]))
            if sentence_index
            else float("nan")
        )
        pairwise_recurrence = bool(
            not is_final_answer_sentence
            and sentence_index
            and prior_sentence_similarity >= float(threshold)
        )
        best_step = (
            int(np.argmax(similarities[sentence_index])) + 1
            if len(step_values)
            else None
        )
        completed_similarity = (
            float(np.max(similarities[sentence_index, :prior_frontier]))
            if prior_frontier
            else float("nan")
        )
        recomputed_subgoal = bool(
            not is_final_answer_sentence
            and
            not advances
            and prior_frontier
            and completed_similarity >= float(threshold)
            and best_step is not None
            and best_step <= prior_frontier
        )
        pairwise_recurrence_count += int(pairwise_recurrence)
        recomputed_subgoal_count += int(recomputed_subgoal)
        row: dict[str, Any] = {
            "sentence_index": sentence_index,
            "sentence": span.text,
            "is_final_answer_sentence": int(is_final_answer_sentence),
            "token_start": token_start,
            "token_end": token_end,
            "next_required_step": (
                prior_frontier + 1
                if prior_frontier < len(step_values)
                else None
            ),
            "next_step_similarity": next_similarity,
            "best_matching_step": best_step,
            "best_step_similarity": float(
                np.max(similarities[sentence_index])
            ),
            "advanced_frontier": int(advances),
            "advanced_step_count": len(advanced_steps),
            "advanced_steps": "|".join(str(value) for value in advanced_steps),
            "frontier_after_sentence": frontier,
            "semantic_progress_after_sentence": end_progress,
            "prior_sentence_max_similarity": prior_sentence_similarity,
            "completed_subgoal_max_similarity": completed_similarity,
            "recomputed_subgoal_sentence": int(recomputed_subgoal),
            "pairwise_sentence_recurrence": int(pairwise_recurrence),
            "threshold": float(threshold),
        }
        for step_index in range(len(step_values)):
            row[f"similarity_step_{step_index + 1}"] = float(
                similarities[sentence_index, step_index]
            )
        rows.append(row)
        previous_token_end = max(previous_token_end, token_end)
    if previous_token_end < len(token_progress):
        token_progress[previous_token_end:] = frontier / len(step_values)
    if int(output_tokens) == 0:
        token_progress = np.empty(0, dtype=np.float64)
    elif len(offsets) != int(output_tokens):
        raise RuntimeError(
            f"semantic alignment has {len(offsets)} tokenizer offsets for "
            f"{output_tokens} captured generation tokens"
        )
    summary = {
        "sentence_count": len(spans),
        "subgoal_count": len(step_values),
        "ordered_subgoals_completed": frontier,
        "ordered_subgoal_coverage": frontier / len(step_values),
        "recomputed_subgoal_sentence_count": recomputed_subgoal_count,
        "recomputed_subgoal_sentence_rate": (
            recomputed_subgoal_count / max(len(spans), 1)
        ),
        "pairwise_sentence_recurrence_count": pairwise_recurrence_count,
        "pairwise_sentence_recurrence_rate": (
            pairwise_recurrence_count / max(len(spans) - 1, 1)
        ),
        "semantic_efficiency_per_100_tokens": (
            100.0 * frontier / max(int(output_tokens), 1)
        ),
        "threshold": float(threshold),
    }
    return rows, token_progress, summary


def _ridge_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _designs(
    positions: np.ndarray,
    semantic_progress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(positions, dtype=np.float64)
    nuisance = np.column_stack(
        (
            np.ones(len(position)),
            position,
            np.cos(2.0 * np.pi * position),
            np.sin(2.0 * np.pi * position),
        )
    )
    return nuisance, np.column_stack((nuisance, semantic_progress))


def _blocked_incremental_r2(
    activations: np.ndarray,
    positions: np.ndarray,
    semantic_progress: np.ndarray,
    ridge: float,
    fold_count: int,
) -> float:
    nuisance, full = _designs(positions, semantic_progress)
    indices = np.arange(len(activations))
    folds = [
        fold for fold in np.array_split(indices, min(fold_count, len(indices)))
        if len(fold)
    ]
    nuisance_errors = []
    full_errors = []
    for test in folds:
        train = np.ones(len(indices), dtype=bool)
        train[test] = False
        if int(train.sum()) <= full.shape[1]:
            continue
        nuisance_prediction = nuisance[test] @ _ridge_coefficients(
            nuisance[train],
            activations[train],
            ridge,
        )
        full_prediction = full[test] @ _ridge_coefficients(
            full[train],
            activations[train],
            ridge,
        )
        nuisance_errors.append(
            float(np.mean(np.square(activations[test] - nuisance_prediction)))
        )
        full_errors.append(
            float(np.mean(np.square(activations[test] - full_prediction)))
        )
    if not nuisance_errors:
        return float("nan")
    denominator = max(float(np.mean(nuisance_errors)), EPS)
    return 1.0 - float(np.mean(full_errors)) / denominator


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _answer_number_signature(value: str) -> list[str]:
    output = []
    for match in ANSWER_NUMBER_PATTERN.finditer(value):
        sign = match.group("sign")
        if match.group("numerator") is not None:
            output.append(
                f"{sign}{match.group('numerator')}/{match.group('denominator')}"
            )
        elif match.group("slash_numerator") is not None:
            output.append(
                f"{sign}{match.group('slash_numerator')}/"
                f"{match.group('slash_denominator')}"
            )
        else:
            output.append(f"{sign}{match.group('number')}")
    return output


def fit_progress_signal(
    activations: np.ndarray,
    semantic_progress: np.ndarray,
    ridge: float = 1e-3,
    fold_count: int = 4,
) -> ProgressSignalFit:
    values = np.asarray(activations, dtype=np.float64)
    progress = np.asarray(semantic_progress, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(progress) or len(values) < 12:
        raise ValueError("fit requires aligned token activations and progress")
    positions = np.arange(len(values), dtype=np.float64) / max(len(values), 1)
    nuisance, full = _designs(positions, progress)
    nuisance_coefficients = _ridge_coefficients(nuisance, values, ridge)
    full_coefficients = _ridge_coefficients(full, values, ridge)
    nuisance_prediction = nuisance @ nuisance_coefficients
    full_prediction = full @ full_coefficients
    nuisance_mse = float(np.mean(np.square(values - nuisance_prediction)))
    full_mse = float(np.mean(np.square(values - full_prediction)))
    semantic_conditional = full_coefficients[-1]
    linear = _ridge_coefficients(
        np.column_stack((np.ones(len(values)), positions)),
        values,
        ridge,
    )[1]
    k1_cosine = nuisance_coefficients[2]
    k1_sine = nuisance_coefficients[3]
    if float(np.linalg.norm(semantic_conditional)) <= EPS:
        raise ValueError("conditional semantic direction collapsed")
    adjacent = np.diff(values, axis=0)
    native_token_step_norm = float(
        np.median(np.linalg.norm(adjacent, axis=1))
    )
    return ProgressSignalFit(
        semantic_conditional=semantic_conditional,
        linear=linear,
        k1_cosine=k1_cosine,
        k1_sine=k1_sine,
        in_sample_incremental_r2=1.0 - full_mse / max(nuisance_mse, EPS),
        blocked_cv_incremental_r2=_blocked_incremental_r2(
            values,
            positions,
            progress,
            ridge,
            fold_count,
        ),
        native_token_step_norm=native_token_step_norm,
    )


def answer_match(text: str, expected: str) -> tuple[bool, str, str | None]:
    """Conservative automatic check; every extracted answer is still exported."""

    observed, _ = extract_final_answer(text)
    if observed is None:
        return False, "missing_final", None
    observed_normalized = normalize_answer(observed)
    expected_normalized = normalize_answer(expected)
    observed_numbers = _answer_number_signature(observed)
    expected_numbers = _answer_number_signature(expected)
    if (
        observed_normalized == expected_normalized
        and observed_numbers == expected_numbers
    ):
        return True, "normalized_exact", observed
    expected_assignments = sorted(
        (name.casefold(), value)
        for name, value in ASSIGNMENT_PATTERN.findall(expected)
    )
    if expected_assignments:
        observed_assignments = sorted(
            (name.casefold(), value)
            for name, value in ASSIGNMENT_PATTERN.findall(observed)
        )
        if observed_assignments == expected_assignments:
            return True, "variable_assignment_signature", observed
        return False, "manual_review_required", observed
    observed_words = set(re.findall(r"[A-Za-z]+", observed.casefold()))
    expected_words = {
        word
        for word in re.findall(r"[A-Za-z]+", expected.casefold())
        if word not in LATEX_COMMANDS
    }
    if (
        expected_numbers
        and observed_numbers == expected_numbers
        and expected_words.issubset(observed_words)
    ):
        return True, "number_and_symbol_signature", observed
    return False, "manual_review_required", observed
