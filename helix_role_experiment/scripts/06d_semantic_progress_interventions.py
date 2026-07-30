from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from _common import parse_layer_spec, write_csv
from helix_role_experiment.behavioral import split_sentence_spans
from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    environment_record,
    load_config,
    seed_everything,
)
from helix_role_experiment.models import huggingface_collector_from_config
from helix_role_experiment.semantic_progress import (
    embed_reasoning_sentences,
    load_sentence_embedder,
)
from helix_role_experiment.subgoal_progress import (
    align_sentences_to_ordered_subgoals,
    answer_match,
    cosine_similarity,
    fit_progress_signal,
    mean_pairwise_step_similarity,
    ordered_steps,
)


FINAL_LINE_STOP_REGEX = (
    r"(?im)^\s*\**final(?:\s+answer)?\**\s*:\**\s*\S[^\n]*"
    r"(?:\n|<\|im_end\|>)"
)
CONTROLS = (
    "semantic_forward",
    "semantic_reverse",
    "linear_forward",
    "closed_k1_forward",
)
MIN_BASELINE_SUBGOALS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a gold-subgoal progress direction on one fixed problem and "
            "causally test a short local transport pulse on one held-out problem."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset",
        default="data/undergrad_math_dataset_latex.json",
    )
    parser.add_argument("--calibration-id", type=int, default=9)
    parser.add_argument("--test-id", type=int, default=10)
    parser.add_argument(
        "--layers",
        default="16,20,24,28,31",
        help="Late candidate layers captured in the same two baseline rollouts.",
    )
    parser.add_argument(
        "--generation-safety-ceiling",
        type=int,
        default=2048,
        help=(
            "Emergency ceiling only. Normal generation stops at FINAL or EOS; "
            "this is not the intended reasoning budget."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--pulse-after-subgoal",
        type=int,
        default=1,
        help="Apply the held-out pulse immediately after this baseline subgoal.",
    )
    parser.add_argument("--pulse-tokens", type=int, default=4)
    parser.add_argument(
        "--transport-alpha",
        type=float,
        default=0.5,
        help=(
            "Pulse norm as a fraction of the calibration layer's median "
            "adjacent-token activation displacement."
        ),
    )
    parser.add_argument("--event-window-tokens", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--blocked-cv-folds", type=int, default=4)
    parser.add_argument(
        "--embedding-model-id",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args()
    if args.calibration_id == args.test_id:
        raise ValueError("calibration and held-out problem IDs must differ")
    if args.generation_safety_ceiling <= 0:
        raise ValueError("--generation-safety-ceiling must be positive")
    if args.pulse_after_subgoal <= 0:
        raise ValueError("--pulse-after-subgoal must be positive")
    if args.pulse_tokens <= 0:
        raise ValueError("--pulse-tokens must be positive")
    if args.transport_alpha <= 0:
        raise ValueError("--transport-alpha must be positive")
    if args.event_window_tokens <= 0:
        raise ValueError("--event-window-tokens must be positive")
    if args.blocked_cv_folds < 2:
        raise ValueError("--blocked-cv-folds must be at least 2")
    return args


def load_problems(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = int(payload.get("dataset_info", {}).get("num_problems", 0))
    raw_problems = payload.get("problems", [])
    if declared and declared != len(raw_problems):
        raise ValueError(
            f"dataset declares {declared} problems but contains {len(raw_problems)}"
        )
    problems: dict[int, dict[str, Any]] = {}
    for problem in raw_problems:
        problem_id = int(problem["id"])
        if problem_id in problems:
            raise ValueError(f"duplicate problem id {problem_id}")
        ordered_steps(problem)
        for required in ("topic", "question", "answer"):
            if not str(problem.get(required, "")).strip():
                raise ValueError(
                    f"problem {problem_id} has no non-empty {required}"
                )
        problems[problem_id] = problem
    if not problems:
        raise ValueError(f"no problems found in {path}")
    return problems


def evaluation_prompt(problem: dict[str, Any]) -> str:
    return (
        f"{problem['question']}\n\n"
        "Give a concise solution containing only the necessary mathematical "
        "steps. Put each distinct step in its own sentence; do not repeat, "
        "re-plan, or re-check completed work. End with a separate line beginning "
        "`FINAL:` followed only by the answer."
    )


def collect_generation(
    backend,
    prompt: str,
    layers: list[int],
    args: argparse.Namespace,
    seed: int,
    *,
    intervention=None,
    capture_activations: bool,
):
    return backend.collect(
        prompt,
        layers,
        args.generation_safety_ceiling,
        seed,
        temperature=args.temperature,
        disable_eos=False,
        intervention=intervention,
        capture_activations=capture_activations,
        capture_eos_logits=False,
        stop_regex=FINAL_LINE_STOP_REGEX,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_check_interval=4,
    )


def semantic_alignment(
    sentence_model,
    tokenizer,
    text: str,
    token_ids: list[int],
    step_embeddings: np.ndarray,
    threshold: float,
):
    spans = split_sentence_spans(text)
    sentence_embeddings = (
        embed_reasoning_sentences(
            sentence_model,
            [span.text for span in spans],
        )
        if spans
        else np.empty((0, step_embeddings.shape[1]), dtype=np.float32)
    )
    return align_sentences_to_ordered_subgoals(
        tokenizer,
        text,
        len(token_ids),
        sentence_embeddings,
        step_embeddings,
        threshold,
        expected_token_ids=token_ids,
    )


def unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("an intervention direction has zero norm")
    return array / norm


def control_delta(
    control: str,
    fit,
    target_norm: float,
    pulse_position: float,
    semantic_step: float,
) -> np.ndarray:
    if control == "semantic_forward":
        direction = fit.semantic_conditional
    elif control == "semantic_reverse":
        direction = -fit.semantic_conditional
    elif control == "linear_forward":
        direction = fit.linear
    elif control == "closed_k1_forward":
        destination = pulse_position + semantic_step
        direction = (
            fit.k1_cosine
            * (
                np.cos(2.0 * np.pi * destination)
                - np.cos(2.0 * np.pi * pulse_position)
            )
            + fit.k1_sine
            * (
                np.sin(2.0 * np.pi * destination)
                - np.sin(2.0 * np.pi * pulse_position)
            )
        )
    else:
        raise ValueError(f"unknown intervention control {control}")
    return unit(direction) * float(target_norm)


def pulse_callback(
    torch,
    delta: np.ndarray,
    pulse_start: int,
    pulse_tokens: int,
):
    applied_steps: list[int] = []
    pulse_end = int(pulse_start) + int(pulse_tokens)

    def callback(_layer, step, hidden):
        if int(pulse_start) <= int(step) < pulse_end:
            applied_steps.append(int(step))
            return hidden + torch.as_tensor(
                delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        return hidden

    return callback, applied_steps


def progress_auc(
    token_progress: np.ndarray,
    baseline_tokens: int,
) -> float:
    """Area under progress over the fixed held-out baseline horizon [0, 1]."""

    progress = np.asarray(token_progress, dtype=np.float64)
    if not len(progress) or baseline_tokens <= 0:
        return 0.0
    source = np.concatenate(
        (
            np.asarray([0.0]),
            (np.arange(len(progress), dtype=np.float64) + 1.0)
            / float(baseline_tokens),
        )
    )
    values = np.concatenate((np.asarray([0.0]), progress))
    grid = np.linspace(0.0, 1.0, 257)
    interpolated = np.interp(
        grid,
        source,
        values,
        left=0.0,
        right=float(progress[-1]),
    )
    return float(np.trapz(interpolated, grid))


def first_passage_metrics(
    alignment_rows: list[dict[str, Any]],
    subgoal_count: int,
    baseline_tokens: int,
) -> dict[str, float]:
    output = {}
    for subgoal in range(1, subgoal_count + 1):
        boundary = next(
            (
                int(row["token_end"])
                for row in alignment_rows
                if int(row["frontier_after_sentence"]) >= subgoal
            ),
            None,
        )
        output[f"first_passage_subgoal_{subgoal}_over_T_baseline"] = (
            float(boundary) / max(int(baseline_tokens), 1)
            if boundary is not None
            else float("nan")
        )
    return output


def heldout_partial_correlation(
    activations: np.ndarray,
    semantic_progress: np.ndarray,
    direction: np.ndarray,
) -> float:
    score = np.asarray(activations, dtype=np.float64) @ unit(direction)
    count = len(score)
    position = np.arange(count, dtype=np.float64) / max(count, 1)
    nuisance = np.column_stack(
        (
            np.ones(count),
            position,
            np.cos(2.0 * np.pi * position),
            np.sin(2.0 * np.pi * position),
        )
    )
    residual_score = score - nuisance @ np.linalg.lstsq(
        nuisance,
        score,
        rcond=None,
    )[0]
    residual_progress = semantic_progress - nuisance @ np.linalg.lstsq(
        nuisance,
        semantic_progress,
        rcond=None,
    )[0]
    denominator = float(
        np.linalg.norm(residual_score) * np.linalg.norm(residual_progress)
    )
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(residual_score, residual_progress) / denominator)


def event_locked_tracking(
    activations: np.ndarray,
    alignment_rows: list[dict[str, Any]],
    direction: np.ndarray,
    window: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score = np.asarray(activations, dtype=np.float64) @ unit(direction)
    rows = []
    for row in alignment_rows:
        boundary = int(row["token_end"])
        left = max(0, boundary - int(window))
        right = min(len(score), boundary + int(window))
        if boundary <= left or right <= boundary:
            continue
        delta = float(np.mean(score[boundary:right]) - np.mean(score[left:boundary]))
        rows.append(
            {
                "direction": label,
                "sentence_index": row["sentence_index"],
                "boundary_token": boundary,
                "advanced_step_count": row["advanced_step_count"],
                "frontier_after_sentence": row["frontier_after_sentence"],
                "score_delta": delta,
            }
        )
    advancing = [
        float(row["score_delta"])
        for row in rows
        if int(row["advanced_step_count"]) > 0
    ]
    nonadvancing = [
        float(row["score_delta"])
        for row in rows
        if int(row["advanced_step_count"]) == 0
    ]
    advancing_median = (
        float(np.median(advancing)) if advancing else float("nan")
    )
    nonadvancing_median = (
        float(np.median(nonadvancing)) if nonadvancing else float("nan")
    )
    return rows, {
        "direction": label,
        "event_count": len(rows),
        "advancing_event_count": len(advancing),
        "median_advancing_score_delta": advancing_median,
        "fraction_advancing_events_positive": (
            float(np.mean(np.asarray(advancing) > 0))
            if advancing
            else float("nan")
        ),
        "median_nonadvancing_score_delta": nonadvancing_median,
        "advancing_minus_nonadvancing_median": (
            advancing_median - nonadvancing_median
            if advancing and nonadvancing
            else float("nan")
        ),
    }


def outcome_row(
    control: str,
    problem: dict[str, Any],
    generation,
    alignment_rows: list[dict[str, Any]],
    token_progress: np.ndarray,
    alignment_summary: dict[str, Any],
    baseline_tokens: int,
    elapsed_seconds: float,
    target_norm: float,
    applied_steps: list[int],
) -> dict[str, Any]:
    correct, method, extracted = answer_match(
        generation.text,
        str(problem["answer"]),
    )
    output_tokens = len(generation.token_ids)
    thinking_tag_detected = bool(
        re.search(r"</?think>", generation.text, flags=re.IGNORECASE)
    )
    return {
        "control": control,
        "problem_id": problem["id"],
        "topic": problem["topic"],
        "expected_answer": problem["answer"],
        "extracted_final_answer": extracted,
        "correct_final": int(correct),
        "answer_match_method": method,
        "output_tokens": output_tokens,
        "output_tokens_over_T_baseline": (
            output_tokens / max(int(baseline_tokens), 1)
        ),
        "reached_eos": int(generation.reached_eos),
        "thinking_tag_detected": int(thinking_tag_detected),
        "progress_auc_to_T_baseline": progress_auc(
            token_progress,
            baseline_tokens,
        ),
        "pulse_target_norm": float(target_norm),
        "pulse_application_count": len(applied_steps),
        "pulse_applied_steps": "|".join(str(value) for value in applied_steps),
        "elapsed_seconds": elapsed_seconds,
        **alignment_summary,
        **first_passage_metrics(
            alignment_rows,
            int(alignment_summary["subgoal_count"]),
            baseline_tokens,
        ),
        "generated_text": generation.text,
    }


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["control"] == "baseline")
    for row in rows:
        for metric in (
            "output_tokens",
            "ordered_subgoal_coverage",
            "semantic_efficiency_per_100_tokens",
            "recomputed_subgoal_sentence_count",
            "pairwise_sentence_recurrence_count",
            "progress_auc_to_T_baseline",
        ):
            delta = float(row[metric]) - float(baseline[metric])
            row[f"delta_{metric}"] = delta
            row[f"percent_delta_{metric}"] = (
                100.0 * delta / float(baseline[metric])
                if float(baseline[metric])
                else float("nan")
            )


def pulse_start_after_subgoal(
    alignment_rows: list[dict[str, Any]],
    subgoal: int,
    baseline_tokens: int,
) -> int:
    value = next(
        (
            int(row["token_end"])
            for row in alignment_rows
            if int(row["frontier_after_sentence"]) >= int(subgoal)
        ),
        None,
    )
    if value is None:
        raise RuntimeError(
            f"held-out baseline never completed subgoal {subgoal}; "
            "the preregistered pulse time is undefined"
        )
    return min(max(value, 1), max(int(baseline_tokens) - 1, 1))


def falsification_gates(
    fit,
    heldout_correlation: float,
    event_summary: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_control = {row["control"]: row for row in outcomes}
    baseline = by_control["baseline"]
    semantic = by_control["semantic_forward"]
    reverse = by_control["semantic_reverse"]
    comparators = [
        by_control["linear_forward"],
        by_control["closed_k1_forward"],
    ]

    def gate(name: str, passed: bool, value: Any, criterion: str):
        return {
            "gate": name,
            "status": "PASS" if passed else "FALSIFIED",
            "value": value,
            "criterion": criterion,
        }

    baseline_valid = bool(
        baseline["correct_final"]
        and baseline["ordered_subgoals_completed"] >= MIN_BASELINE_SUBGOALS
        and not baseline["thinking_tag_detected"]
    )
    return [
        gate(
            "heldout_baseline_valid",
            baseline_valid,
            (
                f"correct={baseline['correct_final']}; "
                f"coverage={baseline['ordered_subgoal_coverage']:.3f}"
            ),
            "held-out baseline is correct and has measurable gold progress",
        ),
        gate(
            "calibration_encoding_increment",
            np.isfinite(fit.blocked_cv_incremental_r2)
            and fit.blocked_cv_incremental_r2 > 0,
            fit.blocked_cv_incremental_r2,
            "calibration blocked-CV incremental R2 beyond position and k=1 > 0",
        ),
        gate(
            "heldout_observational_tracking",
            (
                np.isfinite(heldout_correlation)
                and heldout_correlation > 0
                and np.isfinite(event_summary["median_advancing_score_delta"])
                and event_summary["median_advancing_score_delta"] > 0
            ),
            (
                f"partial_r={heldout_correlation:.4f}; "
                "median_event_delta="
                f"{event_summary['median_advancing_score_delta']:.4f}"
            ),
            "frozen direction has positive global and event-locked held-out tracking",
        ),
        gate(
            "accuracy_and_coverage_preserved",
            bool(
                semantic["correct_final"]
                and semantic["ordered_subgoal_coverage"]
                >= baseline["ordered_subgoal_coverage"]
            ),
            (
                f"baseline_correct={baseline['correct_final']}; "
                f"semantic_correct={semantic['correct_final']}; "
                f"coverage_delta={semantic['delta_ordered_subgoal_coverage']:.3f}"
            ),
            "semantic-forward stays correct and does not reduce ordered coverage",
        ),
        gate(
            "causal_progress_improves_over_baseline",
            bool(
                semantic["correct_final"]
                and semantic["ordered_subgoal_coverage"]
                >= baseline["ordered_subgoal_coverage"]
                and semantic["progress_auc_to_T_baseline"]
                > baseline["progress_auc_to_T_baseline"]
            ),
            (
                f"baseline={baseline['progress_auc_to_T_baseline']:.4f}; "
                f"semantic={semantic['progress_auc_to_T_baseline']:.4f}"
            ),
            "semantic-forward preserves accuracy/coverage and improves progress AUC",
        ),
        gate(
            "causal_specificity_vs_linear_and_k1",
            bool(
                semantic["correct_final"]
                and semantic["progress_auc_to_T_baseline"]
                > max(row["progress_auc_to_T_baseline"] for row in comparators)
            ),
            (
                f"semantic={semantic['progress_auc_to_T_baseline']:.4f}; "
                f"linear={comparators[0]['progress_auc_to_T_baseline']:.4f}; "
                f"k1={comparators[1]['progress_auc_to_T_baseline']:.4f}"
            ),
            "accuracy-preserving semantic-forward progress AUC beats linear and k=1",
        ),
        gate(
            "direction_sign_test",
            bool(
                semantic["progress_auc_to_T_baseline"]
                > reverse["progress_auc_to_T_baseline"]
            ),
            (
                f"forward={semantic['progress_auc_to_T_baseline']:.4f}; "
                f"reverse={reverse['progress_auc_to_T_baseline']:.4f}"
            ),
            "semantic-forward progress AUC exceeds equal-norm semantic-reverse",
        ),
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["model"].setdefault("chat_template_kwargs", {})[
        "enable_thinking"
    ] = False
    paths = ensure_output_dirs(config)
    seed = int(config["study"]["seed"]) + 664
    seed_everything(seed)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[1] / dataset_path
    problems = load_problems(dataset_path)
    try:
        calibration_problem = problems[args.calibration_id]
        test_problem = problems[args.test_id]
    except KeyError as exc:
        raise ValueError(f"unknown problem id {exc.args[0]}") from exc
    calibration_steps = ordered_steps(calibration_problem)
    test_steps = ordered_steps(test_problem)
    if args.pulse_after_subgoal >= len(test_steps):
        raise ValueError(
            "--pulse-after-subgoal must leave at least one held-out subgoal ahead"
    )
    calibration_prompt = evaluation_prompt(calibration_problem)
    test_prompt = evaluation_prompt(test_problem)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    environment = environment_record(config)
    design_basis = {
        "experiment": "06d_gold_subgoal_progress_pilot",
        "arguments": vars(args),
        "dataset_sha256": dataset_sha256,
        "config_hash": environment["config_hash"],
        "repository_commit": environment["repository_commit"],
    }
    design_hash = hashlib.sha256(
        json.dumps(
            design_basis,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        "experiment": "06d_gold_subgoal_progress_pilot",
        "design_status": "fixed_before_generation",
        "calibration_problem_id": args.calibration_id,
        "heldout_problem_id": args.test_id,
        "reserved_problem_ids": sorted(
            set(problems) - {args.calibration_id, args.test_id}
        ),
        "threshold_definition": (
            "per-problem mean cosine over distinct pairs of gold-step embeddings"
        ),
        "threshold_frozen_across_conditions": True,
        "reasoning_mode_requested": False,
        "arguments": vars(args),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "design_hash": design_hash,
        "environment": environment,
    }
    manifest_path = paths["tables"] / "semantic_run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("design_hash") != design_hash:
            raise RuntimeError(
                f"{paths['root']} contains a different 06d design hash "
                f"({previous.get('design_hash')}); use a new output.root "
                "rather than mixing pilots"
            )
    for stale_name in (
        "semantic_key_results.csv",
        "semantic_falsification_gates.csv",
        "semantic_intervention_outcomes.csv",
    ):
        write_csv(paths["tables"] / stale_name, [])
    atomic_json(manifest_path, manifest)
    write_csv(
        paths["tables"] / "semantic_subgoal_experiment_design.csv",
        [
            {
                "role": "calibration",
                "problem_id": calibration_problem["id"],
                "topic": calibration_problem["topic"],
                "question": calibration_problem["question"],
                "steps": calibration_steps,
                "answer": calibration_problem["answer"],
            },
            {
                "role": "heldout_test",
                "problem_id": test_problem["id"],
                "topic": test_problem["topic"],
                "question": test_problem["question"],
                "steps": test_steps,
                "answer": test_problem["answer"],
            },
        ],
    )
    print(
        "Loading Qwen3.5-9B with the chat template enabled and thinking disabled; "
        f"calibration={args.calibration_id}, heldout={args.test_id}",
        flush=True,
    )
    backend = huggingface_collector_from_config(
        config["model"],
        config["collection"],
    )
    if not getattr(backend.tokenizer, "chat_template", None):
        raise RuntimeError(
            "Qwen chat template is unavailable; the no-thinking prompt contract "
            "cannot be verified"
        )
    candidate_layers = parse_layer_spec(
        args.layers,
        list(range(len(backend.layers))),
    )
    baseline_generations = {}
    baseline_seconds = {}
    started = time.perf_counter()
    baseline_generations["calibration"] = collect_generation(
        backend,
        calibration_prompt,
        candidate_layers,
        args,
        seed,
        capture_activations=True,
    )
    baseline_seconds["calibration"] = time.perf_counter() - started
    print(
        "calibration: "
        f"{len(baseline_generations['calibration'].token_ids)} tokens "
        f"in {baseline_seconds['calibration']:.1f}s",
        flush=True,
    )

    print(
        f"Loading {args.embedding_model_id} on {args.embedding_device}...",
        flush=True,
    )
    sentence_model = load_sentence_embedder(
        args.embedding_model_id,
        args.embedding_device,
    )
    calibration_step_embeddings = embed_reasoning_sentences(
        sentence_model,
        calibration_steps,
    )
    test_step_embeddings = embed_reasoning_sentences(
        sentence_model,
        test_steps,
    )
    thresholds = {}
    threshold_rows = []
    for role, problem, steps, embeddings in (
        (
            "calibration",
            calibration_problem,
            calibration_steps,
            calibration_step_embeddings,
        ),
        ("heldout_test", test_problem, test_steps, test_step_embeddings),
    ):
        local_threshold, local_rows = mean_pairwise_step_similarity(embeddings)
        thresholds[role] = local_threshold
        values = [float(row["cosine_similarity"]) for row in local_rows]
        for row in local_rows:
            row.update(
                {
                    "role": role,
                    "problem_id": problem["id"],
                    "step_i_text": steps[int(row["step_i"]) - 1],
                    "step_j_text": steps[int(row["step_j"]) - 1],
                    "threshold_mean": local_threshold,
                    "pair_count": len(local_rows),
                    "pairwise_std": float(np.std(values)),
                    "pairwise_min": float(np.min(values)),
                    "pairwise_max": float(np.max(values)),
                }
            )
            threshold_rows.append(row)
    calibration_threshold = thresholds["calibration"]
    test_threshold = thresholds["heldout_test"]
    write_csv(
        paths["tables"] / "semantic_subgoal_threshold.csv",
        threshold_rows,
    )
    print(
        "Frozen per-problem thresholds: "
        f"calibration={calibration_threshold:.4f}, "
        f"heldout={test_threshold:.4f}",
        flush=True,
    )

    calibration_generation = baseline_generations["calibration"]
    (
        calibration_alignment,
        calibration_progress,
        calibration_summary,
    ) = semantic_alignment(
        sentence_model,
        backend.tokenizer,
        calibration_generation.text,
        calibration_generation.token_ids,
        calibration_step_embeddings,
        calibration_threshold,
    )
    calibration_correct, calibration_method, calibration_answer = answer_match(
        calibration_generation.text,
        str(calibration_problem["answer"]),
    )
    if (
        not calibration_correct
        or calibration_summary["ordered_subgoals_completed"]
        < MIN_BASELINE_SUBGOALS
        or re.search(
            r"</?think>",
            calibration_generation.text,
            flags=re.IGNORECASE,
        )
        or len(calibration_generation.token_ids)
        >= args.generation_safety_ceiling
    ):
        diagnostic = {
            "study_status": "INVALID_BASELINE",
            "problem_id": calibration_problem["id"],
            "correct_final": int(calibration_correct),
            "answer_match_method": calibration_method,
            "extracted_final_answer": calibration_answer,
            "output_tokens": len(calibration_generation.token_ids),
            "ordered_subgoal_coverage": calibration_summary[
                "ordered_subgoal_coverage"
            ],
            "required_minimum_subgoals": MIN_BASELINE_SUBGOALS,
            "thinking_tag_detected": int(
                bool(
                    re.search(
                        r"</?think>",
                        calibration_generation.text,
                        flags=re.IGNORECASE,
                    )
                )
            ),
            "generated_text": calibration_generation.text,
        }
        write_csv(
            paths["tables"] / "semantic_baseline_diagnostic.csv",
            [diagnostic],
        )
        raise RuntimeError(
            "Pilot is INVALID: the fixed calibration baseline was incorrect, "
            "hit the emergency ceiling, emitted thinking tags, or completed "
            f"fewer than {MIN_BASELINE_SUBGOALS} authored subgoals. Inspect "
            "semantic_baseline_diagnostic.csv; do not silently replace the prompt."
        )

    layer_rows = []
    fits = {}
    for layer in candidate_layers:
        fit = fit_progress_signal(
            calibration_generation.activations_by_layer[layer],
            calibration_progress,
            ridge=args.ridge,
            fold_count=args.blocked_cv_folds,
        )
        fits[layer] = fit
        layer_rows.append(
            {
                "layer": layer,
                "in_sample_incremental_r2": fit.in_sample_incremental_r2,
                "blocked_cv_incremental_r2": fit.blocked_cv_incremental_r2,
                "semantic_direction_norm": float(
                    np.linalg.norm(fit.semantic_conditional)
                ),
                "native_token_step_norm": fit.native_token_step_norm,
                "semantic_linear_cosine": cosine_similarity(
                    fit.semantic_conditional,
                    fit.linear,
                ),
                "semantic_k1_cosine_cosine": cosine_similarity(
                    fit.semantic_conditional,
                    fit.k1_cosine,
                ),
                "semantic_k1_sine_cosine": cosine_similarity(
                    fit.semantic_conditional,
                    fit.k1_sine,
                ),
                "selected": 0,
            }
        )
    selected_row = max(
        layer_rows,
        key=lambda row: (
            np.nan_to_num(row["blocked_cv_incremental_r2"], nan=-np.inf),
            row["in_sample_incremental_r2"],
        ),
    )
    selected_row["selected"] = 1
    selected_layer = int(selected_row["layer"])
    selected_fit = fits[selected_layer]
    write_csv(
        paths["tables"] / "semantic_signal_layer_selection.csv",
        layer_rows,
    )
    print(
        f"Selected layer {selected_layer}: calibration blocked-CV incremental "
        f"R2={selected_fit.blocked_cv_incremental_r2:.4f}",
        flush=True,
    )

    started = time.perf_counter()
    baseline_generations["heldout_test"] = collect_generation(
        backend,
        test_prompt,
        candidate_layers,
        args,
        seed + 1000,
        capture_activations=True,
    )
    baseline_seconds["heldout_test"] = time.perf_counter() - started
    test_generation = baseline_generations["heldout_test"]
    print(
        f"heldout_test: {len(test_generation.token_ids)} tokens "
        f"in {baseline_seconds['heldout_test']:.1f}s",
        flush=True,
    )
    test_alignment, test_progress, test_summary = semantic_alignment(
        sentence_model,
        backend.tokenizer,
        test_generation.text,
        test_generation.token_ids,
        test_step_embeddings,
        test_threshold,
    )
    test_correct, test_method, test_answer = answer_match(
        test_generation.text,
        str(test_problem["answer"]),
    )
    if (
        not test_correct
        or test_summary["ordered_subgoals_completed"] < MIN_BASELINE_SUBGOALS
        or re.search(
            r"</?think>",
            test_generation.text,
            flags=re.IGNORECASE,
        )
        or len(test_generation.token_ids) >= args.generation_safety_ceiling
    ):
        diagnostic = {
            "study_status": "INVALID_BASELINE",
            "problem_id": test_problem["id"],
            "correct_final": int(test_correct),
            "answer_match_method": test_method,
            "extracted_final_answer": test_answer,
            "output_tokens": len(test_generation.token_ids),
            "ordered_subgoal_coverage": test_summary[
                "ordered_subgoal_coverage"
            ],
            "required_minimum_subgoals": MIN_BASELINE_SUBGOALS,
            "thinking_tag_detected": int(
                bool(
                    re.search(
                        r"</?think>",
                        test_generation.text,
                        flags=re.IGNORECASE,
                    )
                )
            ),
            "generated_text": test_generation.text,
        }
        write_csv(
            paths["tables"] / "semantic_baseline_diagnostic.csv",
            [diagnostic],
        )
        raise RuntimeError(
            "Pilot is INVALID: the fixed held-out baseline was incorrect, hit "
            "the emergency ceiling, or had no measurable gold-subgoal progress. "
            "It may also have emitted thinking tags or completed fewer than "
            f"{MIN_BASELINE_SUBGOALS} authored subgoals. Inspect "
            "semantic_baseline_diagnostic.csv; do not reinterpret this as "
            "evidence against the activation signal."
        )

    heldout_correlation = heldout_partial_correlation(
        test_generation.activations_by_layer[selected_layer],
        test_progress,
        selected_fit.semantic_conditional,
    )
    event_rows = []
    event_summaries = []
    for label, direction in (
        ("semantic_conditional", selected_fit.semantic_conditional),
        ("linear", selected_fit.linear),
    ):
        local_rows, local_summary = event_locked_tracking(
            test_generation.activations_by_layer[selected_layer],
            test_alignment,
            direction,
            args.event_window_tokens,
            label,
        )
        event_rows.extend(local_rows)
        event_summaries.append(local_summary)
    semantic_event_summary = next(
        row
        for row in event_summaries
        if row["direction"] == "semantic_conditional"
    )
    write_csv(
        paths["tables"] / "semantic_event_locked_tracking.csv",
        event_rows,
    )
    write_csv(
        paths["tables"] / "semantic_event_locked_summary.csv",
        event_summaries,
    )

    baseline_tokens = len(test_generation.token_ids)
    pulse_start = pulse_start_after_subgoal(
        test_alignment,
        args.pulse_after_subgoal,
        baseline_tokens,
    )
    pulse_position = pulse_start / max(baseline_tokens, 1)
    semantic_step = 1.0 / len(test_steps)
    target_norm = (
        float(args.transport_alpha)
        * float(selected_fit.native_token_step_norm)
    )
    if target_norm <= 1e-12:
        write_csv(
            paths["tables"] / "semantic_control_diagnostic.csv",
            [
                {
                    "study_status": "INVALID_COLLAPSED_DOSE",
                    "selected_layer": selected_layer,
                    "native_token_step_norm": (
                        selected_fit.native_token_step_norm
                    ),
                    "transport_alpha": args.transport_alpha,
                }
            ],
        )
        raise RuntimeError(
            "Pilot is INVALID: calibration produced a zero pulse norm. "
            "Inspect semantic_control_diagnostic.csv."
        )
    try:
        deltas = {
            control: control_delta(
                control,
                selected_fit,
                target_norm,
                pulse_position,
                semantic_step,
            )
            for control in CONTROLS
        }
    except ValueError as exc:
        write_csv(
            paths["tables"] / "semantic_control_diagnostic.csv",
            [
                {
                    "study_status": "INVALID_COLLAPSED_CONTROL",
                    "selected_layer": selected_layer,
                    "error": str(exc),
                }
            ],
        )
        raise RuntimeError(
            "Pilot is INVALID: a norm-matched control direction collapsed. "
            "Inspect semantic_control_diagnostic.csv."
        ) from exc
    baseline_outcome = outcome_row(
        "baseline",
        test_problem,
        test_generation,
        test_alignment,
        test_progress,
        test_summary,
        baseline_tokens,
        baseline_seconds["heldout_test"],
        0.0,
        [],
    )
    outcomes = [baseline_outcome]
    alignment_rows = []
    for role, control, problem, rows in (
        (
            "calibration",
            "baseline",
            calibration_problem,
            calibration_alignment,
        ),
        ("heldout_test", "baseline", test_problem, test_alignment),
    ):
        for row in rows:
            alignment_rows.append(
                {
                    "role": role,
                    "control": control,
                    "problem_id": problem["id"],
                    **row,
                }
            )

    for control in CONTROLS:
        callback, applied_steps = pulse_callback(
            backend.torch,
            deltas[control],
            pulse_start,
            args.pulse_tokens,
        )
        started = time.perf_counter()
        generated = collect_generation(
            backend,
            test_prompt,
            [selected_layer],
            args,
            seed + 1000,
            intervention=callback,
            capture_activations=False,
        )
        elapsed = time.perf_counter() - started
        thinking_tag_detected = bool(
            re.search(r"</?think>", generated.text, flags=re.IGNORECASE)
        )
        if len(applied_steps) != args.pulse_tokens or thinking_tag_detected:
            write_csv(
                paths["tables"] / "semantic_control_diagnostic.csv",
                [
                    {
                        "study_status": "INVALID_INTERVENTION_DELIVERY",
                        "control": control,
                        "expected_pulse_applications": args.pulse_tokens,
                        "observed_pulse_applications": len(applied_steps),
                        "thinking_tag_detected": int(thinking_tag_detected),
                        "generated_text": generated.text,
                    }
                ],
            )
            raise RuntimeError(
                f"Pilot is INVALID: {control} did not receive the complete "
                "pulse or emitted thinking tags. Inspect "
                "semantic_control_diagnostic.csv."
            )
        rows, progress, summary = semantic_alignment(
            sentence_model,
            backend.tokenizer,
            generated.text,
            generated.token_ids,
            test_step_embeddings,
            test_threshold,
        )
        for row in rows:
            alignment_rows.append(
                {
                    "role": "heldout_test",
                    "control": control,
                    "problem_id": test_problem["id"],
                    **row,
                }
            )
        outcomes.append(
            outcome_row(
                control,
                test_problem,
                generated,
                rows,
                progress,
                summary,
                baseline_tokens,
                elapsed,
                target_norm,
                applied_steps,
            )
        )
        print(
            f"{control}: {len(generated.token_ids)} tokens; "
            f"coverage={summary['ordered_subgoal_coverage']:.2f}; "
            f"correct={outcomes[-1]['correct_final']}; "
            f"pulse applications={len(applied_steps)}",
            flush=True,
        )

    add_baseline_deltas(outcomes)
    for row in outcomes:
        row.update(
            {
                "selected_layer": selected_layer,
                "calibration_problem_id": calibration_problem["id"],
                "heldout_problem_id": test_problem["id"],
                "calibration_similarity_threshold": calibration_threshold,
                "heldout_similarity_threshold": test_threshold,
                "threshold_source": (
                    "per-problem mean distinct-pair cosine among gold steps; "
                    "frozen across baseline and intervention conditions"
                ),
                "chat_template_used": int(
                    bool(getattr(backend.tokenizer, "chat_template", None))
                ),
                "thinking_mode_requested": False,
                "progress_timebase": "t/T_heldout_baseline",
                "pulse_after_subgoal": args.pulse_after_subgoal,
                "pulse_start_token": pulse_start,
                "pulse_tokens": args.pulse_tokens,
                "transport_alpha_native_step": args.transport_alpha,
            }
        )
    gates = falsification_gates(
        selected_fit,
        heldout_correlation,
        semantic_event_summary,
        outcomes,
    )
    by_control = {row["control"]: row for row in outcomes}
    semantic = by_control["semantic_forward"]
    key_results = [
        {
            "result": "study_scope",
            "value": "one fixed calibration + one fixed held-out problem",
            "interpretation": (
                "Falsification pilot only; eight problems remain reserved"
            ),
        },
        {
            "result": "calibration_similarity_threshold",
            "value": calibration_threshold,
            "interpretation": (
                "Calibration problem mean cosine among distinct gold steps"
            ),
        },
        {
            "result": "heldout_similarity_threshold",
            "value": test_threshold,
            "interpretation": (
                "Held-out problem mean cosine, frozen across its conditions"
            ),
        },
        {
            "result": "selected_layer",
            "value": selected_layer,
            "interpretation": "Selected using calibration trace only",
        },
        {
            "result": "calibration_blocked_cv_incremental_r2",
            "value": selected_fit.blocked_cv_incremental_r2,
            "interpretation": (
                "Encoding diagnostic beyond token position and closed k=1"
            ),
        },
        {
            "result": "heldout_partial_correlation",
            "value": heldout_correlation,
            "interpretation": (
                "Frozen semantic direction versus held-out gold progress, "
                "residualized over position and k=1"
            ),
        },
        {
            "result": "heldout_median_event_locked_delta",
            "value": semantic_event_summary[
                "median_advancing_score_delta"
            ],
            "interpretation": (
                "Frozen direction movement at held-out subgoal boundaries"
            ),
        },
        {
            "result": "semantic_forward_correct",
            "value": semantic["correct_final"],
            "interpretation": "Must remain 1 before efficiency is considered",
        },
        {
            "result": "semantic_forward_coverage_delta",
            "value": semantic["delta_ordered_subgoal_coverage"],
            "interpretation": "Must be non-negative",
        },
        {
            "result": "semantic_forward_percent_token_delta",
            "value": semantic["percent_delta_output_tokens"],
            "interpretation": "Negative is shorter, conditional on correctness",
        },
        {
            "result": "semantic_forward_progress_auc_delta",
            "value": semantic["delta_progress_auc_to_T_baseline"],
            "interpretation": (
                "Positive means earlier gold-subgoal progress on the baseline clock"
            ),
        },
    ]
    write_csv(
        paths["tables"] / "semantic_subgoal_alignment.csv",
        alignment_rows,
    )
    write_csv(
        paths["tables"] / "semantic_intervention_outcomes.csv",
        outcomes,
    )
    write_csv(
        paths["tables"] / "semantic_falsification_gates.csv",
        gates,
    )
    write_csv(
        paths["tables"] / "semantic_key_results.csv",
        key_results,
    )
    trace_path = paths["tables"] / "semantic_selected_layer_traces.npz"
    temporary = trace_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        calibration_activations=np.asarray(
            calibration_generation.activations_by_layer[selected_layer],
            dtype=np.float16,
        ),
        calibration_semantic_progress=calibration_progress,
        heldout_activations=np.asarray(
            test_generation.activations_by_layer[selected_layer],
            dtype=np.float16,
        ),
        heldout_semantic_progress=test_progress,
        semantic_direction=np.asarray(
            selected_fit.semantic_conditional,
            dtype=np.float32,
        ),
    )
    temporary.replace(trace_path)
    print(json.dumps(gates, indent=2), flush=True)
    print(
        f"Wrote the focused 06d pilot results to {paths['tables']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
