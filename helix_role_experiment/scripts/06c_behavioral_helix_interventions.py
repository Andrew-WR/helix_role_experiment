from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from _common import write_csv
from helix_role_experiment.benchmarks import load_math500_integer_problems
from helix_role_experiment.behavioral import (
    anchor_sentence_fraction,
    extract_final_answer,
    final_answer_is_correct,
    repeated_ngram_fraction,
    repeated_sentence_fraction,
)
from helix_role_experiment.config import (
    ensure_output_dirs,
    load_config,
    seed_everything,
)
from helix_role_experiment.models import huggingface_collector_from_config
from helix_role_experiment.semantic_progress import (
    add_baseline_deltas,
    analyze_condition_sentences,
    load_sentence_embedder,
)
from helix_role_experiment.trajectory_geometry import (
    centered_transfer_metrics,
    fit_trajectory_models,
    match_norm,
)


FINAL_LINE_STOP_REGEX = (
    r"(?ims)</think>.*?^\s*\**final(?:\s+answer)?\**\s*:\**\s*\S[^\n]*"
    r"(?:\n|<\|im_end\|>)"
)
CONTROL_TO_MODEL = {
    "helix_local_forward": "generalized_helix",
    "linear_local_forward": "linear",
    "linear_plus_closed_k1_local_forward": "linear_plus_closed_k1",
}


def evaluation_prompt(problem) -> str:
    return (
        f"{problem.prompt}\n"
        "Solve the problem carefully. You may use as much internal reasoning "
        "as needed. Check the result, then end with a separate line beginning "
        "`FINAL:` followed by only the answer."
    )


def wrong_answer(problem) -> str:
    return str(int(problem.metadata["numeric_answer"]) + 1)


def generated_metrics(
    backend,
    generated,
    expected_answer: str,
    generation_safety_ceiling: int,
) -> dict[str, Any]:
    answer, marker_index = extract_final_answer(generated.text)
    correct = final_answer_is_correct(generated.text, expected_answer)
    reasoning_tokens = (
        len(generated.token_ids)
        if marker_index is None
        else len(
            backend.tokenizer(
                generated.text[:marker_index],
                add_special_tokens=False,
            )["input_ids"]
        )
    )
    return {
        "correct_final": int(correct),
        "answer_emitted": int(answer is not None),
        "extracted_final_answer": answer,
        "tokens_to_correct_final": (
            reasoning_tokens
            if correct
            else int(generation_safety_ceiling) + 1
        ),
        "reasoning_tokens_before_final": reasoning_tokens,
        "output_tokens": len(generated.token_ids),
        "normalized_correct_final_position": (
            reasoning_tokens / max(len(generated.token_ids), 1)
            if correct
            else float("nan")
        ),
        "hit_safety_ceiling": int(
            len(generated.token_ids) >= int(generation_safety_ceiling)
            and answer is None
            and not generated.reached_eos
        ),
        "anchor_sentence_fraction": anchor_sentence_fraction(generated.text),
        "repeated_sentence_fraction": repeated_sentence_fraction(
            generated.text
        ),
        "repeated_4gram_fraction": repeated_ngram_fraction(
            list(generated.token_ids),
            4,
        ),
        "reached_eos": int(generated.reached_eos),
        "generated_text": generated.text,
    }


def require_viable_baseline(
    label: str,
    metrics: dict[str, Any],
    diagnostics_path: Path,
) -> None:
    if metrics["answer_emitted"] and metrics["correct_final"]:
        return
    write_csv(diagnostics_path, [{"trace": label, **metrics}])
    raise RuntimeError(
        f"{label} baseline must emit a correct FINAL answer before this "
        "two-prompt case study can continue. This fixed prompt is not silently "
        f"replaced because that would introduce selection bias. Diagnostics: "
        f"{diagnostics_path}"
    )


def local_transport_callback(
    torch,
    primary_curve,
    control_curve,
    baseline_tokens: int,
    progress_step: float,
    alpha: float,
):
    norms: list[float] = []

    def callback(_layer, step, hidden):
        progress = min(float(step) / max(int(baseline_tokens), 1), 1.0)
        primary = (
            primary_curve.local_delta(progress, progress_step)
            * float(alpha)
        )
        delta = (
            primary
            if control_curve is primary_curve
            else match_norm(
                control_curve.local_delta(progress, progress_step),
                primary,
            )
        )
        norms.append(float(np.linalg.norm(delta)))
        return hidden + torch.as_tensor(
            delta,
            device=hidden.device,
            dtype=hidden.dtype,
        )

    return callback, norms


def sequence_log_odds(
    backend,
    prompt: str,
    correct_answer: str,
    incorrect_answer: str,
    layer: int,
    intervention,
) -> float:
    scored = backend.score_continuations(
        prompt,
        [
            f"\nFINAL: {correct_answer}",
            f"\nFINAL: {incorrect_answer}",
        ],
        layer,
        intervention=intervention,
    )
    return float(
        scored[0]["total_log_probability"]
        - scored[1]["total_log_probability"]
    )


def add_length_deltas(
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = next(row for row in outcomes if row["control"] == "baseline")
    for row in outcomes:
        for metric in (
            "output_tokens",
            "reasoning_tokens_before_final",
            "tokens_to_correct_final",
        ):
            difference = float(row[metric]) - float(baseline[metric])
            denominator = float(baseline[metric])
            row[f"delta_{metric}"] = difference
            row[f"percent_delta_{metric}"] = (
                100.0 * difference / denominator
                if denominator
                else float("nan")
            )
    return outcomes


def summarize_key_results(
    outcomes: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = next(row for row in outcomes if row["control"] == "baseline")
    rows = []
    for outcome in outcomes:
        rows.append(
            {
                "result_type": "behavioral_condition",
                "control": outcome["control"],
                "correct_final": outcome["correct_final"],
                "output_tokens": outcome["output_tokens"],
                "delta_output_tokens": outcome["delta_output_tokens"],
                "percent_delta_output_tokens": outcome[
                    "percent_delta_output_tokens"
                ],
                "reasoning_tokens_before_final": outcome[
                    "reasoning_tokens_before_final"
                ],
                "percent_delta_reasoning_tokens_before_final": outcome[
                    "percent_delta_reasoning_tokens_before_final"
                ],
                "sequence_log_odds_correct": outcome[
                    "sequence_log_odds_correct"
                ],
                "mean_intervention_norm": outcome[
                    "mean_intervention_norm"
                ],
            }
        )
    helix = next(
        row for row in outcomes if row["control"] == "helix_local_forward"
    )
    comparators = [
        row
        for row in outcomes
        if row["control"]
        in {
            "linear_local_forward",
            "linear_plus_closed_k1_local_forward",
        }
    ]
    transfer = next(
        row
        for row in transfer_rows
        if row["model"] == "generalized_helix"
    )
    selected_fit = {
        row["model"]: row
        for row in selection_rows
        if row["selected_within_model"]
    }
    helix_fit = selected_fit["generalized_helix"]
    rows.append(
        {
            "result_type": "geometry",
            "control": "generalized_helix",
            "selected_turns": helix_fit["turns"],
            "selected_radius_slope": helix_fit["radius_slope"],
            "blocked_cv_r2": helix_fit["blocked_cv_r2"],
            "incremental_blocked_cv_r2_vs_linear": (
                float(helix_fit["blocked_cv_r2"])
                - float(selected_fit["linear"]["blocked_cv_r2"])
            ),
            "incremental_blocked_cv_r2_vs_closed_k1": (
                float(helix_fit["blocked_cv_r2"])
                - float(
                    selected_fit["linear_plus_closed_k1"]["blocked_cv_r2"]
                )
            ),
            "test_centered_transfer_r2": transfer[
                "centered_transfer_r2"
            ],
        }
    )
    gate_passed = bool(
        int(helix["correct_final"])
        and float(helix["output_tokens"]) < float(baseline["output_tokens"])
        and all(
            float(helix["output_tokens"]) < float(row["output_tokens"])
            for row in comparators
            if int(row["correct_final"])
        )
        and float(helix["sequence_log_odds_correct"])
        > float(baseline["sequence_log_odds_correct"])
    )
    rows.append(
        {
            "result_type": "primary_gate",
            "control": "helix_local_forward",
            "gate": "faster_correct_reasoning_than_baseline_and_controls",
            "status": (
                "descriptive_pass_single_test_prompt"
                if gate_passed
                else "descriptive_fail_single_test_prompt"
            ),
            "correct_final": helix["correct_final"],
            "percent_delta_output_tokens": helix[
                "percent_delta_output_tokens"
            ],
            "sequence_log_odds_improvement": (
                float(helix["sequence_log_odds_correct"])
                - float(baseline["sequence_log_odds_correct"])
            ),
            "test_centered_transfer_r2": transfer[
                "centered_transfer_r2"
            ],
        }
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a generalized helix to one natural MATH-500 reasoning "
            "trajectory and test local forward transport on a second problem."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--math500-path",
        default=None,
        help="Optional local MATH-500 test.jsonl.",
    )
    parser.add_argument("--math500-level", type=int, default=1)
    parser.add_argument("--generation-safety-ceiling", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--progress-step", type=float, default=0.05)
    parser.add_argument("--transport-alpha", type=float, default=1.0)
    parser.add_argument("--blocked-cv-folds", type=int, default=5)
    parser.add_argument(
        "--embedding-model-id",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument(
        "--skip-semantic-analysis",
        action="store_true",
        help="Write causal results without loading the sentence embedder.",
    )
    args = parser.parse_args()
    if args.generation_safety_ceiling <= 0:
        raise ValueError("--generation-safety-ceiling must be positive")
    if not 0.0 < args.progress_step <= 1.0:
        raise ValueError("--progress-step must be in (0, 1]")
    if args.transport_alpha <= 0:
        raise ValueError("--transport-alpha must be positive")
    if args.blocked_cv_folds < 2:
        raise ValueError("--blocked-cv-folds must be at least 2")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    seed = int(config["study"]["seed"]) + 663
    seed_everything(seed)
    math500_path = (
        Path(args.math500_path)
        if args.math500_path
        else paths["root"] / "math500_test.jsonl"
    )
    calibration_problem, test_problem = load_math500_integer_problems(
        2,
        {int(args.math500_level)},
        seed,
        math500_path,
    )
    design_rows = [
        {
            "role": "calibration",
            "problem_id": calibration_problem.problem_id,
            "level": calibration_problem.metadata["level"],
            "subject": calibration_problem.metadata["subject"],
            "prompt": calibration_problem.prompt,
            "expected_answer": calibration_problem.answer,
        },
        {
            "role": "test",
            "problem_id": test_problem.problem_id,
            "level": test_problem.metadata["level"],
            "subject": test_problem.metadata["subject"],
            "prompt": test_problem.prompt,
            "expected_answer": test_problem.answer,
        },
    ]
    write_csv(
        paths["tables"] / "behavioral_helix_two_prompt_design.csv",
        design_rows,
    )
    print(
        "Standalone 06c uses two fixed, disjoint MATH-500 prompts: "
        f"calibration={calibration_problem.problem_id}; "
        f"test={test_problem.problem_id}",
        flush=True,
    )
    print("Loading Qwen for natural reasoning traces...", flush=True)
    backend = huggingface_collector_from_config(
        config["model"],
        config["collection"],
    )
    layer = int(args.layer)
    if layer < 0 or layer >= len(backend.layers):
        raise ValueError(
            f"layer {layer} is invalid; model has {len(backend.layers)} layers"
        )

    calibration_prompt = evaluation_prompt(calibration_problem)
    started = time.perf_counter()
    calibration_generation = backend.collect(
        calibration_prompt,
        [layer],
        args.generation_safety_ceiling,
        seed,
        temperature=args.temperature,
        disable_eos=False,
        intervention=None,
        capture_activations=True,
        capture_eos_logits=False,
        stop_regex=FINAL_LINE_STOP_REGEX,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_check_interval=8,
    )
    calibration_metrics = generated_metrics(
        backend,
        calibration_generation,
        calibration_problem.answer,
        args.generation_safety_ceiling,
    )
    require_viable_baseline(
        "calibration",
        calibration_metrics,
        paths["tables"] / "behavioral_helix_calibration_diagnostic.csv",
    )
    print(
        f"Calibration trace: {len(calibration_generation.token_ids)} tokens "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_calibration_trace.csv",
        [
            {
                "layer": layer,
                "problem_id": calibration_problem.problem_id,
                "seed": seed,
                **calibration_metrics,
            }
        ],
    )
    curves, selection_rows = fit_trajectory_models(
        calibration_generation.activations_by_layer[layer],
        ridge=float(config["analysis"].get("ridge", 0.001)),
        fold_count=args.blocked_cv_folds,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_trajectory_model_fit.csv",
        selection_rows,
    )

    test_prompt = evaluation_prompt(test_problem)
    test_seed = seed + 1000
    started = time.perf_counter()
    baseline_generation = backend.collect(
        test_prompt,
        [layer],
        args.generation_safety_ceiling,
        test_seed,
        temperature=args.temperature,
        disable_eos=False,
        intervention=None,
        capture_activations=True,
        capture_eos_logits=False,
        stop_regex=FINAL_LINE_STOP_REGEX,
        top_p=args.top_p,
        top_k=args.top_k,
        stop_check_interval=8,
    )
    baseline_metrics = generated_metrics(
        backend,
        baseline_generation,
        test_problem.answer,
        args.generation_safety_ceiling,
    )
    require_viable_baseline(
        "test",
        baseline_metrics,
        paths["tables"] / "behavioral_helix_test_diagnostic.csv",
    )
    baseline_tokens = len(baseline_generation.token_ids)
    print(
        f"Test baseline: {baseline_tokens} tokens in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    transfer_rows = []
    for model, curve in curves.items():
        transfer_rows.append(
            {
                "calibration_problem_id": calibration_problem.problem_id,
                "test_problem_id": test_problem.problem_id,
                "layer": layer,
                "model": model,
                **centered_transfer_metrics(
                    curve,
                    baseline_generation.activations_by_layer[layer],
                ),
            }
        )
    write_csv(
        paths["tables"] / "behavioral_helix_test_transfer.csv",
        transfer_rows,
    )
    trace_path = (
        paths["tables"] / f"behavioral_helix_natural_traces_layer_{layer}.npz"
    )
    temporary_trace_path = trace_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_trace_path,
        calibration_activations=np.asarray(
            calibration_generation.activations_by_layer[layer],
            dtype=np.float16,
        ),
        calibration_token_ids=np.asarray(
            calibration_generation.token_ids,
            dtype=np.int64,
        ),
        calibration_progress=np.linspace(
            0.0,
            1.0,
            len(calibration_generation.token_ids),
            dtype=np.float64,
        ),
        test_baseline_activations=np.asarray(
            baseline_generation.activations_by_layer[layer],
            dtype=np.float16,
        ),
        test_baseline_token_ids=np.asarray(
            baseline_generation.token_ids,
            dtype=np.int64,
        ),
        test_baseline_progress=np.arange(
            len(baseline_generation.token_ids),
            dtype=np.float64,
        )
        / max(len(baseline_generation.token_ids), 1),
    )
    temporary_trace_path.replace(trace_path)

    primary_curve = curves["generalized_helix"]
    torch = backend.torch
    sequence_scores = {
        "baseline": sequence_log_odds(
            backend,
            test_prompt,
            test_problem.answer,
            wrong_answer(test_problem),
            layer,
            None,
        )
    }
    outcomes = [
        {
            "layer": layer,
            "calibration_problem_id": calibration_problem.problem_id,
            "problem_id": test_problem.problem_id,
            "benchmark": "math500",
            "benchmark_level": test_problem.metadata["level"],
            "seed": test_seed,
            "control": "baseline",
            "reference_baseline_tokens": baseline_tokens,
            "progress_definition": "min(t/T_baseline,1)",
            "progress_step": args.progress_step,
            "transport_alpha": args.transport_alpha,
            "mean_intervention_norm": 0.0,
            "max_intervention_norm": 0.0,
            "sequence_log_odds_correct": sequence_scores["baseline"],
            "chat_template_used": int(
                bool(getattr(backend.tokenizer, "chat_template", None))
            ),
            "thinking_mode_requested": backend.chat_template_kwargs.get(
                "enable_thinking"
            ),
            **baseline_metrics,
        }
    ]
    for control, model in CONTROL_TO_MODEL.items():
        score_callback, _ = local_transport_callback(
            torch,
            primary_curve,
            curves[model],
            baseline_tokens,
            args.progress_step,
            args.transport_alpha,
        )
        sequence_scores[control] = sequence_log_odds(
            backend,
            test_prompt,
            test_problem.answer,
            wrong_answer(test_problem),
            layer,
            score_callback,
        )
        callback, intervention_norms = local_transport_callback(
            torch,
            primary_curve,
            curves[model],
            baseline_tokens,
            args.progress_step,
            args.transport_alpha,
        )
        started = time.perf_counter()
        generated = backend.collect(
            test_prompt,
            [layer],
            args.generation_safety_ceiling,
            test_seed,
            temperature=args.temperature,
            disable_eos=False,
            intervention=callback,
            capture_activations=False,
            capture_eos_logits=False,
            stop_regex=FINAL_LINE_STOP_REGEX,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_check_interval=8,
        )
        metrics = generated_metrics(
            backend,
            generated,
            test_problem.answer,
            args.generation_safety_ceiling,
        )
        outcomes.append(
            {
                "layer": layer,
                "calibration_problem_id": calibration_problem.problem_id,
                "problem_id": test_problem.problem_id,
                "benchmark": "math500",
                "benchmark_level": test_problem.metadata["level"],
                "seed": test_seed,
                "control": control,
                "reference_baseline_tokens": baseline_tokens,
                "progress_definition": "min(t/T_baseline,1)",
                "progress_step": args.progress_step,
                "transport_alpha": args.transport_alpha,
                "mean_intervention_norm": float(
                    np.mean(intervention_norms)
                ),
                "max_intervention_norm": float(
                    np.max(intervention_norms)
                ),
                "sequence_log_odds_correct": sequence_scores[control],
                "chat_template_used": int(
                    bool(getattr(backend.tokenizer, "chat_template", None))
                ),
                "thinking_mode_requested": backend.chat_template_kwargs.get(
                    "enable_thinking"
                ),
                **metrics,
            }
        )
        print(
            f"{control}: {len(generated.token_ids)} tokens, "
            f"correct={metrics['correct_final']}, "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    outcomes = add_length_deltas(outcomes)
    key_results = summarize_key_results(
        outcomes,
        transfer_rows,
        selection_rows,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_outcomes.csv",
        outcomes,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_key_results.csv",
        key_results,
    )

    if not args.skip_semantic_analysis:
        generation_tokenizer = backend.tokenizer
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"Loading {args.embedding_model_id} on "
            f"{args.embedding_device} for semantic outcomes...",
            flush=True,
        )
        sentence_model = load_sentence_embedder(
            args.embedding_model_id,
            args.embedding_device,
        )
        sentence_rows = []
        semantic_summaries = []
        for outcome in outcomes:
            rows, summary = analyze_condition_sentences(
                sentence_model,
                generation_tokenizer,
                outcome["control"],
                outcome["generated_text"],
                int(outcome["output_tokens"]),
            )
            sentence_rows.extend(rows)
            semantic_summaries.append(summary)
        semantic_summaries = add_baseline_deltas(semantic_summaries)
        for summary in semantic_summaries:
            key_results.append(
                {
                    "result_type": "semantic_condition",
                    "control": summary["control"],
                    "redundant_sentence_count_0.65": summary[
                        "redundant_sentence_count_0.65"
                    ],
                    "redundant_sentence_rate_0.65": summary[
                        "redundant_sentence_rate_0.65"
                    ],
                    "delta_redundant_sentence_count_0.65": summary[
                        "delta_redundant_sentence_count_0.65"
                    ],
                    "delta_redundant_sentence_rate_0.65": summary[
                        "delta_redundant_sentence_rate_0.65"
                    ],
                    "backward_stage_transitions": summary[
                        "backward_stage_transitions"
                    ],
                    "stage_revisits": summary["stage_revisits"],
                }
            )
        write_csv(
            paths["tables"] / "behavioral_helix_sentence_audit.csv",
            sentence_rows,
        )
        write_csv(
            paths["tables"] / "behavioral_helix_semantic_summary.csv",
            semantic_summaries,
        )
        write_csv(
            paths["tables"] / "behavioral_helix_key_results.csv",
            key_results,
        )

    print(
        json.dumps(
            [
                row
                for row in key_results
                if row["result_type"] == "primary_gate"
            ],
            indent=2,
        ),
        flush=True,
    )
    print(f"Wrote file 06c results to {paths['tables']}", flush=True)


if __name__ == "__main__":
    main()
