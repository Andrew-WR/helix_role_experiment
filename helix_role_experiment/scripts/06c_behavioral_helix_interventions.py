from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

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
    config_hash,
    ensure_output_dirs,
    load_config,
    seed_everything,
)
from helix_role_experiment.controlled_tasks import (
    ControlledProblem,
    generate_iterative_problem,
)
from helix_role_experiment.counterfactuals import build_progress_position_cross
from helix_role_experiment.models import huggingface_collector_from_config


EPS = 1e-12
FINAL_LINE_STOP_REGEX = (
    r"(?im)^\s*\**final(?:\s+answer)?\**\s*:\**\s*\S[^\n]*\n"
)


def load_geometry_module():
    """Reuse the audited generalized-helix estimator from file 06b."""

    path = Path(__file__).with_name("06b_falsify_generalized_helix.py")
    spec = importlib.util.spec_from_file_location("_helix_geometry_06b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generalized-helix code from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wrong_answer(problem: ControlledProblem) -> str:
    if problem.metadata.get("benchmark") == "math500":
        return str(int(problem.metadata["numeric_answer"]) + 1)
    if problem.family == "iterative_state_machine":
        modulus = int(problem.metadata["modulus"])
        return str((int(problem.answer) + 1) % modulus)
    if problem.family == "fictional_ontology":
        return next(
            value
            for value in problem.metadata["entities"]
            if value != problem.answer
        )
    if problem.family == "dependency_graph":
        return "BLOCKED"
    raise ValueError(f"unknown controlled family {problem.family!r}")


def evaluation_prompt(
    problem: ControlledProblem,
    mid_prefix: str,
    scenario: str,
    incorrect_answer: str,
) -> str:
    suffix = (
        "\nSolve efficiently without restating the problem. Use at most 12 "
        "short reasoning lines, check the calculation once, and then end with "
        "a separate line beginning `FINAL:` followed by only your answer."
    )
    if problem.metadata.get("benchmark") == "math500":
        if scenario == "accelerate":
            return problem.prompt + suffix
        if scenario == "repair_wrong_commitment":
            return (
                f"{problem.prompt}\nA previous solver proposed "
                f"`{incorrect_answer}`. Check that candidate and correct it "
                "if needed."
                + suffix
            )
    if scenario == "accelerate":
        return mid_prefix + suffix
    if scenario == "repair_wrong_commitment":
        return (
            f"{problem.prompt}\nA previous solver confidently asserted "
            f"`{incorrect_answer}` without deriving it. Re-evaluate that "
            "commitment and repair it if necessary."
            + suffix
        )
    raise ValueError(f"unknown scenario {scenario!r}")


def pulse_callback(torch, delta: np.ndarray, pulse_tokens: int):
    def callback(_layer, step, hidden):
        if int(step) >= int(pulse_tokens):
            return hidden
        return hidden + torch.as_tensor(
            delta,
            device=hidden.device,
            dtype=hidden.dtype,
        )

    return callback


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


def answer_logit_direction(
    backend,
    correct_answer: str,
    incorrect_answer: str,
    target_norm: float,
) -> np.ndarray:
    correct_ids = backend.tokenizer(
        f"\nFINAL: {correct_answer}",
        add_special_tokens=False,
    )["input_ids"]
    incorrect_ids = backend.tokenizer(
        f"\nFINAL: {incorrect_answer}",
        add_special_tokens=False,
    )["input_ids"]
    shared = 0
    for left, right in zip(correct_ids, incorrect_ids):
        if left != right:
            break
        shared += 1
    correct_answer_ids = correct_ids[shared:] or correct_ids[-1:]
    incorrect_answer_ids = incorrect_ids[shared:] or incorrect_ids[-1:]
    weights = backend.model.get_output_embeddings().weight.detach()
    direction = (
        weights[correct_answer_ids].float().mean(dim=0)
        - weights[incorrect_answer_ids].float().mean(dim=0)
    ).cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise ValueError("answer-logit positive-control direction collapsed")
    return direction * (float(target_norm) / norm)


def generated_metrics(
    backend,
    generated,
    expected_answer: str,
    generation_safety_ceiling: int,
) -> dict:
    answer, marker_index = extract_final_answer(generated.text)
    correct = final_answer_is_correct(generated.text, expected_answer)
    if marker_index is None:
        reasoning_tokens = len(generated.token_ids)
    else:
        reasoning_tokens = len(
            backend.tokenizer(
                generated.text[:marker_index],
                add_special_tokens=False,
            )["input_ids"]
        )
    return {
        "correct_final": int(correct),
        "answer_emitted": int(answer is not None),
        "tokens_to_correct_final": (
            reasoning_tokens
            if correct
            else int(generation_safety_ceiling) + 1
        ),
        "reasoning_tokens_before_final": reasoning_tokens,
        "output_tokens": len(generated.token_ids),
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


def fit_internal_calibration_geometry(
    backend,
    geometry_code,
    config: dict,
    paths: dict[str, Path],
    layer: int,
    problem_count: int,
    rebuild: bool,
):
    """Collect the minimal concise trajectories needed by file 06c itself."""

    seed = int(config["study"]["seed"]) + 660
    cache_key = config_hash(
        {
            "model": config["model"],
            "layer": layer,
            "problem_count": problem_count,
            "seed": seed,
            "calibration": "iterative_concise_v1",
        }
    )
    cache_path = (
        paths["tables"]
        / f"behavioral_helix_internal_calibration_layer_{layer}.npz"
    )
    calibration_rng = np.random.default_rng(seed)
    problems = [
        generate_iterative_problem(index, calibration_rng)
        for index in range(problem_count)
    ]
    concise_variants = [
        variant
        for problem in problems
        for variant in build_progress_position_cross(problem)
        if variant.condition == "concise"
    ]
    rows = None
    activations = None
    token_counts = None
    cache_complete = False
    if cache_path.is_file() and not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["cache_key"].item()) == cache_key:
                    rows = json.loads(str(data["rows_json"].item()))
                    activations = np.asarray(
                        data["activations"],
                        dtype=np.float32,
                    )
                    token_counts = np.asarray(
                        data["token_counts"],
                        dtype=np.float64,
                    )
                    cache_complete = bool(data["complete"].item())
                    if not (
                        len(rows)
                        == len(activations)
                        == len(token_counts)
                        <= len(concise_variants)
                    ):
                        raise ValueError("calibration cache is misaligned")
                    if (
                        cache_complete
                        and len(rows) != len(concise_variants)
                    ):
                        raise ValueError(
                            "complete calibration cache is truncated"
                        )
                    print(
                        (
                            "Reused"
                            if cache_complete
                            else "Resuming"
                        )
                        + f" {len(rows)}/{len(concise_variants)} internal "
                        f"calibration states from {cache_path}",
                        flush=True,
                    )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            print(
                f"Ignoring incomplete calibration cache {cache_path}",
                flush=True,
            )

    if rows is None or not cache_complete:
        if rows is None:
            rows = []
            activation_values = []
            token_values = []
        else:
            activation_values = list(activations)
            token_values = list(token_counts)
        temporary_cache = cache_path.with_suffix(".tmp.npz")
        for index in range(len(rows), len(concise_variants)):
            variant = concise_variants[index]
            row = asdict(variant)
            formatted = backend.format_prompt(row["text"])
            token_count = len(
                backend.tokenizer(
                    formatted,
                    add_special_tokens=True,
                )["input_ids"]
            )
            generated = backend.collect(
                row["text"],
                [layer],
                1,
                seed + index,
                temperature=0.0,
                disable_eos=True,
                capture_activations=True,
                capture_eos_logits=True,
            )
            row["sentence_count_proxy"] = row["text"].count(".")
            row["termination_allowed"] = int(
                row["termination_allowed"]
            )
            row["eos_logit"] = generated.eos_logits[-1]
            for column in geometry_code.OPERATION_COLUMNS:
                operation = column.removeprefix("operation_")
                row[column] = int(row["operation"] == operation)
            rows.append(row)
            activation_values.append(
                generated.activations_by_layer[layer][-1]
            )
            token_values.append(token_count)
            activations = np.asarray(
                activation_values,
                dtype=np.float32,
            )
            token_counts = np.asarray(token_values, dtype=np.float64)
            np.savez_compressed(
                temporary_cache,
                cache_key=np.asarray(cache_key),
                complete=np.asarray(
                    len(rows) == len(concise_variants)
                ),
                rows_json=np.asarray(json.dumps(rows)),
                activations=activations.astype(np.float16),
                token_counts=token_counts,
            )
            temporary_cache.replace(cache_path)
            print(
                f"Internal calibration {index + 1}/"
                f"{len(concise_variants)}",
                flush=True,
            )
        print(
            f"Cached internal calibration at {cache_path}",
            flush=True,
        )

    centered = geometry_code.center_within_problem_trajectory(
        activations,
        rows,
    )
    ridge = float(config["analysis"].get("ridge", 0.001))
    geometry, selection = geometry_code.fit_geometry(
        rows,
        centered,
        token_counts,
        ridge,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_calibration_selection.csv",
        selection,
    )
    return geometry, rows


def summarize(rows: list[dict], assay_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["control"])].append(row)
    output = []
    for (scenario, control), values in sorted(grouped.items()):
        output.append(
            {
                "result_type": "behavioral_condition",
                "scenario": scenario,
                "control": control,
                "sample_count": len(values),
                "problem_count": len(
                    {row["problem_id"] for row in values}
                ),
                "correct_final_rate": float(
                    np.mean([row["correct_final"] for row in values])
                ),
                "answer_emission_rate": float(
                    np.mean([row["answer_emitted"] for row in values])
                ),
                "mean_tokens_to_correct_final_censored": float(
                    np.mean(
                        [row["tokens_to_correct_final"] for row in values]
                    )
                ),
                "mean_reasoning_tokens_before_final": float(
                    np.mean(
                        [
                            row["reasoning_tokens_before_final"]
                            for row in values
                        ]
                    )
                ),
                "mean_output_tokens": float(
                    np.mean([row["output_tokens"] for row in values])
                ),
                "mean_anchor_sentence_fraction": float(
                    np.mean(
                        [row["anchor_sentence_fraction"] for row in values]
                    )
                ),
                "mean_repeated_sentence_fraction": float(
                    np.mean(
                        [row["repeated_sentence_fraction"] for row in values]
                    )
                ),
                "mean_repeated_4gram_fraction": float(
                    np.mean(
                        [row["repeated_4gram_fraction"] for row in values]
                    )
                ),
                "mean_sequence_log_odds_correct": float(
                    np.mean(finite_sequence_scores)
                    if (
                        finite_sequence_scores := [
                            row["sequence_log_odds_correct"]
                            for row in values
                            if np.isfinite(
                                row["sequence_log_odds_correct"]
                            )
                        ]
                    )
                    else float("nan")
                ),
            }
        )

    assay_differences = np.asarray(
        [
            row["positive_control_log_odds"]
            - row["baseline_log_odds"]
            for row in assay_rows
        ],
        dtype=np.float64,
    )
    sequence_assay_valid = bool(
        len(assay_differences)
        and np.median(assay_differences) > 0.5
        and np.mean(assay_differences > 0) >= 2.0 / 3.0
    )
    output.append(
        {
            "result_type": "primary_gate",
            "scenario": "all",
            "control": "answer_logit_positive_control",
            "gate": "sequence_assay_has_dynamic_range",
            "median_sequence_log_odds_improvement": (
                float(np.median(assay_differences))
                if len(assay_differences)
                else float("nan")
            ),
            "fraction_positive": (
                float(np.mean(assay_differences > 0))
                if len(assay_differences)
                else float("nan")
            ),
            "status": (
                "assay_valid"
                if sequence_assay_valid
                else "assay_invalid"
            ),
        }
    )

    summary_lookup = {
        (row["scenario"], row["control"]): row
        for row in output
        if row["result_type"] == "behavioral_condition"
    }

    def condition(scenario: str, control: str) -> dict:
        return summary_lookup[(scenario, control)]

    baseline_generations = [
        row for row in rows if row["control"] == "baseline"
    ]
    baseline_completion_rate = float(
        np.mean([row["answer_emitted"] for row in baseline_generations])
    )
    baseline_cap_rate = float(
        np.mean(
            [row["hit_safety_ceiling"] for row in baseline_generations]
        )
    )
    generation_assay_valid = bool(
        baseline_generations
        and baseline_completion_rate == 1.0
        and baseline_cap_rate == 0.0
    )
    output.append(
        {
            "result_type": "primary_gate",
            "scenario": "all",
            "control": "baseline",
            "gate": "generation_assay_reaches_final_answer",
            "baseline_answer_emission_rate": baseline_completion_rate,
            "baseline_safety_ceiling_rate": baseline_cap_rate,
            "status": (
                "assay_valid"
                if generation_assay_valid
                else "assay_invalid"
            ),
        }
    )
    assay_valid = sequence_assay_valid and generation_assay_valid

    acceleration = condition("accelerate", "helix_desired")
    acceleration_comparators = [
        condition("accelerate", value)
        for value in (
            "baseline",
            "helix_opposite",
            "linear_desired",
            "linear_plus_closed_k1_desired",
            "norm_matched_random",
        )
        if ("accelerate", value) in summary_lookup
    ]
    acceleration_survives = bool(
        assay_valid
        and acceleration["correct_final_rate"]
        >= max(row["correct_final_rate"] for row in acceleration_comparators)
        and acceleration["mean_tokens_to_correct_final_censored"]
        < min(
            row["mean_tokens_to_correct_final_censored"]
            for row in acceleration_comparators
        )
        and acceleration["mean_sequence_log_odds_correct"]
        > condition(
            "accelerate",
            "baseline",
        )["mean_sequence_log_odds_correct"]
    )
    output.append(
        {
            "result_type": "primary_gate",
            "scenario": "accelerate",
            "control": "helix_desired",
            "gate": "faster_correct_final_answer",
            "correct_final_rate": acceleration["correct_final_rate"],
            "best_comparator_correct_rate": max(
                row["correct_final_rate"]
                for row in acceleration_comparators
            ),
            "mean_tokens_to_correct_final_censored": acceleration[
                "mean_tokens_to_correct_final_censored"
            ],
            "best_comparator_tokens_to_correct": min(
                row["mean_tokens_to_correct_final_censored"]
                for row in acceleration_comparators
            ),
            "mean_sequence_log_odds_correct": acceleration[
                "mean_sequence_log_odds_correct"
            ],
            "baseline_sequence_log_odds": condition(
                "accelerate",
                "baseline",
            )["mean_sequence_log_odds_correct"],
            "status": (
                "survived_falsification"
                if acceleration_survives
                else (
                    "failed_falsification"
                    if assay_valid
                    else "inconclusive_assay_invalid"
                )
            ),
        }
    )

    repair = condition("repair_wrong_commitment", "helix_desired")
    repair_comparators = [
        condition("repair_wrong_commitment", value)
        for value in (
            "baseline",
            "helix_opposite",
            "linear_desired",
            "linear_plus_closed_k1_desired",
            "norm_matched_random",
        )
        if ("repair_wrong_commitment", value) in summary_lookup
    ]
    comparator_reasoning_tokens = float(
        np.mean(
            [
                row["mean_reasoning_tokens_before_final"]
                for row in repair_comparators
            ]
        )
    )
    comparator_anchor_fraction = float(
        np.mean(
            [
                row["mean_anchor_sentence_fraction"]
                for row in repair_comparators
            ]
        )
    )
    comparator_sentence_repetition = float(
        np.mean(
            [
                row["mean_repeated_sentence_fraction"]
                for row in repair_comparators
            ]
        )
    )
    comparator_ngram_repetition = float(
        np.mean(
            [
                row["mean_repeated_4gram_fraction"]
                for row in repair_comparators
            ]
        )
    )
    induced_reflection = bool(
        repair["mean_anchor_sentence_fraction"]
        > comparator_anchor_fraction
        or repair["mean_repeated_sentence_fraction"]
        > comparator_sentence_repetition
        or repair["mean_repeated_4gram_fraction"]
        > comparator_ngram_repetition
    )
    repair_survives = bool(
        assay_valid
        and repair["correct_final_rate"]
        > condition(
            "repair_wrong_commitment",
            "baseline",
        )["correct_final_rate"]
        and repair["correct_final_rate"]
        >= max(row["correct_final_rate"] for row in repair_comparators)
        and repair["mean_reasoning_tokens_before_final"]
        > comparator_reasoning_tokens
        and induced_reflection
    )
    output.append(
        {
            "result_type": "primary_gate",
            "scenario": "repair_wrong_commitment",
            "control": "helix_desired",
            "gate": "slow_reflection_improves_reasoning",
            "correct_final_rate": repair["correct_final_rate"],
            "baseline_correct_rate": condition(
                "repair_wrong_commitment",
                "baseline",
            )["correct_final_rate"],
            "mean_reasoning_tokens_before_final": repair[
                "mean_reasoning_tokens_before_final"
            ],
            "comparator_mean_reasoning_tokens": comparator_reasoning_tokens,
            "mean_anchor_sentence_fraction": repair[
                "mean_anchor_sentence_fraction"
            ],
            "comparator_mean_anchor_fraction": comparator_anchor_fraction,
            "mean_repeated_sentence_fraction": repair[
                "mean_repeated_sentence_fraction"
            ],
            "comparator_mean_repeated_sentence_fraction": (
                comparator_sentence_repetition
            ),
            "mean_repeated_4gram_fraction": repair[
                "mean_repeated_4gram_fraction"
            ],
            "comparator_mean_repeated_4gram_fraction": (
                comparator_ngram_repetition
            ),
            "induced_anchor_or_repetition": int(induced_reflection),
            "status": (
                "survived_falsification"
                if repair_survives
                else (
                    "failed_falsification"
                    if assay_valid
                    else "inconclusive_assay_invalid"
                )
            ),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether generalized-helix pulses accelerate correct answers "
            "or induce useful reflective repair"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--problems",
        "--problems-per-family",
        dest="problems",
        type=int,
        default=1,
        help="Number of held-out MATH-500 problems.",
    )
    parser.add_argument(
        "--calibration-problems",
        type=int,
        default=4,
        help=(
            "Small synthetic problems used internally to fit the helix. "
            "Their concise state activations are cached after the first run."
        ),
    )
    parser.add_argument(
        "--rebuild-internal-calibration",
        action="store_true",
        help="Ignore and replace file 06c's own layer calibration cache.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("math500",),
        default="math500",
        help=(
            "Behavioral task. math500 uses deterministic level 2-3 integer-"
            "answer MATH-500 items."
        ),
    )
    parser.add_argument(
        "--math500-path",
        default=None,
        help=(
            "Optional local MATH-500 test.jsonl. Otherwise it is downloaded "
            "once from Hugging Face and cached under output.root."
        ),
    )
    parser.add_argument(
        "--math500-levels",
        default="2,3",
        help="Comma-separated MATH-500 difficulty levels.",
    )
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument(
        "--generation-safety-ceiling",
        type=int,
        default=8192,
        help=(
            "Emergency loop guard, not a reasoning budget. Generation stops "
            "normally at a complete FINAL line or EOS."
        ),
    )
    parser.add_argument("--pulse-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--controls",
        choices=("core", "full"),
        default="core",
        help=(
            "core keeps baseline, generalized helix, linear, and closed k=1; "
            "full also runs opposite and random controls."
        ),
    )
    args = parser.parse_args()
    if args.problems <= 0:
        raise ValueError("--problems must be positive")
    if args.calibration_problems < 3:
        raise ValueError("--calibration-problems must be at least 3")
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.generation_safety_ceiling <= 0 or args.pulse_tokens <= 0:
        raise ValueError("token limits must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    try:
        math500_levels = {
            int(value.strip())
            for value in args.math500_levels.split(",")
            if value.strip()
        }
    except ValueError as exc:
        raise ValueError("--math500-levels must contain integers") from exc
    if args.benchmark == "math500" and not math500_levels:
        raise ValueError("--math500-levels cannot be empty")

    geometry_code = load_geometry_module()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 663)
    print(
        "Standalone 06c: no outputs from files 01-05, 05b, or 06b are read.",
        flush=True,
    )
    math500_path = (
        Path(args.math500_path)
        if args.math500_path
        else paths["root"] / "math500_test.jsonl"
    )
    evaluation_problems = load_math500_integer_problems(
        args.problems,
        math500_levels,
        int(config["study"]["seed"]) + 663,
        math500_path,
    )
    print("Loading model for behavioral interventions...", flush=True)
    backend = huggingface_collector_from_config(
        config["model"],
        config["collection"],
    )
    layer = int(args.layer)
    if layer < 0 or layer >= len(backend.layers):
        raise ValueError(
            f"layer {layer} is invalid; model has {len(backend.layers)} layers"
        )
    print(
        f"Loaded model; testing layer {layer} on "
        f"{len(evaluation_problems)} MATH-500 problems",
        flush=True,
    )
    geometry, _calibration_rows = fit_internal_calibration_geometry(
        backend,
        geometry_code,
        config,
        paths,
        layer,
        args.calibration_problems,
        args.rebuild_internal_calibration,
    )

    torch = backend.torch
    outcomes: list[dict] = []
    assay_rows: list[dict] = []
    baseline_pilots = {}
    pilot_rows = []
    for problem_index, problem in enumerate(evaluation_problems):
        incorrect = wrong_answer(problem)
        for scenario in ("accelerate", "repair_wrong_commitment"):
            prompt = evaluation_prompt(
                problem,
                "",
                scenario,
                incorrect,
            )
            for rollout in range(args.rollouts):
                seed = (
                    int(config["study"]["seed"])
                    + 1000 * problem_index
                    + 100 * (scenario == "repair_wrong_commitment")
                    + rollout
                )
                started = time.perf_counter()
                generated = backend.collect(
                    prompt,
                    [layer],
                    args.generation_safety_ceiling,
                    seed,
                    temperature=args.temperature,
                    disable_eos=False,
                    intervention=None,
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
                    problem.answer,
                    args.generation_safety_ceiling,
                )
                baseline_pilots[
                    (problem.problem_id, scenario, rollout)
                ] = generated
                pilot_rows.append(
                    {
                        "problem_id": problem.problem_id,
                        "family": problem.family,
                        "scenario": scenario,
                        "rollout": rollout,
                        "seed": seed,
                        "benchmark": args.benchmark,
                        "benchmark_level": problem.metadata.get("level"),
                        "generation_safety_ceiling": (
                            args.generation_safety_ceiling
                        ),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        **metrics,
                    }
                )
                print(
                    f"Baseline viability: {problem.problem_id} {scenario} "
                    f"rollout {rollout} "
                    f"({len(generated.token_ids)} tokens, "
                    f"{time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    failed_pilots = [
        row for row in pilot_rows if not int(row["answer_emitted"])
    ]
    if failed_pilots:
        diagnostic_path = (
            paths["tables"] / "behavioral_helix_baseline_pilot.csv"
        )
        write_csv(diagnostic_path, pilot_rows)
        failed_labels = ", ".join(
            f"{row['problem_id']}/{row['scenario']}/r{row['rollout']}"
            for row in failed_pilots
        )
        raise RuntimeError(
            "Behavioral assay aborted before interventions: the unmodified "
            "baseline reached the emergency generation safety ceiling "
            f"({args.generation_safety_ceiling} tokens) without FINAL for "
            f"{failed_labels}. Treat this as a looping or prompt-calibration "
            "failure rather than increasing the ceiling. "
            f"Diagnostics: {diagnostic_path}"
        )
    print(
        "Baseline viability passed for every prompt; starting causal assays.",
        flush=True,
    )
    control_count = 4 if args.controls == "core" else 6
    total_generations = (
        len(evaluation_problems) * 2 * args.rollouts * control_count
    )
    completed_generations = 0

    initial_progress = 0.0
    final_progress = 1.0
    geometry_family = "iterative_state_machine"
    for problem_index, problem in enumerate(evaluation_problems):

        def helix_delta(
            desired: float,
            source: float,
        ) -> np.ndarray:
            return geometry.helix_value(
                geometry_family,
                desired,
            ) - geometry.helix_value(geometry_family, source)

        def linear_delta(
            desired: float,
            source: float,
        ) -> np.ndarray:
            return geometry.axial_value(desired) - geometry.axial_value(
                source
            )

        def closed_delta(
            desired: float,
            source: float,
        ) -> np.ndarray:
            return linear_delta(desired, source) + (
                geometry.closed_after_axis_value(
                    geometry_family,
                    desired,
                )
                - geometry.closed_after_axis_value(
                    geometry_family,
                    source,
                )
            )

        model_directions = [
            geometry.axial_direction,
            *[
                coefficient
                for model in geometry.rotation_by_family.values()
                for coefficient in model.coefficients
            ],
        ]
        incorrect = wrong_answer(problem)
        for scenario in ("accelerate", "repair_wrong_commitment"):
            source_progress = (
                initial_progress
                if scenario == "accelerate"
                else final_progress
            )
            desired_progress = (
                final_progress
                if scenario == "accelerate"
                else initial_progress
            )
            desired = helix_delta(desired_progress, source_progress)
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm <= EPS:
                raise ValueError(
                    f"collapsed helix delta for {problem.problem_id}"
                )
            opposite = (
                -desired
            )
            linear = geometry_code.match_norm(
                linear_delta(desired_progress, source_progress),
                desired_norm,
            )[0]
            closed = geometry_code.match_norm(
                closed_delta(desired_progress, source_progress),
                desired_norm,
            )[0]
            random_delta = geometry_code.random_orthogonal_delta(
                len(desired),
                model_directions,
                desired_norm,
                rng,
            )
            controls = {
                "baseline": None,
                "helix_desired": desired,
                "helix_opposite": opposite,
                "linear_desired": linear,
                "linear_plus_closed_k1_desired": closed,
                "norm_matched_random": random_delta,
            }
            if args.controls == "core":
                controls = {
                    key: controls[key]
                    for key in (
                        "baseline",
                        "helix_desired",
                        "linear_desired",
                        "linear_plus_closed_k1_desired",
                    )
                }
            prompt = evaluation_prompt(
                problem,
                "",
                scenario,
                incorrect,
            )
            sequence_scores = {}
            for control in ("baseline", "helix_desired"):
                delta = controls[control]
                callback = (
                    None
                    if delta is None
                    else pulse_callback(torch, delta, args.pulse_tokens)
                )
                sequence_scores[control] = sequence_log_odds(
                    backend,
                    prompt,
                    problem.answer,
                    incorrect,
                    layer,
                    callback,
                )
                print(
                    f"Sequence score: {problem.problem_id} {scenario} "
                    f"{control}",
                    flush=True,
                )

            if scenario == "accelerate":
                positive_delta = answer_logit_direction(
                    backend,
                    problem.answer,
                    incorrect,
                    desired_norm,
                )
                positive_score = sequence_log_odds(
                    backend,
                    prompt,
                    problem.answer,
                    incorrect,
                    layer,
                    pulse_callback(
                        torch,
                        positive_delta,
                        args.pulse_tokens,
                    ),
                )
                assay_rows.append(
                    {
                        "problem_id": problem.problem_id,
                        "family": problem.family,
                        "scenario": scenario,
                        "baseline_log_odds": sequence_scores["baseline"],
                        "positive_control_log_odds": positive_score,
                    }
                )
                print(
                    f"Positive-control score: {problem.problem_id}",
                    flush=True,
                )

            for rollout in range(args.rollouts):
                seed = (
                    int(config["study"]["seed"])
                    + 1000 * problem_index
                    + 100 * (scenario == "repair_wrong_commitment")
                    + rollout
                )
                for control, delta in controls.items():
                    started = time.perf_counter()
                    callback = (
                        None
                        if delta is None
                        else pulse_callback(
                            torch,
                            delta,
                            args.pulse_tokens,
                        )
                    )
                    if control == "baseline":
                        generated = baseline_pilots[
                            (problem.problem_id, scenario, rollout)
                        ]
                    else:
                        generated = backend.collect(
                            prompt,
                            [layer],
                            args.generation_safety_ceiling,
                            seed,
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
                    completed_generations += 1
                    outcomes.append(
                        {
                            "layer": layer,
                            "problem_id": problem.problem_id,
                            "family": problem.family,
                            "scenario": scenario,
                            "rollout": rollout,
                            "seed": seed,
                            "control": control,
                            "benchmark": args.benchmark,
                            "benchmark_level": problem.metadata.get("level"),
                            "geometry_calibration_problems": (
                                args.calibration_problems
                            ),
                            "source_progress": source_progress,
                            "desired_progress": desired_progress,
                            "pulse_tokens": args.pulse_tokens,
                            "generation_safety_ceiling": (
                                args.generation_safety_ceiling
                            ),
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "chat_template_used": int(
                                bool(
                                    getattr(
                                        backend.tokenizer,
                                        "chat_template",
                                        None,
                                    )
                                )
                            ),
                            "thinking_mode_requested": (
                                backend.chat_template_kwargs.get(
                                    "enable_thinking"
                                )
                            ),
                            "intervention_norm": (
                                0.0
                                if delta is None
                                else float(np.linalg.norm(delta))
                            ),
                            "sequence_log_odds_correct": sequence_scores.get(
                                control,
                                float("nan"),
                            ),
                            **generated_metrics(
                                backend,
                                generated,
                                problem.answer,
                                args.generation_safety_ceiling,
                            ),
                        }
                    )
                    print(
                        f"Generation {completed_generations}/"
                        f"{total_generations}: {problem.problem_id} "
                        f"{scenario} {control} "
                        f"({len(generated.token_ids)} tokens, "
                        f"{time.perf_counter() - started:.1f}s)",
                        flush=True,
                    )
            print(
                f"Behavioral helix: {problem.problem_id} {scenario} done",
                flush=True,
            )

    summary = summarize(outcomes, assay_rows)
    write_csv(
        paths["tables"] / "behavioral_helix_outcomes.csv",
        outcomes,
    )
    write_csv(
        paths["tables"] / "behavioral_helix_key_results.csv",
        summary,
    )
    key = [row for row in summary if row["result_type"] == "primary_gate"]
    print(json.dumps(key, indent=2), flush=True)
    print(
        "Wrote the two behavioral-helix result tables to "
        f"{paths['tables']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
