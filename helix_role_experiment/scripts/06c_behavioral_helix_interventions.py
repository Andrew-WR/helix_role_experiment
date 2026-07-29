from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import parse_layer_spec, read_csv, write_csv
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
    read_jsonl,
    seed_everything,
)
from helix_role_experiment.controlled_tasks import (
    ControlledProblem,
    generate_suite,
)
from helix_role_experiment.models import huggingface_collector_from_config


EPS = 1e-12


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
        "\nContinue the reasoning from this state. Think for as long as needed. "
        "When ready, end with a separate line beginning `FINAL:` followed by "
        "only your answer."
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
    max_new_tokens: int,
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
            reasoning_tokens if correct else int(max_new_tokens) + 1
        ),
        "reasoning_tokens_before_final": reasoning_tokens,
        "output_tokens": len(generated.token_ids),
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


def select_problems(
    problems: dict[str, ControlledProblem],
    per_family: int,
    requested_families: set[str] | None = None,
) -> list[ControlledProblem]:
    by_family: dict[str, list[ControlledProblem]] = defaultdict(list)
    for problem in problems.values():
        by_family[problem.family].append(problem)
    output = []
    for family in sorted(by_family):
        if requested_families is not None and family not in requested_families:
            continue
        output.extend(
            sorted(
                by_family[family],
                key=lambda value: value.problem_id,
            )[:per_family]
        )
    return output


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
    assay_valid = bool(
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
            "status": "assay_valid" if assay_valid else "assay_invalid",
        }
    )

    summary_lookup = {
        (row["scenario"], row["control"]): row
        for row in output
        if row["result_type"] == "behavioral_condition"
    }

    def condition(scenario: str, control: str) -> dict:
        return summary_lookup[(scenario, control)]

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
    parser.add_argument("--problems-per-family", type=int, default=1)
    parser.add_argument(
        "--families",
        default="all",
        help=(
            "'all' or a comma-separated subset of iterative_state_machine,"
            "fictional_ontology,dependency_graph"
        ),
    )
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--pulse-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    if args.problems_per_family <= 0:
        raise ValueError("--problems-per-family must be positive")
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.max_new_tokens <= 0 or args.pulse_tokens <= 0:
        raise ValueError("token budgets must be positive")

    geometry_code = load_geometry_module()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 663)
    observational = read_csv(paths["tables"] / "observational_cross.csv")
    available_layers = sorted({int(row["layer"]) for row in observational})
    layer = parse_layer_spec(str(args.layer), available_layers)[0]
    prefixes = {
        row["variant_id"]: row
        for row in read_jsonl(paths["root"] / "counterfactual_prefixes.jsonl")
    }
    token_path = paths["tables"] / "counterfactual_actual_token_counts.csv"
    token_lookup = (
        {
            row["variant_id"]: int(row["actual_formatted_token_count"])
            for row in read_csv(token_path)
        }
        if token_path.is_file()
        else {}
    )
    with np.load(
        paths["tables"] / "counterfactual_activations.npz",
        allow_pickle=False,
    ) as data:
        activations = np.asarray(data["activations"])
        variant_ids = data["variant_ids"].astype(str)
        activation_layers = data["layers"].astype(int)
    if len(observational) != len(activations):
        raise ValueError("observational CSV and activation NPZ disagree")
    for index, row in enumerate(observational):
        if (
            row["variant_id"] != variant_ids[index]
            or int(row["layer"]) != int(activation_layers[index])
        ):
            raise ValueError("observational and activation rows are misaligned")

    all_problems = {
        problem.problem_id: problem
        for problem in generate_suite(
            int(config["tasks"]["problems_per_family"]),
            int(config["study"]["seed"]),
        )
    }
    available_families = {problem.family for problem in all_problems.values()}
    requested_families = None
    if args.families != "all":
        requested_families = {
            value.strip()
            for value in args.families.split(",")
            if value.strip()
        }
        missing_families = requested_families - available_families
        if missing_families:
            raise ValueError(
                f"unknown families: {sorted(missing_families)}"
            )
    selected_problems = select_problems(
        all_problems,
        args.problems_per_family,
        requested_families,
    )
    if not selected_problems:
        raise ValueError("no controlled problems were selected")
    print("Loading model for behavioral interventions...", flush=True)
    backend = huggingface_collector_from_config(
        config["model"],
        config["collection"],
    )
    print(
        f"Loaded model; testing layer {layer} on "
        f"{len(selected_problems)} problems",
        flush=True,
    )
    indices = np.flatnonzero(activation_layers == layer)
    layer_rows = [observational[index] for index in indices]
    layer_activations = np.asarray(activations[indices])
    centered = geometry_code.center_within_problem_trajectory(
        layer_activations,
        layer_rows,
    )
    ordinary = np.asarray(
        [
            row["condition"] in geometry_code.TRAJECTORY_CONDITIONS
            for row in layer_rows
        ],
        dtype=bool,
    )
    ordinary_rows = [
        row for row, keep in zip(layer_rows, ordinary) if keep
    ]
    ordinary_centered = centered[ordinary]
    missing_token_counts = [
        row["variant_id"]
        for row in ordinary_rows
        if row["variant_id"] not in token_lookup
    ]
    for variant_id in missing_token_counts:
        formatted = backend.format_prompt(prefixes[variant_id]["text"])
        token_lookup[variant_id] = len(
            backend.tokenizer(
                formatted,
                add_special_tokens=True,
            )["input_ids"]
        )
    if missing_token_counts:
        print(
            f"Computed {len(missing_token_counts)} tokenizer counts directly; "
            "file 05b is not required",
            flush=True,
        )
    ordinary_tokens = np.asarray(
        [token_lookup[row["variant_id"]] for row in ordinary_rows],
        dtype=np.float64,
    )
    ridge = float(config["analysis"].get("ridge", 0.001))
    geometry_by_problem = {}
    for problem in selected_problems:
        keep = np.asarray(
            [
                row["problem_id"] != problem.problem_id
                for row in ordinary_rows
            ],
            dtype=bool,
        )
        geometry_by_problem[problem.problem_id] = geometry_code.fit_geometry(
            [row for row, value in zip(ordinary_rows, keep) if value],
            ordinary_centered[keep],
            ordinary_tokens[keep],
            ridge,
        )[0]

    torch = backend.torch
    outcomes: list[dict] = []
    assay_rows: list[dict] = []
    rows_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in layer_rows:
        rows_by_problem[row["problem_id"]].append(row)
    total_generations = (
        len(selected_problems) * 2 * args.rollouts * 6
    )
    completed_generations = 0

    for problem_index, problem in enumerate(selected_problems):
        concise = sorted(
            [
                row
                for row in rows_by_problem[problem.problem_id]
                if row["condition"] == "concise"
            ],
            key=lambda row: float(row["structural_progress"]),
        )
        if len(concise) < 3:
            raise ValueError(
                f"{problem.problem_id} has too few concise states"
            )
        target = min(
            concise[1:-1],
            key=lambda row: abs(float(row["structural_progress"]) - 0.5),
        )
        target_progress = float(target["structural_progress"])
        geometry = geometry_by_problem[problem.problem_id]
        initial_progress = float(concise[0]["structural_progress"])
        final_progress = float(concise[-1]["structural_progress"])

        def helix_delta(desired: float) -> np.ndarray:
            return geometry.helix_value(
                problem.family,
                desired,
            ) - geometry.helix_value(problem.family, target_progress)

        def linear_delta(desired: float) -> np.ndarray:
            return geometry.axial_value(desired) - geometry.axial_value(
                target_progress
            )

        def closed_delta(desired: float) -> np.ndarray:
            return linear_delta(desired) + (
                geometry.closed_after_axis_value(problem.family, desired)
                - geometry.closed_after_axis_value(
                    problem.family,
                    target_progress,
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
            desired_progress = (
                final_progress
                if scenario == "accelerate"
                else initial_progress
            )
            opposite_progress = (
                initial_progress
                if scenario == "accelerate"
                else final_progress
            )
            desired = helix_delta(desired_progress)
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm <= EPS:
                raise ValueError(
                    f"collapsed helix delta for {problem.problem_id}"
                )
            opposite = geometry_code.match_norm(
                helix_delta(opposite_progress),
                desired_norm,
            )[0]
            linear = geometry_code.match_norm(
                linear_delta(desired_progress),
                desired_norm,
            )[0]
            closed = geometry_code.match_norm(
                closed_delta(desired_progress),
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
            prompt = evaluation_prompt(
                problem,
                prefixes[target["variant_id"]]["text"],
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
                    generated = backend.collect(
                        prompt,
                        [layer],
                        args.max_new_tokens,
                        seed,
                        temperature=args.temperature,
                        disable_eos=False,
                        intervention=callback,
                        capture_activations=False,
                        capture_eos_logits=False,
                        stop_regex=(
                            r"(?im)^\s*\**final(?:\s+answer)?"
                            r"\**\s*:\**\s*\S+"
                        ),
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
                            "target_progress": target_progress,
                            "desired_progress": desired_progress,
                            "pulse_tokens": args.pulse_tokens,
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
                                args.max_new_tokens,
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
