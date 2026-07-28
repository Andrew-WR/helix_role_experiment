from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from _common import write_csv
from helix_role_experiment.config import (
    atomic_json,
    config_hash,
    deterministic_id,
    ensure_output_dirs,
    environment_record,
    load_config,
    seed_everything,
    write_jsonl,
)
from helix_role_experiment.controlled_tasks import generate_suite
from helix_role_experiment.hooks import select_layer_indices
from helix_role_experiment.models import (
    SyntheticActivationBackend,
    huggingface_collector_from_config,
)
from helix_role_experiment.natural_tasks import load_natural_tasks
from helix_role_experiment.traces import TraceRecord, TraceStore, split_for_problem


def state_alignment(problem, length: int):
    state_indices = np.floor(
        np.linspace(0, len(problem.states) - 1, length)
    ).astype(int)
    states = [problem.states[index] for index in state_indices]
    return {
        "state_ids": [state.state_id for state in states],
        "progress": np.asarray([state.structural_progress for state in states]),
        "remaining": [float(state.remaining_distance) for state in states],
        "operation": [
            "planning"
            if index == 0
            else ("checking" if state.remaining_distance == 0 else "calculation")
            for index, state in enumerate(states)
        ],
        "confidence": np.asarray([0.45 + 0.45 * state.structural_progress for state in states]),
        "termination_allowed": np.asarray(
            [state.answer_known and not state.verification_remaining for state in states],
            dtype=bool,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect token-aligned activation traces")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    seed = int(config["study"]["seed"])
    seed_everything(seed)
    environment_path = paths["root"] / "environment.json"
    current_environment = environment_record(config)
    if environment_path.exists():
        import json

        previous_environment = json.loads(
            environment_path.read_text(encoding="utf-8")
        )
        if previous_environment.get("config_hash") != current_environment["config_hash"]:
            raise RuntimeError(
                f"{paths['root']} already contains a different config hash "
                f"({previous_environment.get('config_hash')}); choose a new "
                "output.root rather than mixing experiments"
            )
    atomic_json(environment_path, current_environment)
    problems = generate_suite(int(config["tasks"]["problems_per_family"]), seed)
    natural_path = config["tasks"].get("natural_jsonl")
    if natural_path:
        if config["model"]["backend"] == "synthetic":
            raise ValueError(
                "natural_jsonl requires a language-model backend; synthetic "
                "known-latent traces cannot establish external validity"
            )
        natural = load_natural_tasks(natural_path)
        problems.extend(
            SimpleNamespace(
                problem_id=row["problem_id"],
                family=row["task_family"],
                prompt=row["prompt"],
                answer=row["reference_answer"],
                states=[],
                metadata=row.get("metadata", {}),
                to_json=lambda row=row: row,
            )
            for row in natural
        )
    if args.limit is not None:
        problems = problems[: args.limit]
    write_jsonl(
        paths["root"] / "problems.jsonl", [problem.to_json() for problem in problems]
    )
    trace_store = TraceStore(paths["traces"])
    backend_name = config["model"]["backend"]
    if backend_name == "synthetic":
        backend = SyntheticActivationBackend(
            hidden_size=int(config["model"]["hidden_size"]),
            layers=int(config["model"]["layers"]),
            seed=seed,
            role=config["model"]["synthetic_role"],
            noise=float(config["model"]["synthetic_noise"]),
        )
        layer_indices = list(range(backend.layers))
    elif backend_name == "huggingface":
        backend = huggingface_collector_from_config(
            config["model"], config["collection"]
        )
        configured = config["collection"]["layers"]
        layer_indices = select_layer_indices(
            configured,
            len(backend.layers),
            backend.adapter_target_layers,
        )
        print(
            f"Resolved base model {backend.model_id!r}; "
            f"adapter={'enabled' if backend.adapter_enabled else 'disabled'}; "
            f"adapter target layers={backend.adapter_target_layers or 'not declared'}; "
            f"recording layers={layer_indices}"
        )
    else:
        raise ValueError(f"unknown backend: {backend_name}")

    summary = []
    for problem_index, problem in enumerate(problems):
        request_base = deterministic_id(
            config_hash(config), seed, problem.problem_id, "normal"
        )
        if backend_name == "synthetic":
            length = int(config["collection"]["synthetic_base_length"]) + (
                problem_index % int(config["collection"]["synthetic_length_jitter"])
            )
            labels = state_alignment(problem, length)
            generation = backend.generate(
                request_base,
                length,
                labels["progress"],
                confidence=labels["confidence"],
                termination_allowed=labels["termination_allowed"],
            )
        else:
            generation = backend.collect(
                problem.prompt,
                layer_indices,
                int(config["collection"]["max_new_tokens"]),
                seed + problem_index,
                temperature=float(config["collection"]["temperature"]),
                disable_eos=False,
            )
            length = len(generation.token_ids)
            labels = {
                "state_ids": [],
                "progress": np.array([]),
                "remaining": [],
                "operation": [],
                "confidence": np.array([]),
                "termination_allowed": np.array([]),
            }
        for layer in layer_indices:
            request_id = deterministic_id(request_base, layer)
            record = TraceRecord(
                request_id=request_id,
                problem_id=problem.problem_id,
                task_family=problem.family,
                condition="normal",
                split=split_for_problem(problem.problem_id, seed),
                layer=layer,
                prompt_token_count=generation.prompt_token_count,
                token_ids=generation.token_ids,
                tokens=generation.tokens,
                activation_file="",
                generated_token_count=length,
                reached_eos=generation.reached_eos,
                truncated=not generation.reached_eos,
                model_id=(
                    backend.model_id
                    if backend_name == "huggingface"
                    else config["model"].get("id", "synthetic")
                ),
                model_revision=(
                    backend.revision
                    if backend_name == "huggingface"
                    else config["model"].get("revision")
                ),
                tokenizer_revision=(
                    backend.tokenizer_revision
                    if backend_name == "huggingface"
                    else config["model"].get("tokenizer_revision")
                ),
                seed=seed + problem_index,
                state_ids=labels["state_ids"],
                structural_progress=labels["progress"].tolist(),
                remaining_distance=labels["remaining"],
                operation=labels["operation"],
                confidence=labels["confidence"].tolist(),
                eos_logit=generation.eos_logits,
                termination_allowed=labels["termination_allowed"].tolist(),
                metadata={
                    "backend": backend_name,
                    "adapter_path": (
                        backend.adapter_path
                        if backend_name == "huggingface"
                        else None
                    ),
                    "adapter_enabled": (
                        backend.adapter_enabled
                        if backend_name == "huggingface"
                        else False
                    ),
                    "adapter_target_layers": (
                        backend.adapter_target_layers
                        if backend_name == "huggingface"
                        else []
                    ),
                    "activation_dtype": str(
                        generation.activations_by_layer[layer].dtype
                    ),
                    "hook_alignment": "activation_used_to_predict_aligned_generated_token",
                },
            )
            trace_store.write(record, generation.activations_by_layer[layer])
            summary.append(
                {
                    "request_id": request_id,
                    "problem_id": problem.problem_id,
                    "family": problem.family,
                    "layer": layer,
                    "length": length,
                    "split": record.split,
                    "reached_eos": record.reached_eos,
                }
            )
    write_csv(paths["tables"] / "collection_summary.csv", summary)
    print(f"Collected {len(summary)} layer-traces from {len(problems)} problems in {paths['root']}")


if __name__ == "__main__":
    main()
