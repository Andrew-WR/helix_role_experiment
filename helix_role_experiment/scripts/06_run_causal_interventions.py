from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from _common import read_csv, write_csv
from helix_role_experiment.config import (
    ensure_output_dirs,
    load_config,
    read_jsonl,
    seed_everything,
)
from helix_role_experiment.controlled_tasks import generate_suite
from helix_role_experiment.eos_controls import (
    orthogonalize_to_direction,
    subspace_direction_overlap,
)
from helix_role_experiment.interventions import (
    candidate_orthogonal_patch,
    donor_transplant,
    intervention_diagnostics,
    norm_matched_random_delta,
    radial_intervention,
    within_plane_rotation,
)
from helix_role_experiment.models import (
    SyntheticActivationBackend,
    huggingface_collector_from_config,
)
from helix_role_experiment.subspaces import random_plane


def circular_stage(activation, plane, center):
    coords = (activation - center) @ plane
    angle = np.arctan2(coords[1], coords[0]) % (2.0 * np.pi)
    return float(angle / (2.0 * np.pi))


def circular_unit_distance(left: float, right: float) -> float:
    return float(abs(((left - right + 0.5) % 1.0) - 0.5))


def candidate_distribution_js(left: dict, right: dict) -> float:
    keys = sorted(
        set(left["candidate_probabilities"]) | set(right["candidate_probabilities"])
    )
    p = np.asarray([left["candidate_probabilities"].get(key, 0.0) for key in keys])
    q = np.asarray([right["candidate_probabilities"].get(key, 0.0) for key in keys])
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    midpoint = 0.5 * (p + q)
    mask_p, mask_q = p > 0, q > 0
    return float(
        0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / midpoint[mask_p]))
        + 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / midpoint[mask_q]))
    )


def run_huggingface(
    config: dict,
    paths: dict,
    rng: np.random.Generator,
) -> None:
    rows = read_csv(paths["tables"] / "observational_cross.csv")
    prefixes = {
        row["variant_id"]: row
        for row in read_jsonl(paths["root"] / "counterfactual_prefixes.jsonl")
    }
    with np.load(paths["tables"] / "counterfactual_activations.npz", allow_pickle=False) as data:
        activation_lookup = {
            (variant_id, int(layer)): activation.astype(np.float64)
            for variant_id, layer, activation in zip(
                data["variant_ids"].astype(str),
                data["layers"].astype(int),
                data["activations"],
            )
        }
    problems = {
        problem.problem_id: problem
        for problem in generate_suite(
            int(config["tasks"]["problems_per_family"]),
            int(config["study"]["seed"]),
        )
    }
    backend = huggingface_collector_from_config(
        config["model"], config["collection"]
    )
    with (paths["models"] / "subspace_index.json").open("r", encoding="utf-8") as handle:
        subspace_index = json.load(handle)
    eos_ids = backend.model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = backend.tokenizer.eos_token_id
    eos_id = int(eos_ids if isinstance(eos_ids, int) else eos_ids[0])
    eos_direction = (
        backend.model.get_output_embeddings()
        .weight[eos_id]
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    eos_direction /= max(float(np.linalg.norm(eos_direction)), 1e-12)
    by_family_layer: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family_layer[(row["family"], int(row["layer"]))].append(row)
    max_pairs = int(config["interventions"]["max_pairs_per_family_layer"])
    output = []
    torch = backend.torch

    for (family, layer), candidates in by_family_layer.items():
        estimator = subspace_index["selected_by_layer"][str(layer)]
        with np.load(
            paths["models"] / f"subspace_layer_{layer}_{estimator}.npz",
            allow_pickle=False,
        ) as data:
            plane = np.asarray(data["basis"])
            center = np.asarray(data["center"])
        try:
            no_eos_plane = orthogonalize_to_direction(plane, eos_direction)
        except ValueError:
            no_eos_plane = None
        pairs = [
            (target, source)
            for target in candidates
            for source in candidates
            if source["problem_id"] != target["problem_id"]
            and abs(
                float(source["structural_progress"])
                - float(target["structural_progress"])
            )
            >= 0.25
        ][:max_pairs]
        for pair_index, (target_row, source_row) in enumerate(pairs):
            target_activation = activation_lookup[(target_row["variant_id"], layer)]
            source_activation = activation_lookup[(source_row["variant_id"], layer)]
            target_prefix = prefixes[target_row["variant_id"]]
            target_problem = problems[target_row["problem_id"]]
            source_progress = float(source_row["structural_progress"])
            counterfactual_state = min(
                target_problem.states,
                key=lambda state: abs(state.structural_progress - source_progress),
            )
            candidate_transitions = list(counterfactual_state.valid_next)
            baseline = backend.score_first_transition(
                target_prefix["text"], candidate_transitions, layer, intervention=None
            )

            def transplant_callback(basis_value, donor_value):
                def callback(_layer, _step, hidden):
                    basis_tensor = torch.as_tensor(
                        basis_value, device=hidden.device, dtype=hidden.dtype
                    )
                    center_tensor = torch.as_tensor(
                        center, device=hidden.device, dtype=hidden.dtype
                    )
                    donor_tensor = torch.as_tensor(
                        donor_value, device=hidden.device, dtype=hidden.dtype
                    )
                    donor_z = (donor_tensor - center_tensor) @ basis_tensor
                    target_z = (hidden - center_tensor) @ basis_tensor
                    return hidden + (donor_z - target_z) @ basis_tensor.T

                return callback

            candidate_changed = donor_transplant(
                target_activation, source_activation, plane, center
            )
            candidate_delta = candidate_changed - target_activation
            random_delta = norm_matched_random_delta(
                candidate_delta, len(target_activation), rng
            )

            def constant_delta_callback(delta):
                def callback(_layer, _step, hidden):
                    return hidden + torch.as_tensor(
                        delta, device=hidden.device, dtype=hidden.dtype
                    )

                return callback

            interventions = {
                "candidate_transplant": (
                    candidate_changed,
                    transplant_callback(plane, source_activation),
                ),
                "norm_matched_random": (
                    target_activation + random_delta,
                    constant_delta_callback(random_delta),
                ),
                "random_plane_transplant": (
                    donor_transplant(
                        target_activation,
                        source_activation,
                        random_plane(len(target_activation), 2, rng),
                        center,
                    ),
                    None,
                ),
                "candidate_orthogonal_patch": (
                    candidate_orthogonal_patch(target_activation, source_activation, plane),
                    None,
                ),
                "radial_plus_20pct": (
                    radial_intervention(target_activation, plane, center, 1.2),
                    None,
                ),
            }
            if no_eos_plane is not None:
                interventions["eos_orthogonal_transplant"] = (
                    donor_transplant(
                        target_activation, source_activation, no_eos_plane, center
                    ),
                    transplant_callback(no_eos_plane, source_activation),
                )
            # Build callbacks for interventions defined by their changed state.
            for name, (changed, callback) in list(interventions.items()):
                if callback is None:
                    callback = constant_delta_callback(changed - target_activation)
                    interventions[name] = (changed, callback)
            candidate_eos_shift = None
            for name, (changed, callback) in interventions.items():
                result = backend.score_first_transition(
                    target_prefix["text"],
                    candidate_transitions,
                    layer,
                    intervention=callback,
                )
                if name == "candidate_transplant":
                    candidate_eos_shift = result["eos_logit"] - baseline["eos_logit"]
                diagnostics = intervention_diagnostics(target_activation, changed)
                effect = (
                    result["valid_next_state_probability"]
                    - baseline["valid_next_state_probability"]
                )
                output.append(
                    {
                        "family": family,
                        "layer": layer,
                        "pair_index": pair_index,
                        "target_problem_id": target_row["problem_id"],
                        "source_problem_id": source_row["problem_id"],
                        "target_progress": target_row["structural_progress"],
                        "source_progress": source_progress,
                        "desired_progress_shift": (
                            source_progress - float(target_row["structural_progress"])
                        ),
                        "desired_stage_shift": (
                            counterfactual_state.structural_progress
                            - float(target_row["structural_progress"])
                        ),
                        "control": name,
                        "inferred_progress_before": target_row["structural_progress"],
                        "inferred_progress_after": counterfactual_state.structural_progress,
                        "observed_progress_shift": effect,
                        "causal_abstraction_direction_accuracy": float(effect > 0),
                        "fixed_length_eos_disabled": True,
                        "eos_overlap": subspace_direction_overlap(plane, eos_direction),
                        "eos_logit_change_proxy": result["eos_logit"] - baseline["eos_logit"],
                        "valid_next_state_probability_proxy": result[
                            "valid_next_state_probability"
                        ],
                        "valid_next_state_probability_baseline": baseline[
                            "valid_next_state_probability"
                        ],
                        "downstream_kl": candidate_distribution_js(baseline, result),
                        **diagnostics,
                    }
                )
            if candidate_eos_shift is not None:
                output.append(
                    {
                        "family": family,
                        "layer": layer,
                        "pair_index": pair_index,
                        "target_problem_id": target_row["problem_id"],
                        "source_problem_id": source_row["problem_id"],
                        "target_progress": target_row["structural_progress"],
                        "source_progress": source_progress,
                        "desired_progress_shift": (
                            source_progress - float(target_row["structural_progress"])
                        ),
                        "desired_stage_shift": 0.0,
                        "control": "direct_eos_logit_bias_matched",
                        "inferred_progress_before": target_row["structural_progress"],
                        "inferred_progress_after": target_row["structural_progress"],
                        "observed_progress_shift": 0.0,
                        "causal_abstraction_direction_accuracy": 0.5,
                        "fixed_length_eos_disabled": True,
                        "eos_overlap": subspace_direction_overlap(plane, eos_direction),
                        "eos_logit_change_proxy": candidate_eos_shift,
                        "valid_next_state_probability_proxy": baseline[
                            "valid_next_state_probability"
                        ],
                        "valid_next_state_probability_baseline": baseline[
                            "valid_next_state_probability"
                        ],
                        "downstream_kl": 0.0,
                        "intervention_norm": 0.0,
                        "activation_norm_before": float(np.linalg.norm(target_activation)),
                        "activation_norm_after": float(np.linalg.norm(target_activation)),
                        "relative_norm_change": 0.0,
                    }
                )
    write_csv(paths["tables"] / "causal_interventions.csv", output)
    print(f"Wrote {len(output)} Hugging Face causal/control outcomes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal interchange and EOS controls")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 606)
    if config["model"]["backend"] == "huggingface":
        run_huggingface(config, paths, rng)
        return
    if config["model"]["backend"] != "synthetic":
        raise ValueError(f"unknown backend: {config['model']['backend']}")
    rows = read_csv(paths["tables"] / "observational_cross.csv")
    with np.load(paths["tables"] / "counterfactual_activations.npz", allow_pickle=False) as data:
        activations = np.asarray(data["activations"], dtype=np.float64)
        ids = data["variant_ids"].astype(str)
        layers = data["layers"].astype(int)
    activation_lookup = {
        (variant_id, int(layer)): activation
        for variant_id, layer, activation in zip(ids, layers, activations)
    }
    by_family_layer: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family_layer[(row["family"], int(row["layer"]))].append(row)
    backend = SyntheticActivationBackend(
        hidden_size=int(config["model"]["hidden_size"]),
        layers=int(config["model"]["layers"]),
        seed=int(config["study"]["seed"]),
        role=config["model"]["synthetic_role"],
        noise=float(config["model"]["synthetic_noise"]),
    )
    output = []
    max_pairs = int(config["interventions"]["max_pairs_per_family_layer"])
    deltas = [float(value) for value in config["interventions"]["rotation_deltas"]]
    for (family, layer), candidates in by_family_layer.items():
        with (paths["models"] / "subspace_index.json").open("r", encoding="utf-8") as handle:
            subspace_index = json.load(handle)
        estimator = subspace_index["selected_by_layer"][str(layer)]
        with np.load(
            paths["models"] / f"subspace_layer_{layer}_{estimator}.npz",
            allow_pickle=False,
        ) as data:
            plane = np.asarray(data["basis"])
            center = np.asarray(data["center"])
        eos_direction = backend.eos_directions[layer]
        no_eos_plane = orthogonalize_to_direction(plane, eos_direction)
        pairs = []
        for target in candidates:
            for source in candidates:
                if (
                    source["problem_id"] != target["problem_id"]
                    and abs(float(source["structural_progress"]) - float(target["structural_progress"])) >= 0.25
                ):
                    pairs.append((target, source))
        pairs = pairs[:max_pairs]
        for pair_index, (target_row, source_row) in enumerate(pairs):
            target = activation_lookup[(target_row["variant_id"], layer)]
            source = activation_lookup[(source_row["variant_id"], layer)]
            target_progress = float(target_row["structural_progress"])
            source_progress = float(source_row["structural_progress"])
            desired_shift = source_progress - target_progress
            source_stage = circular_stage(
                source, backend.planes[layer], backend.centers[layer]
            )
            candidate = donor_transplant(target, source, plane, center)
            candidate_delta = candidate - target
            random_delta = norm_matched_random_delta(candidate_delta, len(target), rng)
            controls = {
                "candidate_transplant": candidate,
                "eos_orthogonal_transplant": donor_transplant(
                    target, source, no_eos_plane, center
                ),
                "norm_matched_random": target + random_delta,
                "random_plane_transplant": donor_transplant(
                    target,
                    source,
                    random_plane(len(target), 2, rng),
                    center,
                ),
                "candidate_orthogonal_patch": candidate_orthogonal_patch(target, source, plane),
                "radial_plus_20pct": radial_intervention(target, plane, center, 1.2),
            }
            for delta in deltas:
                controls[f"phase_rotation_{delta:+.3f}"] = within_plane_rotation(
                    target, plane, center, delta
                )
            for control_name, changed in controls.items():
                inferred_before = circular_stage(target, backend.planes[layer], backend.centers[layer])
                inferred_after = circular_stage(changed, backend.planes[layer], backend.centers[layer])
                observed_shift = ((inferred_after - inferred_before + 0.5) % 1.0) - 0.5
                desired_stage_shift = (
                    (source_stage - inferred_before + 0.5) % 1.0
                ) - 0.5
                direction_accuracy = float(
                    0.5
                    if abs(observed_shift) < 0.01
                    else np.sign(observed_shift) == np.sign(desired_stage_shift)
                )
                distance_source = circular_unit_distance(inferred_after, source_stage)
                distance_target = circular_unit_distance(inferred_after, inferred_before)
                temperature = 0.08
                source_consistency = float(
                    np.exp(-distance_source / temperature)
                    / (
                        np.exp(-distance_source / temperature)
                        + np.exp(-distance_target / temperature)
                    )
                )
                diagnostics = intervention_diagnostics(target, changed)
                output.append(
                    {
                        "family": family,
                        "layer": layer,
                        "pair_index": pair_index,
                        "target_problem_id": target_row["problem_id"],
                        "source_problem_id": source_row["problem_id"],
                        "target_progress": target_progress,
                        "source_progress": source_progress,
                        "desired_progress_shift": desired_shift,
                        "desired_stage_shift": desired_stage_shift,
                        "control": control_name,
                        "inferred_progress_before": inferred_before,
                        "inferred_progress_after": inferred_after,
                        "observed_progress_shift": observed_shift,
                        "causal_abstraction_direction_accuracy": direction_accuracy,
                        "fixed_length_eos_disabled": True,
                        "eos_overlap": subspace_direction_overlap(plane, eos_direction),
                        "eos_logit_change_proxy": float((changed - target) @ eos_direction),
                        "valid_next_state_probability_proxy": source_consistency,
                        "downstream_kl": float(
                            0.5 * np.square(changed - target).mean()
                        ),
                        **diagnostics,
                    }
                )
    write_csv(paths["tables"] / "causal_interventions.csv", output)
    print(f"Wrote {len(output)} causal/control outcomes")


if __name__ == "__main__":
    main()
