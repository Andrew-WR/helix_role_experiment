from __future__ import annotations

import argparse
import json

import numpy as np

from _common import write_csv
from helix_role_experiment.config import (
    deterministic_id,
    ensure_output_dirs,
    load_config,
    read_jsonl,
)
from helix_role_experiment.models import (
    SyntheticActivationBackend,
    huggingface_collector_from_config,
)
from helix_role_experiment.subspaces import projected_features


OPERATIONS = (
    "planning",
    "calculation",
    "uncertainty",
    "backtracking",
    "checking",
    "consolidation",
    "final_emission",
)


def synthetic_schedule(row: dict, length: int) -> np.ndarray:
    final = float(row["structural_progress"])
    if row["condition"] == "teleport":
        progress = np.zeros(length)
        progress[max(1, int(0.8 * length)) :] = final
        return progress
    if row["condition"] == "rollback":
        peak = min(1.0, final + 0.35)
        split = max(2, int(0.7 * length))
        return np.r_[np.linspace(0.0, peak, split), np.linspace(peak, final, length - split)]
    if row["condition"] == "loop":
        split = max(2, length // 2)
        return np.r_[np.linspace(0.0, final, split), np.full(length - split, final)]
    return np.linspace(0.0, final, length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run progress-position observational cross")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    prefixes = read_jsonl(paths["root"] / "counterfactual_prefixes.jsonl")
    with (paths["models"] / "subspace_index.json").open("r", encoding="utf-8") as handle:
        subspace_index = json.load(handle)
    layers = sorted(int(value) for value in subspace_index["selected_by_layer"])
    models = {}
    for layer in layers:
        estimator = subspace_index["selected_by_layer"][str(layer)]
        with np.load(
            paths["models"] / f"subspace_layer_{layer}_{estimator}.npz",
            allow_pickle=False,
        ) as data:
            models[layer] = {key: np.asarray(data[key]) for key in data.files}
    backend_name = config["model"]["backend"]
    if backend_name == "synthetic":
        backend = SyntheticActivationBackend(
            hidden_size=int(config["model"]["hidden_size"]),
            layers=int(config["model"]["layers"]),
            seed=int(config["study"]["seed"]),
            role=config["model"]["synthetic_role"],
            noise=float(config["model"]["synthetic_noise"]),
        )
    else:
        backend = huggingface_collector_from_config(
            config["model"], config["collection"]
        )
    output_rows = []
    activation_rows = []
    activation_ids = []
    activation_layers = []
    reference_tokens = int(config["counterfactuals"]["position_reference_tokens"])
    max_prefix_tokens = int(config["counterfactuals"]["max_prefix_proxy_tokens"])
    for row_index, row in enumerate(prefixes):
        if not row["exact_state_valid"]:
            continue
        request_id = deterministic_id(row["variant_id"], "observational")
        if backend_name == "synthetic":
            length = int(np.clip(row["token_count_proxy"], 8, max_prefix_tokens))
            progress = synthetic_schedule(row, length)
            position = np.arange(length) / max(1, reference_tokens - 1)
            confidence = np.linspace(0.4, float(row["confidence"]), length)
            termination = np.full(length, bool(row["termination_allowed"]))
            generated = backend.generate(
                request_id,
                length,
                progress,
                position=position,
                confidence=confidence,
                termination_allowed=termination,
            )
        else:
            generated = backend.collect(
                row["text"],
                layers,
                int(config["collection"]["observational_probe_tokens"]),
                int(config["study"]["seed"]) + row_index,
                temperature=0.0,
                disable_eos=True,
            )
            length = len(generated.token_ids)
        for layer in layers:
            activation = generated.activations_by_layer[layer][-1]
            model = models[layer]
            feature = projected_features(
                activation[None, :],
                model["basis"],
                model["center"],
                model["whitener"],
                float(model["radius_threshold"]),
            )
            output = {
                "variant_id": row["variant_id"],
                "problem_id": row["problem_id"],
                "family": row["family"],
                "condition": row["condition"],
                "layer": layer,
                "token_count_proxy": row["token_count_proxy"],
                "sentence_count_proxy": row["text"].count("."),
                "structural_progress": row["structural_progress"],
                "remaining_distance": row["remaining_distance"],
                "operation": row["operation"],
                "termination_allowed": int(row["termination_allowed"]),
                "confidence": row["confidence"],
                "eos_logit": generated.eos_logits[-1],
                "coordinate_1": float(feature["coordinate_1"][0]),
                "coordinate_2": float(feature["coordinate_2"][0]),
                "radius": float(feature["radius"][0]),
                "raw_angle": float(feature["raw_angle"][0]),
                "phase_reliable": bool(feature["phase_reliable"][0]),
                "manifold_distance": float(feature["manifold_distance"][0]),
                "request_id": request_id,
            }
            for operation in OPERATIONS:
                output[f"operation_{operation}"] = int(row["operation"] == operation)
            output_rows.append(output)
            activation_rows.append(activation)
            activation_ids.append(row["variant_id"])
            activation_layers.append(layer)
    write_csv(paths["tables"] / "observational_cross.csv", output_rows)
    activation_storage_dtype = (
        np.float16
        if config["collection"].get("activation_dtype", "float32") == "float16"
        else np.float32
    )
    np.savez_compressed(
        paths["tables"] / "counterfactual_activations.npz",
        activations=np.asarray(activation_rows, dtype=activation_storage_dtype),
        variant_ids=np.asarray(activation_ids),
        layers=np.asarray(activation_layers, dtype=int),
    )
    print(f"Recorded {len(output_rows)} counterfactual layer-observations")


if __name__ == "__main__":
    main()
