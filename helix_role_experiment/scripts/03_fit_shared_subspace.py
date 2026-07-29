from __future__ import annotations

import argparse
import time

import numpy as np

from _common import parse_layer_spec, write_csv
from helix_role_experiment.config import atomic_json, ensure_output_dirs, load_config, seed_everything
from helix_role_experiment.subspaces import (
    fit_complex_coefficient_model,
    fit_whitener,
    generalized_spectral_plane,
    generalized_spectral_plane_iterative,
    grassmann_mean,
    heldout_spectral_selectivity,
    random_plane,
    trace_harmonic_plane,
)
from helix_role_experiment.traces import TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit frozen shared candidate subspaces")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--layers",
        default=None,
        help="'all', 'late-half', comma-separated layers, or ranges such as 32-63",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 202)
    store = TraceStore(paths["traces"])
    manifest = store.read_manifest()
    selected_layers = set(
        parse_layer_spec(
            args.layers,
            [int(row["layer"]) for row in manifest],
        )
    )
    print(f"Shared-subspace fit layers: {sorted(selected_layers)}", flush=True)
    by_layer: dict[int, list[tuple[dict, np.ndarray]]] = {}
    for row, activations in store.iter_traces():
        if (
            int(row["layer"]) in selected_layers
            and len(activations) >= int(config["analysis"]["minimum_trace_length"])
        ):
            by_layer.setdefault(int(row["layer"]), []).append((row, activations))
    index = {"estimators": [], "selection_rule": "validation spectral selectivity only"}
    evaluation_rows = []
    random_draws = int(config["subspace"]["random_plane_draws"])
    for layer, items in sorted(by_layer.items()):
        calibration = [x for row, x in items if row["split"] == "calibration"]
        validation = [x for row, x in items if row["split"] == "validation"]
        test = [x for row, x in items if row["split"] == "test"]
        if len(calibration) < 2:
            calibration = [x for _, x in items[: max(2, len(items) // 2)]]
        if not validation:
            calibration_ids = {id(x) for x in calibration}
            validation = [x for _, x in items if id(x) not in calibration_ids]
        if not validation:
            validation = calibration
        if not test:
            test = validation
        planes = [trace_harmonic_plane(trace)[0] for trace in calibration]
        candidates: dict[str, np.ndarray] = {}
        fit_seconds = {}
        started = time.perf_counter()
        candidates["grassmann"], spectrum = grassmann_mean(planes)
        fit_seconds["grassmann"] = time.perf_counter() - started
        ridge_values = [float(value) for value in config["subspace"]["ridge_grid"]]
        best_ridge, best_score, best_plane = None, -np.inf, None
        generalized_started = time.perf_counter()
        generalized_algorithm = "dense"
        generalized_iterations = None
        threshold = int(config["subspace"].get("iterative_dimension_threshold", 512))
        for ridge in ridge_values:
            if calibration[0].shape[1] >= threshold:
                generalized_algorithm = "dual_conjugate_gradient"
                plane, _, generalized_iterations = generalized_spectral_plane_iterative(
                    calibration,
                    ridge,
                    tolerance=float(config["subspace"].get("cg_tolerance", 1e-6)),
                    max_iterations=int(config["subspace"].get("cg_max_iterations", 300)),
                )
            else:
                plane, _ = generalized_spectral_plane(calibration, ridge)
            score = float(np.mean([heldout_spectral_selectivity(x, plane) for x in validation]))
            if score > best_score:
                best_ridge, best_score, best_plane = ridge, score, plane
        assert best_plane is not None
        candidates["generalized_eigen"] = best_plane
        fit_seconds["generalized_eigen"] = time.perf_counter() - generalized_started
        started = time.perf_counter()
        complex_model = fit_complex_coefficient_model(calibration, complex_rank=2)
        candidates["complex_svd"] = complex_model.real_plane
        fit_seconds["complex_svd"] = time.perf_counter() - started
        estimator_scores = {}
        for name, plane in candidates.items():
            whitener, center = fit_whitener(calibration, plane)
            calibration_radii = np.concatenate(
                [
                    np.linalg.norm((trace - center) @ plane @ whitener, axis=1)
                    for trace in calibration
                ]
            )
            radius_threshold = float(
                np.quantile(calibration_radii, float(config["analysis"]["radius_quantile"]))
            )
            model_path = paths["models"] / f"subspace_layer_{layer}_{name}.npz"
            np.savez_compressed(
                model_path,
                basis=plane,
                whitener=whitener,
                center=center,
                radius_threshold=np.asarray(radius_threshold),
            )
            validation_scores = [heldout_spectral_selectivity(x, plane) for x in validation]
            test_scores = [heldout_spectral_selectivity(x, plane) for x in test]
            estimator_scores[name] = float(np.mean(validation_scores))
            evaluation_rows.append(
                {
                    "layer": layer,
                    "estimator": name,
                    "validation_selectivity_mean": float(np.mean(validation_scores)),
                    "test_selectivity_mean": float(np.mean(test_scores)),
                    "test_selectivity_std": float(np.std(test_scores)),
                    "calibration_traces": len(calibration),
                    "validation_traces": len(validation),
                    "test_traces": len(test),
                    "radius_threshold": radius_threshold,
                    "ridge": best_ridge if name == "generalized_eigen" else None,
                    "complex_component_1_fraction": (
                        float(complex_model.variance_fraction[0])
                        if name == "complex_svd"
                        else None
                    ),
                    "fit_seconds": fit_seconds[name],
                    "algorithm": (
                        generalized_algorithm
                        if name == "generalized_eigen"
                        else name
                    ),
                    "cg_iterations": (
                        generalized_iterations
                        if name == "generalized_eigen"
                        else None
                    ),
                    "dense_covariance_bytes": int(
                        calibration[0].shape[1] ** 2 * 8 * 2
                    ),
                }
            )
            index["estimators"].append(
                {
                    "layer": layer,
                    "name": name,
                    "path": str(model_path),
                    "validation_selectivity": estimator_scores[name],
                }
            )
        dimension = calibration[0].shape[1]
        for draw in range(random_draws):
            plane = random_plane(dimension, 2, rng)
            evaluation_rows.append(
                {
                    "layer": layer,
                    "estimator": "random_plane",
                    "draw": draw,
                    "validation_selectivity_mean": float(
                        np.mean([heldout_spectral_selectivity(x, plane) for x in validation])
                    ),
                    "test_selectivity_mean": float(
                        np.mean([heldout_spectral_selectivity(x, plane) for x in test])
                    ),
                }
            )
        winner = max(estimator_scores, key=estimator_scores.get)
        index.setdefault("selected_by_layer", {})[str(layer)] = winner
        print(
            f"Shared-subspace fit: layer {layer} selected {winner}",
            flush=True,
        )
    write_csv(paths["tables"] / "shared_subspace_evaluation.csv", evaluation_rows)
    atomic_json(paths["models"] / "subspace_index.json", index)
    print(f"Fit shared subspaces for {len(by_layer)} layers")


if __name__ == "__main__":
    main()
