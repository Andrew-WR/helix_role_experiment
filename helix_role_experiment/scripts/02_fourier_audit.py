from __future__ import annotations

import argparse

import numpy as np

from _common import write_csv
from helix_role_experiment.config import ensure_output_dirs, load_config, seed_everything
from helix_role_experiment.fourier import (
    exact_whiten_2d,
    isolate_harmonic,
    null_audit,
    preprocessing_sensitivity,
    tautology_audit,
    unwrap_with_orientation,
)
from helix_role_experiment.plotting import heatmap_svg, scatter_svg
from helix_role_experiment.subspaces import (
    plane_similarity_matrix,
    trace_harmonic_plane,
)
from helix_role_experiment.traces import TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Fourier tautology, nulls, and stability")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 101)
    store = TraceStore(paths["traces"])
    tautology_rows = []
    null_rows = []
    sensitivity_rows = []
    phase_x, phase_y, phase_groups = [], [], []
    planes_by_layer: dict[int, list[np.ndarray]] = {}
    labels_by_layer: dict[int, list[str]] = {}
    draws = int(config["audit"]["null_draws"])
    for row, activations in store.iter_traces():
        if len(activations) < int(config["analysis"]["minimum_trace_length"]):
            tautology_rows.append(
                {
                    "request_id": row["request_id"],
                    "problem_id": row["problem_id"],
                    "layer": row["layer"],
                    "exclusion_reason": "trace_too_short",
                }
            )
            continue
        try:
            audit = tautology_audit(activations)
            plane, singular = trace_harmonic_plane(activations)
        except ValueError as exc:
            tautology_rows.append(
                {
                    "request_id": row["request_id"],
                    "problem_id": row["problem_id"],
                    "layer": row["layer"],
                    "exclusion_reason": str(exc),
                }
            )
            continue
        tautology_rows.append(
            {
                "request_id": row["request_id"],
                "problem_id": row["problem_id"],
                "family": row["task_family"],
                "split": row["split"],
                "layer": row["layer"],
                "length": len(activations),
                "singular_1": float(singular[0]),
                "singular_2": float(singular[1]),
                "exclusion_reason": None,
                **audit,
            }
        )
        projected, _, _ = isolate_harmonic(activations)
        u, singular_full, _ = np.linalg.svd(projected, full_matrices=False)
        coords, _ = exact_whiten_2d(u[:, :2] * singular_full[:2])
        phase = unwrap_with_orientation(np.arctan2(coords[:, 1], coords[:, 0]))
        phase -= phase[0]
        target = 2.0 * np.pi * np.arange(len(phase)) / len(phase)
        phase_x.extend(target.tolist())
        phase_y.extend(phase.tolist())
        phase_groups.extend([f"L{row['layer']}"] * len(phase))
        for audit_row in null_audit(activations, draws, rng):
            null_rows.append(
                {
                    "request_id": row["request_id"],
                    "problem_id": row["problem_id"],
                    "family": row["task_family"],
                    "layer": row["layer"],
                    **audit_row,
                }
            )
        for sensitivity in preprocessing_sensitivity(activations):
            sensitivity_rows.append(
                {
                    "request_id": row["request_id"],
                    "problem_id": row["problem_id"],
                    "layer": row["layer"],
                    "endpoint_distance": float(np.linalg.norm(activations[-1] - activations[0])),
                    **sensitivity,
                }
            )
        layer = int(row["layer"])
        planes_by_layer.setdefault(layer, []).append(plane)
        labels_by_layer.setdefault(layer, []).append(row["problem_id"])

    write_csv(paths["tables"] / "tautology_audit.csv", tautology_rows)
    write_csv(paths["tables"] / "spectral_null_audit.csv", null_rows)
    write_csv(paths["tables"] / "endpoint_preprocessing_sensitivity.csv", sensitivity_rows)
    scatter_svg(
        paths["figures"] / "01_tautological_phase_vs_normalized_position.svg",
        np.asarray(phase_x),
        np.asarray(phase_y),
        phase_groups,
        "Per-trace whitened k=1 phase (diagnostic, not evidence)",
        "imposed normalized phase 2πt/L",
        "recovered unwrapped phase",
    )
    if null_rows:
        scatter_svg(
            paths["figures"] / "02_spectral_concentration_nulls.svg",
            np.arange(len(null_rows)),
            np.asarray([float(row["concentration"]) for row in null_rows]),
            [str(row["null"]) for row in null_rows],
            "Observed and null first-harmonic concentration",
            "trace/null draw",
            "E1 / E(non-DC)",
        )
    similarity_rows = []
    for layer, planes in planes_by_layer.items():
        matrix = plane_similarity_matrix(planes)
        np.save(paths["tables"] / f"projector_similarity_layer_{layer}.npy", matrix)
        for i in range(len(planes)):
            for j in range(len(planes)):
                similarity_rows.append(
                    {
                        "layer": layer,
                        "problem_i": labels_by_layer[layer][i],
                        "problem_j": labels_by_layer[layer][j],
                        "projector_similarity": float(matrix[i, j]),
                    }
                )
        heatmap_svg(
            paths["figures"] / f"03_projector_similarity_layer_{layer}.svg",
            matrix,
            labels_by_layer[layer],
            labels_by_layer[layer],
            f"Candidate-plane projector similarity, layer {layer}",
        )
    write_csv(paths["tables"] / "projector_similarity.csv", similarity_rows)
    print(f"Wrote Fourier audit for {len(tautology_rows)} traces")


if __name__ == "__main__":
    main()

