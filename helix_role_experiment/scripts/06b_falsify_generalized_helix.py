from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import parse_layer_spec, read_csv, write_csv
from helix_role_experiment.config import (
    deterministic_id,
    ensure_output_dirs,
    load_config,
    read_jsonl,
    seed_everything,
)
from helix_role_experiment.controlled_tasks import generate_suite
from helix_role_experiment.models import huggingface_collector_from_config


EPS = 1e-12
TRAJECTORY_CONDITIONS = {
    "concise",
    "verbose_paraphrase",
    "redundant_valid",
    "repeated_summary",
    "confirmation",
    "plausible_digression",
    "length_matched_progress",
}
OPERATION_COLUMNS = (
    "operation_planning",
    "operation_calculation",
    "operation_uncertainty",
    "operation_backtracking",
    "operation_checking",
    "operation_consolidation",
    "operation_final_emission",
)
PHASE_SPANS = (0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi)
RADIUS_SLOPES = (-0.75, -0.375, 0.0, 0.375, 0.75)


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def problem_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(np.asarray(groups).tolist())
    return np.asarray([1.0 / counts[value] for value in groups], dtype=np.float64)


@dataclass
class StandardizedRidge:
    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        x = np.asarray(design, dtype=np.float64).copy()
        for column in range(x.shape[1]):
            x[~np.isfinite(x[:, column]), column] = self.mean[column]
        standardized = (x - self.mean) / self.scale
        return self.intercept + standardized @ self.coefficients


def fit_standardized_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    ridge: float,
) -> StandardizedRidge:
    x = np.asarray(design, dtype=np.float64).copy()
    y = np.asarray(targets, dtype=np.float64)
    mean = np.zeros(x.shape[1], dtype=np.float64)
    for column in range(x.shape[1]):
        finite = np.isfinite(x[:, column])
        mean[column] = np.median(x[finite, column]) if finite.any() else 0.0
        x[~finite, column] = mean[column]
    scale = x.std(axis=0)
    scale[scale < EPS] = 1.0
    standardized = (x - mean) / scale
    full = np.column_stack((np.ones(len(x)), standardized))
    weights = problem_weights(groups)
    root = np.sqrt(weights)[:, None]
    weighted_x = full * root
    weighted_y = y * root
    penalty = float(ridge) * np.eye(full.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        weighted_x.T @ weighted_x + penalty,
        weighted_x.T @ weighted_y,
    )
    return StandardizedRidge(mean, scale, beta[0], beta[1:])


@dataclass
class RawRidge:
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(design, dtype=np.float64) @ self.coefficients


def fit_raw_ridge(
    design: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    ridge: float,
) -> RawRidge:
    x = np.column_stack(
        (np.ones(len(design)), np.asarray(design, dtype=np.float64))
    )
    y = np.asarray(targets, dtype=np.float64)
    weights = problem_weights(groups)
    root = np.sqrt(weights)[:, None]
    weighted_x = x * root
    weighted_y = y * root
    penalty = float(ridge) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        weighted_x.T @ weighted_x + penalty,
        weighted_x.T @ weighted_y,
    )
    return RawRidge(beta[0], beta[1:])


def nuisance_design(
    rows: list[dict[str, str]],
    actual_token_counts: np.ndarray,
) -> np.ndarray:
    token = np.asarray(actual_token_counts, dtype=np.float64)
    token /= max(float(np.median(token)), 1.0)
    confidence = np.asarray([numeric(row, "confidence") for row in rows])
    eos = np.asarray([numeric(row, "eos_logit") for row in rows])
    sentence = np.asarray(
        [numeric(row, "sentence_count_proxy") for row in rows]
    )
    return np.column_stack(
        (
            token,
            token**2,
            token**3,
            sentence,
            confidence,
            confidence**2,
            confidence**3,
            eos,
            eos**2,
            np.asarray(
                [numeric(row, "termination_allowed") for row in rows]
            ),
            *[
                np.asarray([numeric(row, column) for row in rows])
                for column in OPERATION_COLUMNS
            ],
        )
    )


def center_within_problem_trajectory(
    activations: np.ndarray,
    rows: list[dict[str, str]],
) -> np.ndarray:
    output = np.asarray(activations, dtype=np.float32).copy()
    problems = np.asarray([row["problem_id"] for row in rows])
    ordinary = np.asarray(
        [row["condition"] in TRAJECTORY_CONDITIONS for row in rows],
        dtype=bool,
    )
    for problem in sorted(set(problems.tolist())):
        problem_mask = problems == problem
        reference = problem_mask & ordinary
        if not reference.any():
            reference = problem_mask
        output[problem_mask] -= output[reference].mean(
            axis=0,
            dtype=np.float64,
        ).astype(output.dtype)
    return output


def rotation_basis(
    progress: np.ndarray,
    omega: float,
    radius_slope: float,
) -> np.ndarray:
    s = np.asarray(progress, dtype=np.float64)
    radius = 1.0 + float(radius_slope) * (s - 0.5)
    if np.min(radius) <= 0:
        raise ValueError("radius function must remain positive")
    return np.column_stack(
        (
            radius * np.cos(float(omega) * s),
            radius * np.sin(float(omega) * s),
        )
    )


def closed_k1_basis(progress: np.ndarray) -> np.ndarray:
    s = np.asarray(progress, dtype=np.float64)
    return np.column_stack(
        (np.cos(2.0 * np.pi * s), np.sin(2.0 * np.pi * s))
    )


def equal_problem_mse(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> float:
    values = []
    group_array = np.asarray(groups)
    for problem in sorted(set(group_array.tolist())):
        mask = group_array == problem
        values.append(
            float(np.mean(np.square(targets[mask] - predictions[mask])))
        )
    return float(np.mean(values))


def select_rotation_hyperparameters(
    residual_after_axis: np.ndarray,
    progress: np.ndarray,
    groups: np.ndarray,
    families: np.ndarray,
    ridge: float,
) -> tuple[dict[str, tuple[float, float]], list[dict]]:
    selected: dict[str, tuple[float, float]] = {}
    rows = []
    for family in sorted(set(families.tolist())):
        family_mask = families == family
        family_groups = groups[family_mask]
        unique_problems = sorted(set(family_groups.tolist()))
        candidates = []
        for omega in PHASE_SPANS:
            for radius_slope in RADIUS_SLOPES:
                design = rotation_basis(
                    progress[family_mask],
                    omega,
                    radius_slope,
                )
                fold_errors = []
                if len(unique_problems) >= 2:
                    for problem in unique_problems:
                        test = family_groups == problem
                        train = ~test
                        model = fit_raw_ridge(
                            design[train],
                            residual_after_axis[family_mask][train],
                            family_groups[train],
                            ridge,
                        )
                        fold_errors.append(
                            float(
                                np.mean(
                                    np.square(
                                        residual_after_axis[family_mask][test]
                                        - model.predict(design[test])
                                    )
                                )
                            )
                        )
                else:
                    model = fit_raw_ridge(
                        design,
                        residual_after_axis[family_mask],
                        family_groups,
                        ridge,
                    )
                    fold_errors.append(
                        float(
                            np.mean(
                                np.square(
                                    residual_after_axis[family_mask]
                                    - model.predict(design)
                                )
                            )
                        )
                    )
                candidates.append(
                    {
                        "family": family,
                        "omega_radians": float(omega),
                        "turns": float(omega / (2.0 * np.pi)),
                        "radius_slope": float(radius_slope),
                        "selection_mse": float(np.mean(fold_errors)),
                        "selection": "leave_one_problem_out"
                        if len(unique_problems) >= 2
                        else "in_sample_fallback",
                    }
                )
        winner = min(candidates, key=lambda row: row["selection_mse"])
        selected[family] = (
            float(winner["omega_radians"]),
            float(winner["radius_slope"]),
        )
        for row in candidates:
            row["selected"] = bool(row is winner)
            rows.append(row)
    return selected, rows


@dataclass
class HelixGeometry:
    nuisance: StandardizedRidge
    axial: RawRidge
    closed_shared: RawRidge
    closed_by_family: dict[str, RawRidge]
    rotation_by_family: dict[str, RawRidge]
    hyperparameters: dict[str, tuple[float, float]]

    @property
    def axial_direction(self) -> np.ndarray:
        return self.axial.coefficients[0]

    def axial_value(self, progress: float) -> np.ndarray:
        return self.axial.predict(np.asarray([[progress]]))[0]

    def closed_shared_value(self, progress: float) -> np.ndarray:
        return self.closed_shared.predict(
            closed_k1_basis(np.asarray([progress]))
        )[0]

    def closed_family_value(self, family: str, progress: float) -> np.ndarray:
        return self.closed_by_family[family].predict(
            closed_k1_basis(np.asarray([progress]))
        )[0]

    def rotation_value(self, family: str, progress: float) -> np.ndarray:
        omega, radius_slope = self.hyperparameters[family]
        return self.rotation_by_family[family].predict(
            rotation_basis(
                np.asarray([progress]),
                omega,
                radius_slope,
            )
        )[0]

    def helix_value(self, family: str, progress: float) -> np.ndarray:
        return self.axial_value(progress) + self.rotation_value(
            family,
            progress,
        )


def fit_geometry(
    rows: list[dict[str, str]],
    centered_activations: np.ndarray,
    actual_token_counts: np.ndarray,
    ridge: float,
    hyperparameters: dict[str, tuple[float, float]] | None = None,
) -> tuple[HelixGeometry, list[dict]]:
    groups = np.asarray([row["problem_id"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    progress = np.asarray(
        [numeric(row, "structural_progress") for row in rows],
        dtype=np.float64,
    )
    nuisance = fit_standardized_ridge(
        nuisance_design(rows, actual_token_counts),
        centered_activations,
        groups,
        ridge,
    )
    residual = centered_activations - nuisance.predict(
        nuisance_design(rows, actual_token_counts)
    )
    axial = fit_raw_ridge(progress[:, None], residual, groups, ridge)
    residual_after_axis = residual - axial.predict(progress[:, None])
    selection_rows: list[dict] = []
    if hyperparameters is None:
        hyperparameters, selection_rows = select_rotation_hyperparameters(
            residual_after_axis,
            progress,
            groups,
            families,
            ridge,
        )

    closed_shared = fit_raw_ridge(
        closed_k1_basis(progress),
        residual,
        groups,
        ridge,
    )
    closed_by_family: dict[str, RawRidge] = {}
    rotation_by_family: dict[str, RawRidge] = {}
    axial_vector = axial.coefficients[0]
    axial_unit = axial_vector / max(float(np.linalg.norm(axial_vector)), EPS)
    for family in sorted(set(families.tolist())):
        mask = families == family
        closed_by_family[family] = fit_raw_ridge(
            closed_k1_basis(progress[mask]),
            residual[mask],
            groups[mask],
            ridge,
        )
        omega, radius_slope = hyperparameters[family]
        rotation = fit_raw_ridge(
            rotation_basis(progress[mask], omega, radius_slope),
            residual_after_axis[mask],
            groups[mask],
            ridge,
        )
        # Preserve the axial/rotational decomposition in activation space.
        coefficients = rotation.coefficients.copy()
        coefficients -= (
            coefficients @ axial_unit
        )[:, None] * axial_unit[None, :]
        intercept = rotation.intercept - (
            float(rotation.intercept @ axial_unit) * axial_unit
        )
        rotation_by_family[family] = RawRidge(intercept, coefficients)
    return (
        HelixGeometry(
            nuisance=nuisance,
            axial=axial,
            closed_shared=closed_shared,
            closed_by_family=closed_by_family,
            rotation_by_family=rotation_by_family,
            hyperparameters=hyperparameters,
        ),
        selection_rows,
    )


def model_fit_metrics(
    geometry: HelixGeometry,
    rows: list[dict[str, str]],
    centered_activations: np.ndarray,
    actual_token_counts: np.ndarray,
) -> list[dict]:
    groups = np.asarray([row["problem_id"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    progress = np.asarray(
        [numeric(row, "structural_progress") for row in rows],
        dtype=np.float64,
    )
    nuisance_prediction = geometry.nuisance.predict(
        nuisance_design(rows, actual_token_counts)
    )
    residual = centered_activations - nuisance_prediction
    predictions = {
        "linear_axial": geometry.axial.predict(progress[:, None]),
        "closed_k1_shared": geometry.closed_shared.predict(
            closed_k1_basis(progress)
        ),
        "closed_k1_by_family": np.vstack(
            [
                geometry.closed_family_value(family, value)
                for family, value in zip(families, progress)
            ]
        ),
        "generalized_helix": np.vstack(
            [
                geometry.helix_value(family, value)
                for family, value in zip(families, progress)
            ]
        ),
    }
    baseline = equal_problem_mse(
        residual,
        np.zeros_like(residual),
        groups,
    )
    output = []
    for name, prediction in predictions.items():
        mse = equal_problem_mse(residual, prediction, groups)
        output.append(
            {
                "model": name,
                "evaluation": "in_sample_descriptive",
                "problem_count": len(set(groups.tolist())),
                "family_count": len(set(families.tolist())),
                "nuisance_residual_mse": mse,
                "in_sample_r2_after_nuisance": 1.0
                - mse / max(baseline, EPS),
            }
        )
    return output


def heldout_model_fit_metrics(
    geometry_without_problem: dict[str, HelixGeometry],
    rows: list[dict[str, str]],
    centered_activations: np.ndarray,
    actual_token_counts: np.ndarray,
) -> list[dict]:
    groups = np.asarray([row["problem_id"] for row in rows])
    families = np.asarray([row["family"] for row in rows])
    progress = np.asarray(
        [numeric(row, "structural_progress") for row in rows],
        dtype=np.float64,
    )
    designs = nuisance_design(rows, actual_token_counts)
    nuisance_errors: list[float] = []
    errors: dict[str, list[float]] = defaultdict(list)
    for problem in sorted(set(groups.tolist())):
        mask = groups == problem
        geometry = geometry_without_problem[problem]
        nuisance_prediction = geometry.nuisance.predict(designs[mask])
        target = centered_activations[mask]
        residual = target - nuisance_prediction
        nuisance_errors.append(float(np.mean(np.square(residual))))
        local_progress = progress[mask]
        local_families = families[mask]
        predictions = {
            "linear_axial": geometry.axial.predict(local_progress[:, None]),
            "closed_k1_shared": geometry.closed_shared.predict(
                closed_k1_basis(local_progress)
            ),
            "closed_k1_by_family": np.vstack(
                [
                    geometry.closed_family_value(family, value)
                    for family, value in zip(local_families, local_progress)
                ]
            ),
            "generalized_helix": np.vstack(
                [
                    geometry.helix_value(family, value)
                    for family, value in zip(local_families, local_progress)
                ]
            ),
        }
        for name, prediction in predictions.items():
            errors[name].append(
                float(np.mean(np.square(residual - prediction)))
            )
    nuisance_mse = float(np.mean(nuisance_errors))
    output = []
    for name, values in sorted(errors.items()):
        mse = float(np.mean(values))
        output.append(
            {
                "model": name,
                "evaluation": "leave_one_problem_out_nested_refit",
                "problem_count": len(set(groups.tolist())),
                "family_count": len(set(families.tolist())),
                "nuisance_residual_mse": mse,
                "heldout_incremental_r2_vs_nuisance": (
                    nuisance_mse - mse
                )
                / max(nuisance_mse, EPS),
            }
        )
    return output


def build_endpoint_pairs(
    rows: list[dict[str, str]],
    pairs_per_family: int,
) -> list[dict]:
    concise = [row for row in rows if row["condition"] == "concise"]
    by_family_problem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in concise:
        by_family_problem[(row["family"], row["problem_id"])].append(row)
    family_candidates: dict[str, list[dict]] = defaultdict(list)
    for (family, problem), values in sorted(by_family_problem.items()):
        ordered = sorted(values, key=lambda row: numeric(row, "structural_progress"))
        if len(ordered) < 2:
            continue
        forward = {
            "family": family,
            "problem_id": problem,
            "direction": "forward",
            "target": ordered[0],
            "desired": ordered[-1],
        }
        backward = {
            "family": family,
            "problem_id": problem,
            "direction": "backward",
            "target": ordered[-1],
            "desired": ordered[0],
        }
        family_candidates[family].extend((forward, backward))
    output = []
    for family, candidates in sorted(family_candidates.items()):
        # Alternate problems and directions before adding additional pairs.
        ordered = sorted(
            candidates,
            key=lambda row: (
                0 if row["direction"] == "forward" else 1,
                row["problem_id"],
            ),
        )
        chosen = []
        problems = sorted({row["problem_id"] for row in ordered})
        for index, problem in enumerate(problems):
            direction = "forward" if index % 2 == 0 else "backward"
            chosen.extend(
                row
                for row in ordered
                if row["problem_id"] == problem and row["direction"] == direction
            )
        for row in ordered:
            if row not in chosen:
                chosen.append(row)
        output.extend(chosen[: int(pairs_per_family)])
    for row in output:
        row["pair_id"] = deterministic_id(
            row["problem_id"],
            row["target"]["variant_id"],
            row["desired"]["variant_id"],
            "generalized_helix_falsification",
        )
    return output


def match_norm(vector: np.ndarray, target_norm: float) -> tuple[np.ndarray, bool]:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= EPS or target_norm <= EPS:
        return np.zeros_like(value), False
    return value * (float(target_norm) / norm), True


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return (
        float(np.dot(left, right) / denominator)
        if denominator > EPS
        else float("nan")
    )


def random_orthogonal_delta(
    dimension: int,
    model_directions: list[np.ndarray],
    target_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.column_stack(
        [
            direction
            for direction in model_directions
            if np.linalg.norm(direction) > EPS
        ]
    )
    random = rng.normal(size=dimension)
    if matrix.size:
        q, _ = np.linalg.qr(matrix)
        random -= q @ (q.T @ random)
    return match_norm(random, target_norm)[0]


def candidate_distribution_js(left: dict, right: dict) -> float:
    keys = sorted(
        set(left["candidate_probabilities"])
        | set(right["candidate_probabilities"])
    )
    p = np.asarray([left["candidate_probabilities"].get(key, 0.0) for key in keys])
    q = np.asarray([right["candidate_probabilities"].get(key, 0.0) for key in keys])
    p /= max(float(p.sum()), EPS)
    q /= max(float(q.sum()), EPS)
    midpoint = 0.5 * (p + q)
    mask_p, mask_q = p > 0, q > 0
    return float(
        0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / midpoint[mask_p]))
        + 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / midpoint[mask_q]))
    )


def constant_delta_callback(torch, delta: np.ndarray):
    def callback(_layer, _step, hidden):
        return hidden + torch.as_tensor(
            delta,
            device=hidden.device,
            dtype=hidden.dtype,
        )

    return callback


def summarize_outcomes(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), row["control"])].append(row)
    output = []
    for (layer, control), values in sorted(grouped.items()):
        effects = np.asarray(
            [float(row["valid_transition_probability_change"]) for row in values]
        )
        output.append(
            {
                "layer": layer,
                "control": control,
                "pair_count": len(values),
                "problem_count": len({row["problem_id"] for row in values}),
                "mean_probability_change": float(np.mean(effects)),
                "median_probability_change": float(np.median(effects)),
                "fraction_positive": float(np.mean(effects > 0)),
                "mean_eos_logit_change": float(
                    np.mean([float(row["eos_logit_change"]) for row in values])
                ),
                "mean_downstream_js": float(
                    np.mean([float(row["downstream_js"]) for row in values])
                ),
                "mean_intervention_norm": float(
                    np.mean([float(row["intervention_norm"]) for row in values])
                ),
            }
        )
    return output


def paired_gate(
    rows: list[dict],
    layer: int,
    gate: str,
    treatment: str,
    comparator: str,
    prediction: str,
) -> dict:
    values = [row for row in rows if int(row["layer"]) == int(layer)]
    lookup = {
        (row["pair_id"], row["control"]): float(
            row["valid_transition_probability_change"]
        )
        for row in values
    }
    pair_ids = sorted(
        {
            row["pair_id"]
            for row in values
            if (row["pair_id"], treatment) in lookup
            and (row["pair_id"], comparator) in lookup
        }
    )
    differences = np.asarray(
        [
            lookup[(pair_id, treatment)] - lookup[(pair_id, comparator)]
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    median = float(np.median(differences)) if len(differences) else float("nan")
    return {
        "layer": layer,
        "gate": gate,
        "treatment": treatment,
        "comparator": comparator,
        "prediction": prediction,
        "pair_count": len(pair_ids),
        "mean_paired_difference": float(np.mean(differences))
        if len(differences)
        else float("nan"),
        "median_paired_difference": median,
        "fraction_prediction_satisfied": float(np.mean(differences > 0))
        if len(differences)
        else float("nan"),
        "smoke_status": (
            "survived_directional_gate"
            if np.isfinite(median) and median > 0
            else "failed_directional_gate"
        ),
    }


def falsification_gates(rows: list[dict], layers: list[int]) -> list[dict]:
    gates = []
    specifications = (
        (
            "beats_norm_matched_random",
            "helix_full",
            "orthogonal_random",
            "Full helix must outperform an equal-norm off-model perturbation.",
        ),
        (
            "directionality",
            "helix_full",
            "reverse_helix",
            "The intended state displacement must outperform its exact reverse.",
        ),
        (
            "beyond_linear",
            "helix_full",
            "linear_axial_matched",
            "The rotational component must add causal value beyond the axial direction.",
        ),
        (
            "beyond_closed_k1",
            "helix_full",
            "closed_k1_family_matched",
            "The open helix must outperform the strongest family-specific closed k=1 comparator.",
        ),
        (
            "family_specific_rotation",
            "helix_full",
            "wrong_family_helix",
            "The correct family rotation must outperform a wrong-family rotation.",
        ),
        (
            "axial_component_needed",
            "helix_full",
            "rotation_only_matched",
            "Removing the axial component must reduce the causal effect.",
        ),
    )
    for layer in layers:
        for gate, treatment, comparator, prediction in specifications:
            gates.append(
                paired_gate(
                    rows,
                    layer,
                    gate,
                    treatment,
                    comparator,
                    prediction,
                )
            )

        values = [row for row in rows if int(row["layer"]) == int(layer)]
        lookup = {
            (row["pair_id"], row["control"]): float(
                row["valid_transition_probability_change"]
            )
            for row in values
        }
        pair_ids = sorted({row["pair_id"] for row in values})
        dose_success = []
        eos_retentions = []
        for pair_id in pair_ids:
            dose_keys = (
                (pair_id, "helix_half_dose"),
                (pair_id, "helix_full"),
                (pair_id, "helix_1_5_dose"),
            )
            if all(key in lookup for key in dose_keys):
                half, full, one_half = [lookup[key] for key in dose_keys]
                dose_success.append(half <= full <= one_half)
            full_key = (pair_id, "helix_full")
            eos_key = (pair_id, "eos_orthogonal_helix")
            if full_key in lookup and eos_key in lookup:
                full = lookup[full_key]
                eos = lookup[eos_key]
                if full > 0:
                    eos_retentions.append(eos / max(full, EPS))
                else:
                    eos_retentions.append(float("-inf"))
        dose_fraction = float(np.mean(dose_success)) if dose_success else float("nan")
        gates.append(
            {
                "layer": layer,
                "gate": "dose_response",
                "treatment": "helix_half/full/1_5_dose",
                "comparator": "ordered doses",
                "prediction": "Effect should increase monotonically from 0.5x to 1.0x to 1.5x.",
                "pair_count": len(dose_success),
                "fraction_prediction_satisfied": dose_fraction,
                "smoke_status": (
                    "survived_directional_gate"
                    if np.isfinite(dose_fraction) and dose_fraction > 0.5
                    else "failed_directional_gate"
                ),
            }
        )
        median_retention = (
            float(np.median(eos_retentions)) if eos_retentions else float("nan")
        )
        gates.append(
            {
                "layer": layer,
                "gate": "eos_independence",
                "treatment": "eos_orthogonal_helix",
                "comparator": "helix_full",
                "prediction": "At least half of a positive full-helix effect should survive EOS orthogonalization.",
                "pair_count": len(eos_retentions),
                "median_effect_retention": median_retention,
                "fraction_prediction_satisfied": float(
                    np.mean(np.asarray(eos_retentions) >= 0.5)
                )
                if eos_retentions
                else float("nan"),
                "smoke_status": (
                    "survived_directional_gate"
                    if np.isfinite(median_retention) and median_retention >= 0.5
                    else "failed_directional_gate"
                ),
            }
        )
        by_direction = {}
        for direction in ("forward", "backward"):
            effects = [
                float(row["valid_transition_probability_change"])
                for row in values
                if row["control"] == "helix_full"
                and row["direction"] == direction
            ]
            by_direction[direction] = (
                float(np.mean(effects)) if effects else float("nan")
            )
        both_positive = all(
            np.isfinite(value) and value > 0 for value in by_direction.values()
        )
        gates.append(
            {
                "layer": layer,
                "gate": "bidirectional_state_control",
                "treatment": "helix_full",
                "comparator": "forward and backward endpoint transfers",
                "prediction": "The same geometry must support both advancement and rollback.",
                "pair_count": len(pair_ids),
                "mean_forward_effect": by_direction["forward"],
                "mean_backward_effect": by_direction["backward"],
                "fraction_prediction_satisfied": float(both_positive),
                "smoke_status": (
                    "survived_directional_gate"
                    if both_positive
                    else "failed_directional_gate"
                ),
            }
        )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and causally falsify a minimal generalized helix against "
            "linear and closed-k1 alternatives"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--layers",
        required=True,
        help="A small locked layer set, for example 51,55,59",
    )
    parser.add_argument(
        "--pairs-per-family",
        type=int,
        default=2,
        help="Endpoint-transfer pairs per family and layer (default: 2)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 661)
    observational = read_csv(paths["tables"] / "observational_cross.csv")
    available_layers = sorted({int(row["layer"]) for row in observational})
    layers = parse_layer_spec(args.layers, available_layers)
    print(f"Generalized-helix falsification layers: {layers}", flush=True)
    prefixes = {
        row["variant_id"]: row
        for row in read_jsonl(paths["root"] / "counterfactual_prefixes.jsonl")
    }
    token_path = paths["tables"] / "counterfactual_actual_token_counts.csv"
    if not token_path.is_file():
        raise FileNotFoundError(
            f"{token_path} is missing; run 05b without --skip-tokenizer first"
        )
    token_lookup = {
        row["variant_id"]: int(row["actual_formatted_token_count"])
        for row in read_csv(token_path)
    }
    with np.load(
        paths["tables"] / "counterfactual_activations.npz",
        allow_pickle=False,
    ) as data:
        activations = np.asarray(data["activations"])
        variant_ids = data["variant_ids"].astype(str)
        activation_layers = data["layers"].astype(int)
    if len(observational) != len(activations):
        raise ValueError("observational CSV and activation NPZ have different lengths")
    for index, row in enumerate(observational):
        if (
            row["variant_id"] != variant_ids[index]
            or int(row["layer"]) != int(activation_layers[index])
        ):
            raise ValueError(
                "observational CSV and activation NPZ are not identically ordered"
            )

    problems = {
        problem.problem_id: problem
        for problem in generate_suite(
            int(config["tasks"]["problems_per_family"]),
            int(config["study"]["seed"]),
        )
    }
    state_lookup = {
        problem_id: {state.state_id: state for state in problem.states}
        for problem_id, problem in problems.items()
    }
    first_layer_rows = [
        row for row in observational if int(row["layer"]) == layers[0]
    ]
    pairs = build_endpoint_pairs(first_layer_rows, args.pairs_per_family)
    if not pairs:
        raise ValueError("no concise endpoint-transfer pairs could be built")
    print(
        f"Built {len(pairs)} endpoint pairs "
        f"({Counter(pair['direction'] for pair in pairs)})",
        flush=True,
    )

    backend = huggingface_collector_from_config(
        config["model"],
        config["collection"],
    )
    torch = backend.torch
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
        .astype(np.float64)
    )
    eos_direction /= max(float(np.linalg.norm(eos_direction)), EPS)

    outcomes: list[dict] = []
    parameter_rows: list[dict] = []
    fit_rows: list[dict] = []
    ridge = float(config["analysis"].get("ridge", 0.001))
    for layer_index, layer in enumerate(layers, 1):
        indices = np.flatnonzero(activation_layers == layer)
        layer_rows = [observational[index] for index in indices]
        layer_activations = np.asarray(activations[indices])
        centered = center_within_problem_trajectory(
            layer_activations,
            layer_rows,
        )
        ordinary = np.asarray(
            [row["condition"] in TRAJECTORY_CONDITIONS for row in layer_rows],
            dtype=bool,
        )
        ordinary_rows = [
            row for row, keep in zip(layer_rows, ordinary) if keep
        ]
        ordinary_centered = centered[ordinary]
        ordinary_tokens = np.asarray(
            [token_lookup[row["variant_id"]] for row in ordinary_rows],
            dtype=np.float64,
        )
        full_geometry, selection = fit_geometry(
            ordinary_rows,
            ordinary_centered,
            ordinary_tokens,
            ridge,
        )
        for row in selection:
            parameter_rows.append(
                {
                    "layer": layer,
                    "fit_scope": "all_problems",
                    "excluded_problem": None,
                    **row,
                }
            )
        row_by_variant = {
            row["variant_id"]: (row, layer_activations[index])
            for index, row in enumerate(layer_rows)
        }
        geometry_without_problem: dict[str, HelixGeometry] = {}
        for problem_id in sorted(
            {row["problem_id"] for row in ordinary_rows}
        ):
            keep = np.asarray(
                [row["problem_id"] != problem_id for row in ordinary_rows],
                dtype=bool,
            )
            heldout_geometry, heldout_selection = fit_geometry(
                [row for row, value in zip(ordinary_rows, keep) if value],
                ordinary_centered[keep],
                ordinary_tokens[keep],
                ridge,
            )
            geometry_without_problem[problem_id] = heldout_geometry
            for row in heldout_selection:
                parameter_rows.append(
                    {
                        "layer": layer,
                        "fit_scope": "target_problem_excluded",
                        "excluded_problem": problem_id,
                        **row,
                    }
                )
        for row in model_fit_metrics(
            full_geometry,
            ordinary_rows,
            ordinary_centered,
            ordinary_tokens,
        ):
            fit_rows.append({"layer": layer, **row})
        for row in heldout_model_fit_metrics(
            geometry_without_problem,
            ordinary_rows,
            ordinary_centered,
            ordinary_tokens,
        ):
            fit_rows.append({"layer": layer, **row})

        for pair_index, pair in enumerate(pairs, 1):
            target_row, target_activation = row_by_variant[
                pair["target"]["variant_id"]
            ]
            desired_row = pair["desired"]
            target_progress = numeric(target_row, "structural_progress")
            desired_progress = numeric(desired_row, "structural_progress")
            family = target_row["family"]
            geometry = geometry_without_problem[pair["problem_id"]]
            axial_delta = (
                geometry.axial_value(desired_progress)
                - geometry.axial_value(target_progress)
            )
            rotation_delta = (
                geometry.rotation_value(family, desired_progress)
                - geometry.rotation_value(family, target_progress)
            )
            full_delta = axial_delta + rotation_delta
            full_norm = float(np.linalg.norm(full_delta))
            if full_norm <= EPS:
                print(
                    f"WARNING: collapsed helix delta for {pair['pair_id']} "
                    f"at layer {layer}",
                    flush=True,
                )
                continue
            closed_shared = (
                geometry.closed_shared_value(desired_progress)
                - geometry.closed_shared_value(target_progress)
            )
            closed_family = (
                geometry.closed_family_value(family, desired_progress)
                - geometry.closed_family_value(family, target_progress)
            )
            other_family = next(
                value
                for value in sorted(geometry.rotation_by_family)
                if value != family
            )
            wrong_family = axial_delta + (
                geometry.rotation_value(other_family, desired_progress)
                - geometry.rotation_value(other_family, target_progress)
            )
            linear_matched, linear_match = match_norm(axial_delta, full_norm)
            rotation_matched, rotation_match = match_norm(
                rotation_delta,
                full_norm,
            )
            closed_shared_matched, closed_shared_match = match_norm(
                closed_shared,
                full_norm,
            )
            closed_family_matched, closed_family_match = match_norm(
                closed_family,
                full_norm,
            )
            wrong_family_matched, wrong_family_match = match_norm(
                wrong_family,
                full_norm,
            )
            eos_orthogonal = full_delta - (
                float(full_delta @ eos_direction) * eos_direction
            )
            eos_orthogonal, eos_match = match_norm(
                eos_orthogonal,
                full_norm,
            )
            model_directions = [
                geometry.axial_direction,
                *[
                    coefficient
                    for model in geometry.rotation_by_family.values()
                    for coefficient in model.coefficients
                ],
            ]
            orthogonal_random = random_orthogonal_delta(
                len(full_delta),
                model_directions,
                full_norm,
                rng,
            )
            interventions = {
                "helix_half_dose": (0.5 * full_delta, True, 0.5),
                "helix_full": (full_delta, True, 1.0),
                "helix_1_5_dose": (1.5 * full_delta, True, 1.5),
                "linear_axial_matched": (linear_matched, linear_match, 1.0),
                "rotation_only_matched": (
                    rotation_matched,
                    rotation_match,
                    1.0,
                ),
                "closed_k1_shared_matched": (
                    closed_shared_matched,
                    closed_shared_match,
                    1.0,
                ),
                "closed_k1_family_matched": (
                    closed_family_matched,
                    closed_family_match,
                    1.0,
                ),
                "wrong_family_helix": (
                    wrong_family_matched,
                    wrong_family_match,
                    1.0,
                ),
                "reverse_helix": (-full_delta, True, -1.0),
                "orthogonal_random": (orthogonal_random, True, 1.0),
                "eos_orthogonal_helix": (
                    eos_orthogonal,
                    eos_match,
                    1.0,
                ),
            }
            desired_prefix = prefixes[desired_row["variant_id"]]
            desired_state = state_lookup[pair["problem_id"]][
                desired_prefix["state_id"]
            ]
            candidate_transitions = list(desired_state.valid_next)
            target_text = prefixes[target_row["variant_id"]]["text"]
            baseline = backend.score_first_transition(
                target_text,
                candidate_transitions,
                layer,
                intervention=None,
            )
            for control, (delta, norm_match_possible, dose) in interventions.items():
                result = backend.score_first_transition(
                    target_text,
                    candidate_transitions,
                    layer,
                    intervention=constant_delta_callback(torch, delta),
                )
                changed = np.asarray(target_activation, dtype=np.float64) + delta
                outcomes.append(
                    {
                        "layer": layer,
                        "pair_id": pair["pair_id"],
                        "pair_index": pair_index,
                        "problem_id": pair["problem_id"],
                        "family": family,
                        "direction": pair["direction"],
                        "target_progress": target_progress,
                        "desired_progress": desired_progress,
                        "progress_shift": desired_progress - target_progress,
                        "target_variant_id": target_row["variant_id"],
                        "desired_variant_id": desired_row["variant_id"],
                        "geometry_excluded_problem": pair["problem_id"],
                        "control": control,
                        "dose": dose,
                        "wrong_rotation_family": (
                            other_family
                            if control == "wrong_family_helix"
                            else None
                        ),
                        "norm_match_possible": norm_match_possible,
                        "candidate_transition_count": len(candidate_transitions),
                        "valid_transition_probability_baseline": baseline[
                            "valid_next_state_probability"
                        ],
                        "valid_transition_probability_after": result[
                            "valid_next_state_probability"
                        ],
                        "valid_transition_probability_change": result[
                            "valid_next_state_probability"
                        ]
                        - baseline["valid_next_state_probability"],
                        "eos_logit_change": result["eos_logit"]
                        - baseline["eos_logit"],
                        "eos_probability_change": result["eos_probability"]
                        - baseline["eos_probability"],
                        "downstream_js": candidate_distribution_js(
                            baseline,
                            result,
                        ),
                        "intervention_norm": float(np.linalg.norm(delta)),
                        "natural_full_helix_norm": full_norm,
                        "axial_component_norm": float(
                            np.linalg.norm(axial_delta)
                        ),
                        "rotation_component_norm": float(
                            np.linalg.norm(rotation_delta)
                        ),
                        "cosine_to_full_helix": cosine_similarity(
                            delta,
                            full_delta,
                        ),
                        "activation_norm_before": float(
                            np.linalg.norm(target_activation)
                        ),
                        "activation_norm_after": float(np.linalg.norm(changed)),
                        "relative_activation_norm_change": (
                            float(np.linalg.norm(changed))
                            - float(np.linalg.norm(target_activation))
                        )
                        / max(float(np.linalg.norm(target_activation)), EPS),
                    }
                )
            print(
                f"Layer {layer}: scored pair {pair_index}/{len(pairs)} "
                f"({pair['direction']} {pair['problem_id']})",
                flush=True,
            )
        print(
            f"Generalized-helix falsification: completed layer {layer} "
            f"({layer_index}/{len(layers)})",
            flush=True,
        )

    summary = summarize_outcomes(outcomes)
    gates = falsification_gates(outcomes, layers)
    write_csv(
        paths["tables"] / "generalized_helix_causal_outcomes.csv",
        outcomes,
    )
    write_csv(
        paths["tables"] / "generalized_helix_intervention_summary.csv",
        summary,
    )
    write_csv(
        paths["tables"] / "generalized_helix_falsification_gates.csv",
        gates,
    )
    write_csv(
        paths["tables"] / "generalized_helix_geometry_selection.csv",
        parameter_rows,
    )
    write_csv(
        paths["tables"] / "generalized_helix_model_fit.csv",
        fit_rows,
    )
    print(
        f"Wrote {len(outcomes)} causal outcomes and {len(gates)} "
        f"falsification gates to {paths['tables']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
