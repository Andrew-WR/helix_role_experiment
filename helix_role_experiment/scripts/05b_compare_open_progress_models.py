from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import read_csv, write_csv
from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    read_jsonl,
)
from helix_role_experiment.models import resolve_adapter_path
from helix_role_experiment.plotting import line_svg
from helix_role_experiment.traces import TraceStore


EPS = 1e-12
OPERATION_COLUMNS = (
    "operation_planning",
    "operation_calculation",
    "operation_uncertainty",
    "operation_backtracking",
    "operation_checking",
    "operation_consolidation",
    "operation_final_emission",
)
TRAJECTORY_CONDITIONS = {
    "concise",
    "verbose_paraphrase",
    "redundant_valid",
    "repeated_summary",
    "confirmation",
    "plausible_digression",
    "length_matched_progress",
}


def parse_layers(specification: str, available: list[int]) -> list[int]:
    if not available:
        raise ValueError("no layers are available")
    if specification == "all":
        return available
    if specification == "late-half":
        midpoint = (max(available) + 1) // 2
        selected = [layer for layer in available if layer >= midpoint]
    else:
        selected_set: set[int] = set()
        for item in specification.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                selected_set.update(range(int(left), int(right) + 1))
            else:
                selected_set.add(int(item))
        selected = sorted(selected_set)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"requested layers are unavailable: {missing}")
    if not selected:
        raise ValueError("layer selection is empty")
    return selected


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def temporal_models(t: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed open and periodic bases; the intercept is added during fitting."""
    x = np.asarray(t, dtype=np.float64)
    hinge = lambda knot: np.maximum(x - knot, 0.0) ** 3
    return {
        "linear_open": x[:, None],
        "quadratic_open": np.column_stack((x, x**2)),
        "dct_open_rank2": np.column_stack(
            (np.cos(np.pi * x), np.cos(2.0 * np.pi * x))
        ),
        "closed_fourier_k1": np.column_stack(
            (np.cos(2.0 * np.pi * x), np.sin(2.0 * np.pi * x))
        ),
        "cubic_open": np.column_stack((x, x**2, x**3)),
        "closed_k1_plus_drift": np.column_stack(
            (x, np.cos(2.0 * np.pi * x), np.sin(2.0 * np.pi * x))
        ),
        "open_spiral_half_turn": np.column_stack(
            (x, np.cos(np.pi * x), np.sin(np.pi * x))
        ),
        "open_spiral_three_quarter_turn": np.column_stack(
            (x, np.cos(1.5 * np.pi * x), np.sin(1.5 * np.pi * x))
        ),
        "open_spline": np.column_stack(
            (x, x**2, x**3, hinge(0.25), hinge(0.50), hinge(0.75))
        ),
    }


def progress_bases(progress: np.ndarray) -> dict[str, np.ndarray]:
    s = np.asarray(progress, dtype=np.float64)
    hinge = lambda knot: np.maximum(s - knot, 0.0) ** 3
    return {
        "progress_linear": s[:, None],
        "progress_quadratic_open": np.column_stack((s, s**2)),
        "progress_dct_open": np.column_stack(
            (np.cos(np.pi * s), np.cos(2.0 * np.pi * s))
        ),
        "progress_cubic_open": np.column_stack((s, s**2, s**3)),
        "progress_spline_open": np.column_stack(
            (s, s**2, s**3, hinge(0.25), hinge(0.50), hinge(0.75))
        ),
        "progress_closed_k1": np.column_stack(
            (np.cos(2.0 * np.pi * s), np.sin(2.0 * np.pi * s))
        ),
        "progress_spiral_half_turn": np.column_stack(
            (s, np.cos(np.pi * s), np.sin(np.pi * s))
        ),
        "progress_spiral_three_quarter_turn": np.column_stack(
            (s, np.cos(1.5 * np.pi * s), np.sin(1.5 * np.pi * s))
        ),
        "progress_spiral_full_turn": np.column_stack(
            (s, np.cos(2.0 * np.pi * s), np.sin(2.0 * np.pi * s))
        ),
    }


def finite_train_transform(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=np.float64).copy()
    test = np.asarray(test_x, dtype=np.float64).copy()
    medians = np.zeros(train.shape[1], dtype=np.float64)
    for column in range(train.shape[1]):
        finite = np.isfinite(train[:, column])
        medians[column] = np.median(train[finite, column]) if finite.any() else 0.0
        train[~finite, column] = medians[column]
        test[~np.isfinite(test[:, column]), column] = medians[column]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < EPS] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), np.asarray(x, dtype=np.float64)))
    target = np.asarray(y, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(design), dtype=np.float64)
    root = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    weighted_x = design * root
    weighted_y = target * root
    penalty = float(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        weighted_x.T @ weighted_x + penalty,
        weighted_x.T @ weighted_y,
    )


def grouped_cv_multioutput(
    design: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
    ridge: float,
) -> dict[str, float | int]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets)
    group_array = np.asarray(groups)
    fold_array = np.asarray(fold_ids)
    problem_mse: list[float] = []
    problem_baseline_mse: list[float] = []
    evaluated: set[str] = set()
    for fold in sorted(set(int(value) for value in fold_array)):
        test = fold_array == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        train_x, test_x = finite_train_transform(x[train], x[test])
        train_groups = group_array[train]
        counts = Counter(train_groups.tolist())
        weights = np.asarray(
            [1.0 / counts[value] for value in train_groups],
            dtype=np.float64,
        )
        coefficients = fit_weighted_ridge(train_x, y[train], ridge, weights)
        prediction = np.column_stack((np.ones(test.sum()), test_x)) @ coefficients
        train_mean = np.average(y[train], axis=0, weights=weights)
        for group in sorted(set(group_array[test].tolist())):
            local = group_array[test] == group
            residual = y[test][local] - prediction[local]
            baseline = y[test][local] - train_mean
            problem_mse.append(float(np.mean(np.square(residual))))
            problem_baseline_mse.append(float(np.mean(np.square(baseline))))
            evaluated.add(str(group))
    if not problem_mse:
        raise ValueError("no held-out groups were evaluated")
    mse = float(np.mean(problem_mse))
    baseline_mse = float(np.mean(problem_baseline_mse))
    return {
        "cv_mse": mse,
        "baseline_mse": baseline_mse,
        "cv_r2": 1.0 - mse / max(baseline_mse, EPS),
        "heldout_groups": len(evaluated),
        "feature_count": x.shape[1],
    }


def fit_predict_multioutput(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    transformed_train, transformed_test = finite_train_transform(train_x, test_x)
    counts = Counter(np.asarray(train_groups).tolist())
    weights = np.asarray(
        [1.0 / counts[value] for value in train_groups],
        dtype=np.float64,
    )
    coefficients = fit_weighted_ridge(
        transformed_train,
        train_y,
        ridge,
        weights,
    )
    return (
        np.column_stack((np.ones(len(transformed_test)), transformed_test))
        @ coefficients
    )


def equal_problem_mse(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> float:
    values = []
    group_array = np.asarray(groups)
    for group in sorted(set(group_array.tolist())):
        mask = group_array == group
        values.append(
            float(np.mean(np.square(targets[mask] - predictions[mask])))
        )
    return float(np.mean(values))


def contiguous_cv_multioutput(
    design: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    folds: int = 4,
) -> dict[str, float | int]:
    length = len(design)
    fold_count = max(2, min(int(folds), length))
    fold_ids = np.minimum(
        np.floor(np.arange(length) * fold_count / length).astype(int),
        fold_count - 1,
    )
    groups = np.asarray(["trace"] * length)
    # Treat each time block as the held-out unit while fitting all other tokens.
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets)
    errors, baselines = [], []
    for fold in range(fold_count):
        test = fold_ids == fold
        train = ~test
        train_x, test_x = finite_train_transform(x[train], x[test])
        coefficients = fit_weighted_ridge(train_x, y[train], ridge)
        prediction = np.column_stack((np.ones(test.sum()), test_x)) @ coefficients
        train_mean = y[train].mean(axis=0)
        errors.append(float(np.mean(np.square(y[test] - prediction))))
        baselines.append(float(np.mean(np.square(y[test] - train_mean))))
    mse = float(np.mean(errors))
    baseline_mse = float(np.mean(baselines))
    return {
        "blocked_cv_mse": mse,
        "blocked_cv_baseline_mse": baseline_mse,
        "blocked_cv_r2": 1.0 - mse / max(baseline_mse, EPS),
        "feature_count": x.shape[1],
    }


def in_sample_fraction(
    design: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> float:
    x, _ = finite_train_transform(design, design)
    y = np.asarray(targets, dtype=np.float64)
    coefficients = fit_weighted_ridge(x, y, ridge)
    prediction = np.column_stack((np.ones(len(x)), x)) @ coefficients
    residual = float(np.square(y - prediction).sum())
    centered = float(np.square(y - y.mean(axis=0)).sum())
    return 1.0 - residual / max(centered, EPS)


def stratified_problem_folds(
    rows: list[dict[str, str]],
    requested_folds: int,
) -> np.ndarray:
    problem_family: dict[str, str] = {}
    for row in rows:
        problem_family[row["problem_id"]] = row["family"]
    by_family: dict[str, list[str]] = defaultdict(list)
    for problem, family in problem_family.items():
        by_family[family].append(problem)
    minimum = min(len(values) for values in by_family.values())
    folds = max(2, min(int(requested_folds), minimum))
    assignment: dict[str, int] = {}
    for family_index, family in enumerate(sorted(by_family)):
        for index, problem in enumerate(sorted(by_family[family])):
            assignment[problem] = (index + family_index) % folds
    return np.asarray([assignment[row["problem_id"]] for row in rows], dtype=int)


def leave_family_out_folds(rows: list[dict[str, str]]) -> np.ndarray:
    families = {value: index for index, value in enumerate(sorted({r["family"] for r in rows}))}
    return np.asarray([families[row["family"]] for row in rows], dtype=int)


def resolve_tokenizer(config: dict, skip: bool):
    if skip or config["model"]["backend"] == "synthetic":
        return None
    model_id = config["model"].get("id")
    if model_id in ("auto_from_adapter", "", None):
        adapter_path = resolve_adapter_path(config["model"])
        if not adapter_path:
            raise ValueError("model.adapter_path is required")
        adapter_file = Path(adapter_path) / "adapter_config.json"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"cannot resolve tokenizer: {adapter_file} is missing")
        adapter = json.loads(adapter_file.read_text(encoding="utf-8"))
        model_id = adapter.get("base_model_name_or_path")
    if not model_id:
        raise ValueError("could not resolve tokenizer model id")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=config["model"].get("tokenizer_revision")
        or config["model"].get("revision"),
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )


def formatted_token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return len(text.split())
    formatted = text
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return len(tokenizer(formatted, add_special_tokens=True)["input_ids"])


def analyze_temporal_bases(
    paths: dict[str, Path],
    layers: list[int],
    minimum_length: int,
    ridge: float,
) -> tuple[list[dict], list[dict]]:
    store = TraceStore(paths["traces"])
    selected = set(layers)
    rows: list[dict] = []
    for trace_index, (metadata, activations) in enumerate(store.iter_traces(), 1):
        layer = int(metadata["layer"])
        if layer not in selected or len(activations) < minimum_length:
            continue
        t = np.linspace(0.0, 1.0, len(activations), endpoint=True)
        endpoint_distance = float(np.linalg.norm(activations[-1] - activations[0]))
        step_scale = float(
            np.sqrt(
                np.mean(
                    np.square(np.linalg.norm(np.diff(activations, axis=0), axis=1))
                )
            )
        )
        for name, design in temporal_models(t).items():
            result = contiguous_cv_multioutput(design, activations, ridge)
            rows.append(
                {
                    "request_id": metadata["request_id"],
                    "problem_id": metadata["problem_id"],
                    "family": metadata["task_family"],
                    "layer": layer,
                    "trace_length": len(activations),
                    "model": name,
                    "rank_without_intercept": design.shape[1],
                    "in_sample_variance_fraction": in_sample_fraction(
                        design, activations, ridge
                    ),
                    "endpoint_distance": endpoint_distance,
                    "endpoint_distance_in_step_rms": endpoint_distance
                    / max(step_scale, EPS),
                    **result,
                }
            )
        if trace_index % 32 == 0:
            print(f"Temporal open-basis audit: inspected {trace_index} traces")
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), str(row["model"]))].append(row)
    summary = []
    for (layer, model), values in sorted(grouped.items()):
        summary.append(
            {
                "layer": layer,
                "model": model,
                "rank_without_intercept": values[0]["rank_without_intercept"],
                "problem_count": len({value["problem_id"] for value in values}),
                "trace_count": len(values),
                "mean_blocked_cv_r2": float(
                    np.mean([value["blocked_cv_r2"] for value in values])
                ),
                "median_blocked_cv_r2": float(
                    np.median([value["blocked_cv_r2"] for value in values])
                ),
                "mean_in_sample_variance_fraction": float(
                    np.mean(
                        [value["in_sample_variance_fraction"] for value in values]
                    )
                ),
            }
        )
    return rows, summary


def build_observational_designs(
    rows: list[dict[str, str]],
    actual_tokens: np.ndarray,
) -> dict[str, np.ndarray]:
    token = np.asarray(actual_tokens, dtype=np.float64)
    token_scale = max(float(np.nanmedian(token)), 1.0)
    token = token / token_scale
    position = np.column_stack(
        (
            token,
            token**2,
            np.asarray([numeric(row, "sentence_count_proxy") for row in rows]),
        )
    )
    nuisance = np.column_stack(
        (
            position,
            np.asarray([numeric(row, "confidence") for row in rows]),
            np.asarray([numeric(row, "eos_logit") for row in rows]),
            np.asarray([numeric(row, "termination_allowed") for row in rows]),
            *[
                np.asarray([numeric(row, column) for row in rows])
                for column in OPERATION_COLUMNS
            ],
        )
    )
    progress = np.asarray([numeric(row, "structural_progress") for row in rows])
    bases = progress_bases(progress)
    designs = {
        "position_only": position,
        "nuisance_full": nuisance,
    }
    for name, basis in bases.items():
        designs[f"nuisance_plus_{name}"] = np.column_stack((nuisance, basis))
    spline = bases["progress_spline_open"]
    families = sorted({row["family"] for row in rows})
    interactions = []
    for family in families:
        indicator = np.asarray([row["family"] == family for row in rows], dtype=float)
        interactions.append(spline * indicator[:, None])
    designs["nuisance_plus_progress_spline_by_family"] = np.column_stack(
        (nuisance, *interactions)
    )
    return designs


def center_targets_within_problem(
    targets: np.ndarray,
    rows: list[dict[str, str]],
) -> np.ndarray:
    output = np.asarray(targets, dtype=np.float32).copy()
    problems = np.asarray([row["problem_id"] for row in rows])
    for problem in sorted(set(problems.tolist())):
        mask = problems == problem
        reference = mask & np.asarray(
            [row["condition"] in TRAJECTORY_CONDITIONS for row in rows],
            dtype=bool,
        )
        if not reference.any():
            reference = mask
        output[mask] -= output[reference].mean(
            axis=0,
            dtype=np.float64,
        ).astype(output.dtype)
    return output


def analyze_progress_models(
    config: dict,
    paths: dict[str, Path],
    layers: list[int],
    ridge: float,
    skip_tokenizer: bool,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    observational = read_csv(paths["tables"] / "observational_cross.csv")
    prefixes = {
        row["variant_id"]: row
        for row in read_jsonl(paths["root"] / "counterfactual_prefixes.jsonl")
    }
    tokenizer = resolve_tokenizer(config, skip_tokenizer)
    token_lookup = {
        variant_id: formatted_token_count(tokenizer, row["text"])
        for variant_id, row in prefixes.items()
        if row.get("exact_state_valid", False)
    }
    token_rows = []
    seen_variants: set[str] = set()
    for row in observational:
        variant = row["variant_id"]
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        actual = token_lookup[variant]
        proxy = int(float(row["token_count_proxy"]))
        token_rows.append(
            {
                "variant_id": variant,
                "problem_id": row["problem_id"],
                "family": row["family"],
                "condition": row["condition"],
                "word_count_proxy": proxy,
                "actual_formatted_token_count": actual,
                "token_minus_word_count": actual - proxy,
                "token_to_word_ratio": actual / max(proxy, 1),
                "source": "word_proxy" if tokenizer is None else "qwen_tokenizer",
            }
        )

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

    comparison: list[dict] = []
    geometry: list[dict] = []
    condition_holdout: list[dict] = []
    for layer_index, layer in enumerate(layers, 1):
        indices = np.flatnonzero(activation_layers == layer)
        layer_rows = [observational[index] for index in indices]
        layer_targets = center_targets_within_problem(
            activations[indices],
            layer_rows,
        )
        actual_tokens = np.asarray(
            [token_lookup[row["variant_id"]] for row in layer_rows],
            dtype=np.float64,
        )
        designs = build_observational_designs(layer_rows, actual_tokens)
        groups = np.asarray([row["problem_id"] for row in layer_rows])
        trajectory = np.asarray(
            [row["condition"] in TRAJECTORY_CONDITIONS for row in layer_rows],
            dtype=bool,
        )
        if not trajectory.any() or trajectory.all():
            raise ValueError(
                "both ordinary trajectory and special-probe conditions are required"
            )
        trajectory_rows = [
            row for row, keep in zip(layer_rows, trajectory) if keep
        ]
        trajectory_groups = groups[trajectory]
        problem_folds = stratified_problem_folds(
            trajectory_rows,
            int(config["analysis"].get("cv_folds", 5)),
        )
        family_folds = leave_family_out_folds(trajectory_rows)
        model_results: dict[tuple[str, str], dict] = {}
        for scheme, folds in (
            ("grouped_problem_cv", problem_folds),
            ("leave_one_family_out", family_folds),
        ):
            for name, design in designs.items():
                result = grouped_cv_multioutput(
                    design[trajectory],
                    layer_targets[trajectory],
                    trajectory_groups,
                    folds,
                    ridge,
                )
                model_results[(scheme, name)] = result
            nuisance_mse = float(
                model_results[(scheme, "nuisance_full")]["cv_mse"]
            )
            for name in designs:
                result = model_results[(scheme, name)]
                comparison.append(
                    {
                        "layer": layer,
                        "split_scheme": scheme,
                        "model": name,
                        "problem_count": len(set(groups.tolist())),
                        "family_count": len(
                            {row["family"] for row in trajectory_rows}
                        ),
                        "dataset_scope": "ordinary_trajectory_conditions",
                        "target_centering": "within_problem_trajectory_mean",
                        "token_count_source": (
                            "word_proxy" if tokenizer is None else "qwen_tokenizer"
                        ),
                        "incremental_r2_vs_nuisance": (
                            float("nan")
                            if name in {"position_only", "nuisance_full"}
                            else (nuisance_mse - float(result["cv_mse"]))
                            / max(nuisance_mse, EPS)
                        ),
                        **result,
                    }
                )

        special_conditions = sorted(
            {
                row["condition"]
                for row, is_trajectory in zip(layer_rows, trajectory)
                if not is_trajectory
            }
        )
        predictions = {
            name: fit_predict_multioutput(
                design[trajectory],
                layer_targets[trajectory],
                trajectory_groups,
                design[~trajectory],
                ridge,
            )
            for name, design in designs.items()
        }
        special_rows = [
            row for row, keep in zip(layer_rows, ~trajectory) if keep
        ]
        special_targets = layer_targets[~trajectory]
        special_groups = groups[~trajectory]
        for condition in special_conditions:
            condition_mask = np.asarray(
                [row["condition"] == condition for row in special_rows],
                dtype=bool,
            )
            nuisance_mse = equal_problem_mse(
                special_targets[condition_mask],
                predictions["nuisance_full"][condition_mask],
                special_groups[condition_mask],
            )
            for name in designs:
                mse = equal_problem_mse(
                    special_targets[condition_mask],
                    predictions[name][condition_mask],
                    special_groups[condition_mask],
                )
                condition_holdout.append(
                    {
                        "layer": layer,
                        "condition": condition,
                        "model": name,
                        "problem_count": len(
                            set(special_groups[condition_mask].tolist())
                        ),
                        "condition_holdout_mse": mse,
                        "improvement_vs_nuisance": (
                            float("nan")
                            if name in {"position_only", "nuisance_full"}
                            else (nuisance_mse - mse) / max(nuisance_mse, EPS)
                        ),
                        "fit_scope": "ordinary_trajectory_conditions",
                        "test_scope": "special_probe_condition",
                    }
                )

        with (
            paths["models"] / "subspace_index.json"
        ).open("r", encoding="utf-8") as handle:
            selected_index = json.load(handle)
        estimator = selected_index["selected_by_layer"][str(layer)]
        with np.load(
            paths["models"] / f"subspace_layer_{layer}_{estimator}.npz",
            allow_pickle=False,
        ) as model:
            basis = np.asarray(model["basis"], dtype=np.float64)
            center = np.asarray(model["center"], dtype=np.float64)
        raw_targets = np.asarray(activations[indices], dtype=np.float64)
        centered = raw_targets - center
        projected = (centered @ basis) @ basis.T
        residual = centered - projected
        total_norm = np.linalg.norm(centered, axis=1)
        residual_norm = np.linalg.norm(residual, axis=1)
        captured = np.square(np.linalg.norm(projected, axis=1)) / np.maximum(
            np.square(total_norm),
            EPS,
        )
        by_condition: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(layer_rows):
            by_condition[row["condition"]].append(index)
        for condition, local_indices in sorted(by_condition.items()):
            local = np.asarray(local_indices, dtype=int)
            normalized = residual_norm[local] / np.maximum(total_norm[local], EPS)
            geometry.append(
                {
                    "layer": layer,
                    "condition": condition,
                    "problem_count": len(
                        {layer_rows[index]["problem_id"] for index in local}
                    ),
                    "observation_count": len(local),
                    "mean_normalized_manifold_distance": float(
                        np.mean(normalized)
                    ),
                    "median_normalized_manifold_distance": float(
                        np.median(normalized)
                    ),
                    "p90_normalized_manifold_distance": float(
                        np.quantile(normalized, 0.90)
                    ),
                    "mean_plane_energy_fraction": float(
                        np.mean(captured[local])
                    ),
                }
            )
        print(
            f"Open progress models: completed layer {layer} "
            f"({layer_index}/{len(layers)})"
        )

    grouped_results: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in comparison:
        grouped_results[(row["split_scheme"], row["model"])].append(row)
    summary = []
    for (scheme, model), values in sorted(grouped_results.items()):
        increments = np.asarray(
            [float(value["incremental_r2_vs_nuisance"]) for value in values],
            dtype=np.float64,
        )
        summary.append(
            {
                "split_scheme": scheme,
                "model": model,
                "layer_count": len(values),
                "mean_cv_r2": float(
                    np.mean([float(value["cv_r2"]) for value in values])
                ),
                "median_cv_r2": float(
                    np.median([float(value["cv_r2"]) for value in values])
                ),
                "mean_incremental_r2_vs_nuisance": (
                    float(np.nanmean(increments))
                    if np.isfinite(increments).any()
                    else float("nan")
                ),
                "median_incremental_r2_vs_nuisance": (
                    float(np.nanmedian(increments))
                    if np.isfinite(increments).any()
                    else float("nan")
                ),
            }
        )
    return comparison, summary, condition_holdout, geometry, token_rows


def plot_layer_series(
    path: Path,
    rows: list[dict],
    models: list[str],
    model_key: str,
    value_key: str,
    title: str,
    y_label: str,
) -> None:
    series = {}
    for model in models:
        selected = sorted(
            [row for row in rows if row[model_key] == model],
            key=lambda row: int(row["layer"]),
        )
        if selected:
            series[model] = (
                np.asarray([int(row["layer"]) for row in selected]),
                np.asarray([float(row[value_key]) for row in selected]),
            )
    if series:
        line_svg(path, series, title, "layer", y_label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare closed k=1 trajectories with open temporal and semantic "
            "progress-manifold models before causal interventions"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--layers",
        default="late-half",
        help="'late-half' (default), 'all', comma-separated layers, or ranges such as 32-63",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Use the existing whitespace word-count proxy instead of Qwen token counts",
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip file-01 output-token open/closed basis comparison",
    )
    parser.add_argument(
        "--skip-progress",
        action="store_true",
        help="Skip file-05 controlled-prefix progress-manifold comparison",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    ridge = float(config["analysis"].get("ridge", 0.001))

    observational_path = paths["tables"] / "observational_cross.csv"
    available: set[int] = set()
    if observational_path.is_file():
        available.update(
            int(row["layer"]) for row in read_csv(observational_path)
        )
    trace_manifest = TraceStore(paths["traces"]).read_manifest()
    available.update(int(row["layer"]) for row in trace_manifest)
    layers = parse_layers(args.layers, sorted(available))
    print(f"Open-manifold comparison layers: {layers}")

    temporal_rows: list[dict] = []
    temporal_summary: list[dict] = []
    if not args.skip_temporal:
        temporal_rows, temporal_summary = analyze_temporal_bases(
            paths,
            layers,
            int(config["analysis"]["minimum_trace_length"]),
            ridge,
        )
        write_csv(
            paths["tables"] / "temporal_open_basis_comparison.csv",
            temporal_rows,
        )
        write_csv(
            paths["tables"] / "temporal_open_basis_summary.csv",
            temporal_summary,
        )
        plot_layer_series(
            paths["figures"] / "14_open_vs_closed_temporal_models.svg",
            temporal_summary,
            [
                "closed_fourier_k1",
                "quadratic_open",
                "dct_open_rank2",
                "cubic_open",
                "closed_k1_plus_drift",
                "open_spiral_half_turn",
            ],
            "model",
            "mean_blocked_cv_r2",
            "Open versus periodic output-token trajectories",
            "mean blocked-token CV R²",
        )

    progress_rows: list[dict] = []
    progress_summary: list[dict] = []
    condition_holdout_rows: list[dict] = []
    geometry_rows: list[dict] = []
    token_rows: list[dict] = []
    if not args.skip_progress:
        if not observational_path.is_file():
            raise FileNotFoundError(
                f"{observational_path} is missing; run files 04 and 05 first"
            )
        (
            progress_rows,
            progress_summary,
            condition_holdout_rows,
            geometry_rows,
            token_rows,
        ) = analyze_progress_models(
            config,
            paths,
            layers,
            ridge,
            args.skip_tokenizer,
        )
        write_csv(
            paths["tables"] / "progress_manifold_model_comparison.csv",
            progress_rows,
        )
        write_csv(
            paths["tables"] / "progress_manifold_model_summary.csv",
            progress_summary,
        )
        write_csv(
            paths["tables"] / "progress_manifold_condition_holdout.csv",
            condition_holdout_rows,
        )
        write_csv(
            paths["tables"] / "observational_geometry_normalized.csv",
            geometry_rows,
        )
        write_csv(
            paths["tables"] / "counterfactual_actual_token_counts.csv",
            token_rows,
        )
        problem_cv = [
            row
            for row in progress_rows
            if row["split_scheme"] == "grouped_problem_cv"
        ]
        plot_layer_series(
            paths["figures"] / "15_open_progress_models_by_layer.svg",
            problem_cv,
            [
                "nuisance_plus_progress_linear",
                "nuisance_plus_progress_quadratic_open",
                "nuisance_plus_progress_spline_open",
                "nuisance_plus_progress_closed_k1",
                "nuisance_plus_progress_spiral_half_turn",
                "nuisance_plus_progress_spiral_full_turn",
            ],
            "model",
            "incremental_r2_vs_nuisance",
            "Held-out progress geometry beyond position/confidence/EOS",
            "incremental R² over nuisance model",
        )

    report = {
        "config": args.config,
        "layers": layers,
        "ridge": ridge,
        "token_count_source": (
            "word_proxy"
            if args.skip_tokenizer or config["model"]["backend"] == "synthetic"
            else "qwen_tokenizer"
        ),
        "target_normalization": (
            "raw activations centered within problem; this removes content "
            "offsets without aligning layer coordinate gauges"
        ),
        "interpretation": {
            "temporal_open_basis_comparison": (
                "Describes output-token trajectory shape only; it does not "
                "supply semantic-state labels."
            ),
            "progress_manifold_model_comparison": (
                "Positive held-out incremental_r2_vs_nuisance means exact "
                "structural progress adds activation information beyond actual "
                "token count, confidence, EOS logit, termination, and operation."
            ),
            "condition_holdout": (
                "Ordinary state trajectories fit each model; teleport, rollback, "
                "loop, confidence, verification, and termination probes only "
                "test generalization and never fit the curve."
            ),
            "closed_k1": (
                "The fixed closed model maps progress 0 and 1 to the same phase."
            ),
            "open_models": (
                "Linear, polynomial, DCT, spline, and spiral-with-drift models "
                "do not require endpoint closure."
            ),
            "smoke_warning": (
                "Six-problem smoke results are descriptive; discovery-scale "
                "problem-grouped and leave-family-out results are required."
            ),
        },
        "outputs": {
            "temporal_rows": len(temporal_rows),
            "progress_rows": len(progress_rows),
            "condition_holdout_rows": len(condition_holdout_rows),
            "geometry_rows": len(geometry_rows),
            "token_rows": len(token_rows),
        },
    }
    atomic_json(paths["tables"] / "open_progress_model_report.json", report)
    print(
        "Wrote open/closed model comparison to "
        f"{paths['tables']} and figures 14-15 to {paths['figures']}"
    )


if __name__ == "__main__":
    main()
