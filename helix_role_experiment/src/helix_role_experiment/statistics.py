from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import deterministic_id


EPS = 1e-12


def grouped_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    if len(y) != len(group_array):
        raise ValueError("values and groups must be aligned")
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    if strata is None:
        group_strata = {"all": unique_groups}
    else:
        strata_array = np.asarray(strata)
        if len(strata_array) != len(y):
            raise ValueError("strata must align with values")
        group_strata = {}
        for stratum in np.unique(strata_array):
            groups_here = np.unique(group_array[strata_array == stratum])
            group_strata[str(stratum)] = groups_here
    estimates = []
    for _ in range(draws):
        sampled = []
        for groups_here in group_strata.values():
            sampled.extend(rng.choice(groups_here, size=len(groups_here), replace=True).tolist())
        sample_values = []
        for group in sampled:
            sample_values.extend(y[group_array == group].tolist())
        estimates.append(float(np.mean(sample_values)))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(np.mean(y)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_draws": int(draws),
        "problem_count": int(len(unique_groups)),
    }


def paired_problem_effect(
    values: np.ndarray,
    conditions: np.ndarray,
    problems: np.ndarray,
    treatment: str,
    control: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=np.float64)
    condition_array = np.asarray(conditions)
    problem_array = np.asarray(problems)
    effects = []
    ids = []
    for problem in np.unique(problem_array):
        mask = problem_array == problem
        treated = y[mask & (condition_array == treatment)]
        controlled = y[mask & (condition_array == control)]
        if len(treated) and len(controlled):
            effects.append(float(treated.mean() - controlled.mean()))
            ids.append(problem)
    return np.asarray(effects), np.asarray(ids)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < EPS] = 1.0
    return mean, scale


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    penalty = ridge * np.eye(features.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T @ targets)


def grouped_fold_ids(groups: Iterable[object], folds: int, seed: int) -> np.ndarray:
    group_array = np.asarray(list(groups))
    unique = np.unique(group_array)
    mapping = {
        group: int(deterministic_id(seed, group)[:12], 16) % folds for group in unique
    }
    return np.asarray([mapping[group] for group in group_array], dtype=int)


@dataclass
class CrossValidatedResult:
    predictions: np.ndarray
    mse: float
    r2: float
    heldout_log_likelihood: float


def grouped_cross_validated_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
    ridge: float = 1e-3,
    seed: int = 0,
) -> CrossValidatedResult:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    fold_ids = grouped_fold_ids(groups, folds, seed)
    predictions = np.full_like(targets, np.nan)
    residual_variances = []
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        if not test.any() or not train.any():
            continue
        mean, scale = _standardize_fit(features[train])
        train_x = np.column_stack(
            (np.ones(train.sum()), (features[train] - mean) / scale)
        )
        test_x = np.column_stack((np.ones(test.sum()), (features[test] - mean) / scale))
        coefficients = ridge_fit(train_x, targets[train], ridge)
        predictions[test] = test_x @ coefficients
        train_residual = targets[train] - train_x @ coefficients
        residual_variances.append(np.var(train_residual, axis=0) + EPS)
    valid = np.isfinite(predictions).all(axis=1)
    if not valid.any():
        raise ValueError("no nonempty held-out folds")
    residual = targets[valid] - predictions[valid]
    mse = float(np.square(residual).mean())
    baseline = targets[valid] - targets[valid].mean(axis=0)
    r2 = 1.0 - float(np.square(residual).sum()) / max(float(np.square(baseline).sum()), EPS)
    variance = np.mean(residual_variances, axis=0)
    log_likelihood = float(
        np.mean(
            -0.5
            * (
                np.log(2.0 * np.pi * variance)
                + np.square(residual) / variance
            ).sum(axis=1)
        )
    )
    return CrossValidatedResult(predictions, mse, r2, log_likelihood)


def incremental_r2(reduced_sse: float, full_sse: float) -> float:
    return float((reduced_sse - full_sse) / max(reduced_sse, EPS))


def predictor_block_comparison(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    ridge: float,
    seed: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    cumulative: list[np.ndarray] = []
    previous_sse: float | None = None
    for name, block in blocks.items():
        cumulative.append(np.asarray(block, dtype=np.float64))
        design = np.column_stack(cumulative)
        result = grouped_cross_validated_ridge(design, y, groups, folds, ridge, seed)
        sse = result.mse * np.asarray(y).size
        rows.append(
            {
                "model": "+".join(list(blocks.keys())[: len(cumulative)]),
                "added_block": name,
                "cv_r2": result.r2,
                "cv_mse": result.mse,
                "heldout_log_likelihood": result.heldout_log_likelihood,
                "partial_r2": (
                    float("nan")
                    if previous_sse is None
                    else incremental_r2(previous_sse, sse)
                ),
            }
        )
        previous_sse = sse
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

