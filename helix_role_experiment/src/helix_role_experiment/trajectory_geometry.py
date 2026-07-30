from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12
PHASE_SPANS = (
    0.5 * np.pi,
    np.pi,
    1.5 * np.pi,
    2.0 * np.pi,
)
RADIUS_SLOPES = (-0.75, -0.375, 0.0, 0.375, 0.75)


def normalized_positions(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("a trajectory requires at least two tokens")
    return np.linspace(0.0, 1.0, int(count), dtype=np.float64)


def rotation_basis(
    progress: np.ndarray,
    omega: float,
    radius_slope: float,
) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float64)
    radius = 1.0 + float(radius_slope) * (progress - 0.5)
    if np.min(radius) <= 0:
        raise ValueError("radius must remain positive")
    return np.column_stack(
        (
            radius * np.cos(float(omega) * progress),
            radius * np.sin(float(omega) * progress),
        )
    )


def design_matrix(
    model: str,
    progress: np.ndarray,
    omega: float = 2.0 * np.pi,
    radius_slope: float = 0.0,
) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float64)
    base = np.column_stack((np.ones(len(progress)), progress))
    if model == "linear":
        return base
    if model == "linear_plus_closed_k1":
        return np.column_stack(
            (
                base,
                np.cos(2.0 * np.pi * progress),
                np.sin(2.0 * np.pi * progress),
            )
        )
    if model == "generalized_helix":
        return np.column_stack(
            (base, rotation_basis(progress, omega, radius_slope))
        )
    raise ValueError(f"unknown trajectory model {model!r}")


def _fit_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.asarray(targets, dtype=np.float64),
    )


def _contiguous_folds(count: int, fold_count: int) -> list[np.ndarray]:
    indices = np.arange(count)
    return [
        value
        for value in np.array_split(indices, min(int(fold_count), count))
        if len(value)
    ]


def blocked_cv_mse(
    activations: np.ndarray,
    progress: np.ndarray,
    model: str,
    ridge: float,
    fold_count: int,
    omega: float = 2.0 * np.pi,
    radius_slope: float = 0.0,
) -> float:
    design = design_matrix(model, progress, omega, radius_slope)
    errors = []
    for test_indices in _contiguous_folds(len(progress), fold_count):
        train = np.ones(len(progress), dtype=bool)
        train[test_indices] = False
        if int(train.sum()) < design.shape[1]:
            raise ValueError("not enough training tokens for blocked validation")
        coefficients = _fit_coefficients(
            design[train],
            activations[train],
            ridge,
        )
        prediction = design[test_indices] @ coefficients
        errors.append(
            float(np.mean(np.square(activations[test_indices] - prediction)))
        )
    return float(np.mean(errors))


@dataclass(frozen=True)
class TrajectoryCurve:
    model: str
    coefficients: np.ndarray
    omega: float = 2.0 * np.pi
    radius_slope: float = 0.0

    def value(self, progress: float | np.ndarray) -> np.ndarray:
        values = np.atleast_1d(np.asarray(progress, dtype=np.float64))
        result = (
            design_matrix(
                self.model,
                values,
                self.omega,
                self.radius_slope,
            )
            @ self.coefficients
        )
        return result[0] if np.ndim(progress) == 0 else result

    def local_delta(self, progress: float, step: float) -> np.ndarray:
        source = float(np.clip(progress, 0.0, 1.0))
        destination = float(np.clip(source + float(step), 0.0, 1.0))
        return self.value(destination) - self.value(source)


def fit_trajectory_models(
    activations: np.ndarray,
    ridge: float = 1e-3,
    fold_count: int = 5,
) -> tuple[dict[str, TrajectoryCurve], list[dict]]:
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or len(values) < 20:
        raise ValueError("trajectory fit requires at least 20 token activations")
    progress = normalized_positions(len(values))
    variance = float(np.mean(np.square(values - values.mean(axis=0))))
    variance = max(variance, EPS)
    candidates: list[dict] = []
    specifications = [
        ("linear", 0.0, 0.0),
        ("linear_plus_closed_k1", 2.0 * np.pi, 0.0),
    ]
    specifications.extend(
        ("generalized_helix", omega, radius_slope)
        for omega in PHASE_SPANS
        for radius_slope in RADIUS_SLOPES
    )
    for model, omega, radius_slope in specifications:
        mse = blocked_cv_mse(
            values,
            progress,
            model,
            ridge,
            fold_count,
            omega,
            radius_slope,
        )
        candidates.append(
            {
                "model": model,
                "omega_radians": float(omega),
                "turns": (
                    float(omega / (2.0 * np.pi))
                    if model == "generalized_helix"
                    else float("nan")
                ),
                "radius_slope": (
                    float(radius_slope)
                    if model == "generalized_helix"
                    else float("nan")
                ),
                "blocked_cv_mse": mse,
                "blocked_cv_r2": 1.0 - mse / variance,
                "selected_within_model": False,
            }
        )
    curves: dict[str, TrajectoryCurve] = {}
    for model in ("linear", "linear_plus_closed_k1", "generalized_helix"):
        model_rows = [row for row in candidates if row["model"] == model]
        winner = min(model_rows, key=lambda row: row["blocked_cv_mse"])
        winner["selected_within_model"] = True
        omega = (
            float(winner["omega_radians"])
            if model == "generalized_helix"
            else 2.0 * np.pi
        )
        radius_slope = (
            float(winner["radius_slope"])
            if model == "generalized_helix"
            else 0.0
        )
        design = design_matrix(
            model,
            progress,
            omega,
            radius_slope,
        )
        curves[model] = TrajectoryCurve(
            model=model,
            coefficients=_fit_coefficients(design, values, ridge),
            omega=omega,
            radius_slope=radius_slope,
        )
    return curves, candidates


def centered_transfer_metrics(
    curve: TrajectoryCurve,
    activations: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(activations, dtype=np.float64)
    progress = normalized_positions(len(values))
    prediction = curve.value(progress)
    centered_values = values - values.mean(axis=0)
    centered_prediction = prediction - prediction.mean(axis=0)
    residual = centered_values - centered_prediction
    mse = float(np.mean(np.square(residual)))
    variance = max(
        float(np.mean(np.square(centered_values))),
        EPS,
    )
    return {
        "centered_transfer_mse": mse,
        "centered_transfer_r2": 1.0 - mse / variance,
        "mean_curve_norm": float(
            np.mean(np.linalg.norm(centered_prediction, axis=1))
        ),
    }


def match_norm(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    value_norm = float(np.linalg.norm(value))
    if target_norm <= EPS:
        return np.zeros_like(value)
    if value_norm <= EPS:
        raise ValueError("cannot norm-match a collapsed local delta")
    return value * (target_norm / value_norm)
