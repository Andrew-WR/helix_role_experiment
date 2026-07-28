from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .fourier import EPS, center_trace, harmonic_basis, temporal_project


def orthonormalize(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(matrix, dtype=np.float64))
    if rank is None:
        rank = min(matrix.shape)
    diagonal = np.abs(np.diag(r))
    available = int(np.sum(diagonal > EPS))
    if available < rank:
        raise ValueError(f"matrix has rank {available}, requested {rank}")
    return q[:, :rank]


def trace_harmonic_plane(x: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    centered, _ = center_trace(x)
    reconstruction = temporal_project(centered, harmonic_basis(len(centered), k))
    _, singular, vt = np.linalg.svd(reconstruction, full_matrices=False)
    if len(singular) < 2 or singular[1] <= EPS:
        raise ValueError("trace harmonic plane is rank deficient")
    return vt[:2].T, singular[:2]


def grassmann_mean(planes: Iterable[np.ndarray], rank: int = 2) -> tuple[np.ndarray, np.ndarray]:
    plane_list = [orthonormalize(plane, rank) for plane in planes]
    if not plane_list:
        raise ValueError("at least one plane is required")
    projector_mean = sum(plane @ plane.T for plane in plane_list) / len(plane_list)
    values, vectors = np.linalg.eigh(projector_mean)
    order = np.argsort(values)[::-1]
    return vectors[:, order[:rank]], values[order]


def pooled_spectral_covariances(
    traces: Iterable[np.ndarray],
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    s_k: np.ndarray | None = None
    s_not_k: np.ndarray | None = None
    for trace in traces:
        centered, _ = center_trace(trace)
        projected = temporal_project(centered, harmonic_basis(len(centered), k))
        residual = centered - projected
        current_k = projected.T @ projected
        current_residual = residual.T @ residual
        s_k = current_k if s_k is None else s_k + current_k
        s_not_k = current_residual if s_not_k is None else s_not_k + current_residual
    if s_k is None or s_not_k is None:
        raise ValueError("at least one trace is required")
    return s_k, s_not_k


def generalized_spectral_plane(
    traces: Iterable[np.ndarray],
    ridge: float,
    rank: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    s_k, s_residual = pooled_spectral_covariances(traces)
    dimension = s_k.shape[0]
    scale = float(np.trace(s_residual) / max(dimension, 1))
    b = s_residual + max(float(ridge), EPS) * max(scale, EPS) * np.eye(dimension)
    b_values, b_vectors = np.linalg.eigh(b)
    b_values = np.maximum(b_values, EPS)
    b_inverse_sqrt = b_vectors @ np.diag(b_values ** -0.5) @ b_vectors.T
    transformed = b_inverse_sqrt @ s_k @ b_inverse_sqrt
    values, vectors = np.linalg.eigh(transformed)
    order = np.argsort(values)[::-1]
    raw = b_inverse_sqrt @ vectors[:, order[:rank]]
    return orthonormalize(raw, rank), values[order]


def generalized_spectral_plane_iterative(
    traces: Iterable[np.ndarray],
    ridge: float,
    rank: int = 2,
    tolerance: float = 1e-7,
    max_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Dual generalized eigensolver without dense hidden-size covariances.

    If A vertically stacks each trace's two orthonormal harmonic coefficients,
    then S1=A.T@A. Nonzero generalized eigenvectors are in
    `(S_not1 + rho I)^-1 A.T`. We obtain that block with independent
    conjugate-gradient solves sharing residual-covariance matvecs, then solve
    the at-most-2N dimensional dual eigenproblem.
    """

    prepared = []
    coefficient_rows = []
    residual_energy = 0.0
    dimension = None
    for trace in traces:
        centered, _ = center_trace(trace)
        q = harmonic_basis(len(centered), 1)
        coefficients = q.T @ centered
        prepared.append((centered, coefficients))
        coefficient_rows.append(coefficients)
        residual_energy += float(
            np.square(centered).sum() - np.square(coefficients).sum()
        )
        dimension = centered.shape[1]
    if not prepared or dimension is None:
        raise ValueError("at least one trace is required")
    a = np.vstack(coefficient_rows)
    ridge_absolute = max(float(ridge), EPS) * max(
        residual_energy / max(dimension, 1), EPS
    )

    def residual_operator(matrix: np.ndarray) -> np.ndarray:
        result = ridge_absolute * matrix
        for centered, coefficients in prepared:
            result += centered.T @ (centered @ matrix)
            result -= coefficients.T @ (coefficients @ matrix)
        return result

    right_hand_side = a.T
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    residual_squared = np.sum(residual * residual, axis=0)
    initial = np.sqrt(np.maximum(residual_squared, EPS))
    iterations = 0
    active = np.ones(right_hand_side.shape[1], dtype=bool)
    for iterations in range(1, max_iterations + 1):
        applied = residual_operator(direction)
        denominator = np.sum(direction * applied, axis=0)
        alpha = np.divide(
            residual_squared,
            denominator,
            out=np.zeros_like(residual_squared),
            where=np.abs(denominator) > EPS,
        )
        alpha[~active] = 0.0
        solution += direction * alpha
        residual -= applied * alpha
        next_squared = np.sum(residual * residual, axis=0)
        active = np.sqrt(np.maximum(next_squared, 0.0)) > tolerance * initial
        if not active.any():
            residual_squared = next_squared
            break
        beta = np.divide(
            next_squared,
            np.maximum(residual_squared, EPS),
            out=np.zeros_like(next_squared),
        )
        beta[~active] = 0.0
        direction = residual + direction * beta
        residual_squared = next_squared
    dual = a @ solution
    dual = 0.5 * (dual + dual.T)
    values, vectors = np.linalg.eigh(dual)
    order = np.argsort(values)[::-1]
    raw = solution @ vectors[:, order[:rank]]
    return orthonormalize(raw, rank), values[order], iterations


def complex_k1_coefficient(x: np.ndarray) -> np.ndarray:
    centered, _ = center_trace(x)
    t = np.arange(len(centered))
    carrier = np.exp(-2j * np.pi * t / len(centered))
    return (2.0 / len(centered)) * np.einsum("t,td->d", carrier, centered)


@dataclass
class ComplexCoefficientModel:
    components: np.ndarray
    singular_values: np.ndarray
    variance_fraction: np.ndarray
    real_plane: np.ndarray


def fit_complex_coefficient_model(
    traces: Iterable[np.ndarray],
    complex_rank: int = 1,
) -> ComplexCoefficientModel:
    coefficients = np.vstack([complex_k1_coefficient(trace) for trace in traces])
    if len(coefficients) == 0:
        raise ValueError("at least one trace is required")
    _, singular, vh = np.linalg.svd(coefficients, full_matrices=False)
    components = vh[:complex_rank]
    variance = singular**2 / max(float(np.sum(singular**2)), EPS)
    first = components[0]
    candidate = np.column_stack((first.real, -first.imag))
    if np.linalg.matrix_rank(candidate, tol=1e-10) < 2:
        # The complex component may be a line; include the strongest second
        # real direction without pretending it is an identified ellipse axis.
        stacked = np.column_stack(
            [component.real for component in components]
            + [-component.imag for component in components]
        )
        _, _, vt = np.linalg.svd(np.real(coefficients), full_matrices=False)
        stacked = np.column_stack((stacked, vt[0]))
        candidate = stacked
    real_plane = orthonormalize(candidate, 2)
    return ComplexCoefficientModel(components, singular, variance, real_plane)


def projector_similarity(a: np.ndarray, b: np.ndarray) -> float:
    qa = orthonormalize(a)
    qb = orthonormalize(b)
    rank = min(qa.shape[1], qb.shape[1])
    return float(np.square(qa.T @ qb).sum() / rank)


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = orthonormalize(a)
    qb = orthonormalize(b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


def chordal_distance(a: np.ndarray, b: np.ndarray) -> float:
    angles = principal_angles(a, b)
    return float(np.sqrt(np.square(np.sin(angles)).sum()))


def plane_similarity_matrix(planes: list[np.ndarray]) -> np.ndarray:
    output = np.empty((len(planes), len(planes)), dtype=np.float64)
    for i, left in enumerate(planes):
        for j, right in enumerate(planes):
            output[i, j] = projector_similarity(left, right)
    return output


def fit_whitener(
    traces: Iterable[np.ndarray],
    basis: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    plane = orthonormalize(basis, 2)
    coordinates = []
    means = []
    for trace in traces:
        centered, mean = center_trace(trace)
        coordinates.append(centered @ plane)
        means.append(mean)
    if not coordinates:
        raise ValueError("at least one trace is required")
    pooled = np.vstack(coordinates)
    covariance = pooled.T @ pooled / len(pooled)
    values, vectors = np.linalg.eigh(covariance)
    floor = max(ridge * float(values.max()), EPS)
    inverse_sqrt = vectors @ np.diag(np.maximum(values, floor) ** -0.5) @ vectors.T
    return inverse_sqrt, np.mean(means, axis=0)


def projected_features(
    x: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    whitener: np.ndarray,
    radius_threshold: float,
) -> dict[str, np.ndarray]:
    array = np.asarray(x, dtype=np.float64)
    plane = orthonormalize(basis, 2)
    raw = (array - np.asarray(center)) @ plane
    whitened = raw @ np.asarray(whitener)
    radius = np.linalg.norm(whitened, axis=1)
    angle = np.arctan2(whitened[:, 1], whitened[:, 0])
    reliable = radius >= radius_threshold
    unwrapped = np.full_like(angle, np.nan)
    if reliable.any():
        indices = np.flatnonzero(reliable)
        # Unwrap contiguous reliable runs; do not bridge low-radius gaps.
        starts = np.r_[0, np.flatnonzero(np.diff(indices) > 1) + 1]
        ends = np.r_[starts[1:], len(indices)]
        for start, end in zip(starts, ends):
            run = indices[start:end]
            unwrapped[run] = np.unwrap(angle[run])
    angular_velocity = np.r_[np.nan, np.diff(unwrapped)]
    radial_velocity = np.r_[np.nan, np.diff(radius)]
    reconstruction = raw @ plane.T
    residual = (array - np.asarray(center)) - reconstruction
    return {
        "coordinate_1": whitened[:, 0],
        "coordinate_2": whitened[:, 1],
        "raw_coordinate_1": raw[:, 0],
        "raw_coordinate_2": raw[:, 1],
        "radius": radius,
        "raw_angle": angle,
        "unwrapped_angle": unwrapped,
        "phase_reliable": reliable,
        "local_angular_velocity": angular_velocity,
        "radial_velocity": radial_velocity,
        "manifold_distance": np.linalg.norm(residual, axis=1),
        "reconstruction_residual": residual,
    }


def heldout_spectral_selectivity(x: np.ndarray, basis: np.ndarray) -> float:
    centered, _ = center_trace(x)
    coords = centered @ orthonormalize(basis, 2)
    harmonic = temporal_project(coords, harmonic_basis(len(coords), 1))
    return float(np.square(harmonic).sum() / max(np.square(coords - harmonic).sum(), EPS))


def random_plane(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    return orthonormalize(rng.normal(size=(dimension, rank)), rank)
