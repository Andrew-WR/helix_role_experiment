from __future__ import annotations

import numpy as np

from .subspaces import orthonormalize


def rotation_matrix(delta: float) -> np.ndarray:
    return np.array(
        [[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]],
        dtype=np.float64,
    )


def within_plane_rotation(
    h: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    delta: float,
    strength: float = 1.0,
) -> np.ndarray:
    activation = np.asarray(h, dtype=np.float64)
    plane = orthonormalize(basis, 2)
    centered = activation - np.asarray(center)
    z = centered @ plane
    target = z @ rotation_matrix(delta).T
    return activation + strength * ((target - z) @ plane.T)


def donor_transplant(
    target: np.ndarray,
    donor: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    plane = orthonormalize(basis, 2)
    origin = np.asarray(center)
    target_z = (np.asarray(target) - origin) @ plane
    donor_z = (np.asarray(donor) - origin) @ plane
    return np.asarray(target) + strength * ((donor_z - target_z) @ plane.T)


def full_frame_interchange(
    target: np.ndarray,
    source: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    target_array = np.asarray(target, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    if target_array.shape != source_array.shape:
        raise ValueError("source and target frames must have equal shape")
    return donor_transplant(target_array, source_array, basis, center, strength)


def radial_intervention(
    h: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    plane = orthonormalize(basis, 2)
    centered = np.asarray(h) - np.asarray(center)
    z = centered @ plane
    return np.asarray(h) + ((scale - 1.0) * z) @ plane.T


def phase_only_intervention(
    h: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    delta: float,
) -> np.ndarray:
    return within_plane_rotation(h, basis, center, delta, strength=1.0)


def magnitude_only_intervention(
    h: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray,
    delta_norm: float,
) -> np.ndarray:
    plane = orthonormalize(basis, 2)
    centered = np.asarray(h) - np.asarray(center)
    z = centered @ plane
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    target_norms = np.maximum(0.0, norms + delta_norm)
    target = z * target_norms / np.maximum(norms, 1e-12)
    return np.asarray(h) + (target - z) @ plane.T


def candidate_orthogonal_patch(
    target: np.ndarray,
    donor: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    plane = orthonormalize(basis, 2)
    projector_orthogonal = np.eye(plane.shape[0]) - plane @ plane.T
    difference = (np.asarray(donor) - np.asarray(target)) @ projector_orthogonal
    return np.asarray(target) + difference


def norm_matched_random_delta(
    reference_delta: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    direction = rng.normal(size=dimension)
    if orthogonal_to is not None:
        plane = orthonormalize(orthogonal_to)
        direction = direction - (direction @ plane) @ plane.T
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError("random direction collapsed after orthogonalization")
    target_norm = np.linalg.norm(np.asarray(reference_delta))
    return direction * target_norm / norm


def intervention_diagnostics(original: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    original_array = np.asarray(original, dtype=np.float64)
    changed_array = np.asarray(changed, dtype=np.float64)
    delta = changed_array - original_array
    return {
        "intervention_norm": float(np.linalg.norm(delta)),
        "activation_norm_before": float(np.linalg.norm(original_array)),
        "activation_norm_after": float(np.linalg.norm(changed_array)),
        "relative_norm_change": float(
            (np.linalg.norm(changed_array) - np.linalg.norm(original_array))
            / max(np.linalg.norm(original_array), 1e-12)
        ),
    }

