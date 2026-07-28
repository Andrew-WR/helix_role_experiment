from __future__ import annotations

import numpy as np

from .subspaces import orthonormalize


def eos_logit_direction(
    lm_head_weight: np.ndarray,
    eos_token_id: int,
    reference_token_ids: list[int] | None = None,
) -> np.ndarray:
    weights = np.asarray(lm_head_weight, dtype=np.float64)
    eos = weights[int(eos_token_id)].copy()
    if reference_token_ids:
        eos -= weights[np.asarray(reference_token_ids, dtype=int)].mean(axis=0)
    norm = np.linalg.norm(eos)
    if norm <= 1e-12:
        raise ValueError("EOS direction has zero norm")
    return eos / norm


def orthogonalize_to_direction(
    basis: np.ndarray,
    direction: np.ndarray,
    rank: int = 2,
) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / max(np.linalg.norm(vector), 1e-12)
    candidate = (np.eye(len(vector)) - np.outer(vector, vector)) @ np.asarray(basis)
    return orthonormalize(candidate, rank)


def subspace_direction_overlap(basis: np.ndarray, direction: np.ndarray) -> float:
    plane = orthonormalize(basis)
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / max(np.linalg.norm(vector), 1e-12)
    return float(np.linalg.norm(plane.T @ vector) ** 2)


def eos_logit_match(
    intervention_logits: np.ndarray,
    control_logits: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    left = np.asarray(intervention_logits, dtype=np.float64)
    right = np.asarray(control_logits, dtype=np.float64)
    return np.abs(left - right) <= tolerance


def direct_eos_bias_for_target(
    baseline_eos_logit: float,
    target_eos_logit: float,
) -> float:
    return float(target_eos_logit - baseline_eos_logit)

