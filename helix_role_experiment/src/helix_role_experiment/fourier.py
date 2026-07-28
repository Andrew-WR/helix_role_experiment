from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EPS = 1e-12


def center_trace(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("trace must have shape [time>=2, features]")
    mean = array.mean(axis=0)
    return array - mean, mean


def harmonic_basis(length: int, k: int = 1) -> np.ndarray:
    if length < 2:
        raise ValueError("length must be at least two")
    if not 0 < k < length:
        raise ValueError("k must satisfy 0 < k < length")
    t = np.arange(length, dtype=np.float64)
    basis = np.column_stack(
        (np.cos(2.0 * np.pi * k * t / length), np.sin(2.0 * np.pi * k * t / length))
    )
    q, _ = np.linalg.qr(basis)
    return q


def temporal_project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    if array.shape[0] != q.shape[0]:
        raise ValueError("temporal basis and trace length disagree")
    return q @ (q.T @ array)


def isolate_harmonic(x: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered, mean = center_trace(x)
    projected = temporal_project(centered, harmonic_basis(len(centered), k))
    return projected, centered - projected, mean


def energy(x: np.ndarray) -> float:
    return float(np.square(np.asarray(x, dtype=np.float64)).sum())


def spectral_concentration(x: np.ndarray, k: int = 1) -> dict[str, float]:
    projected, residual, _ = isolate_harmonic(x, k)
    e1 = energy(projected)
    total = e1 + energy(residual)
    return {
        "e_k": e1,
        "e_total_non_dc": total,
        "concentration": e1 / max(total, EPS),
        "residual_energy": energy(residual),
    }


def exact_whiten_2d(y: np.ndarray, ridge: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(y, dtype=np.float64)
    covariance = coords.T @ coords / max(1, len(coords))
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, ridge)
    inverse_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    return coords @ inverse_sqrt, inverse_sqrt


def unwrap_with_orientation(angle: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(np.asarray(angle, dtype=np.float64))
    if len(unwrapped) > 1 and np.polyfit(np.arange(len(unwrapped)), unwrapped, 1)[0] < 0:
        unwrapped = -unwrapped
    return unwrapped


def circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def tautology_audit(x: np.ndarray) -> dict[str, float]:
    projected, _, _ = isolate_harmonic(x, 1)
    u, singular, _ = np.linalg.svd(projected, full_matrices=False)
    if len(singular) < 2 or singular[1] <= EPS:
        raise ValueError("k=1 reconstruction is rank deficient")
    coords = u[:, :2] * singular[:2]
    whitened, _ = exact_whiten_2d(coords)
    phase = unwrap_with_orientation(np.arctan2(whitened[:, 1], whitened[:, 0]))
    target = 2.0 * np.pi * np.arange(len(x)) / len(x)
    design = np.column_stack((target, np.ones(len(target))))
    slope, intercept = np.linalg.lstsq(design, phase, rcond=None)[0]
    fitted = design @ np.array([slope, intercept])
    ss_res = float(np.square(phase - fitted).sum())
    ss_tot = float(np.square(phase - phase.mean()).sum())
    aligned_target = target + np.mean(phase - target)
    circ = circular_difference(phase, aligned_target)
    covariance = coords.T @ coords / len(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / max(ss_tot, EPS),
        "circular_mae": float(np.mean(np.abs(circ))),
        "circular_resultant": float(np.abs(np.mean(np.exp(1j * circ)))),
        "eigengap_relative": float(
            (eigenvalues[0] - eigenvalues[1]) / max(eigenvalues[0], EPS)
        ),
    }


def detrend(x: np.ndarray, degree: int = 1) -> np.ndarray:
    centered, _ = center_trace(x)
    t = np.linspace(-1.0, 1.0, len(centered))
    design = np.column_stack([t**power for power in range(degree + 1)])
    coefficients = np.linalg.lstsq(design, centered, rcond=None)[0]
    return centered - design @ coefficients


def window_trace(x: np.ndarray, kind: str = "hann") -> np.ndarray:
    centered, _ = center_trace(x)
    if kind == "none":
        return centered
    if kind == "hann":
        weights = np.hanning(len(centered))
    elif kind == "sine":
        weights = np.sin(np.pi * (np.arange(len(centered)) + 0.5) / len(centered))
    else:
        raise ValueError(f"unknown window: {kind}")
    normalized = weights / np.sqrt(np.mean(weights**2))
    return centered * normalized[:, None]


def dct_basis(length: int, k: int = 1) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    vector = np.cos(np.pi * (t + 0.5) * k / length)[:, None]
    return vector / np.linalg.norm(vector)


def boundary_reflect(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    return np.concatenate((array, array[-2:0:-1]), axis=0)


@dataclass(frozen=True)
class NullSpec:
    name: str
    generator: Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _shuffle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(x)[rng.permutation(len(x))]


def _phase_randomized(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centered, mean = center_trace(x)
    spectrum = np.fft.rfft(centered, axis=0)
    phases = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    phases[0] = 0.0
    if len(x) % 2 == 0:
        phases[-1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    return np.fft.irfft(randomized, n=len(x), axis=0) + mean


def _random_walk(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centered, mean = center_trace(x)
    increments = np.diff(centered, axis=0)
    scales = np.std(increments, axis=0, ddof=1) if len(increments) > 1 else np.ones(x.shape[1])
    walk = np.vstack((np.zeros(x.shape[1]), np.cumsum(rng.normal(size=centered.shape[0:1] + centered.shape[1:]) * scales, axis=0)))
    return walk[: len(x)] + mean


def _linear_drift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(array))[:, None]
    drift = array[0] + t * (array[-1] - array[0])
    residual_scale = np.std(detrend(array, 1), axis=0)
    return drift + rng.normal(size=array.shape) * residual_scale


def _polynomial(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    t = np.linspace(-1.0, 1.0, len(array))[:, None]
    scales = np.std(array, axis=0)
    coefficients = rng.normal(size=(3, array.shape[1])) * scales[None, :]
    return coefficients[0] + coefficients[1] * t + coefficients[2] * t**2


def _ar1(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centered, mean = center_trace(x)
    numerator = float(np.sum(centered[1:] * centered[:-1]))
    denominator = float(np.sum(centered[:-1] ** 2))
    phi = float(np.clip(numerator / max(denominator, EPS), -0.98, 0.98))
    innovations = centered[1:] - phi * centered[:-1]
    scale = np.std(innovations, axis=0)
    output = np.zeros_like(centered)
    output[0] = rng.normal(size=centered.shape[1]) * np.std(centered, axis=0)
    for t in range(1, len(output)):
        output[t] = phi * output[t - 1] + rng.normal(size=output.shape[1]) * scale
    return output + mean


def _boundary_bridge(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64).copy()
    shift = rng.integers(1, len(array))
    return np.roll(array, int(shift), axis=0)


NULL_SPECS = (
    NullSpec("temporal_shuffle", _shuffle),
    NullSpec("phase_randomized", _phase_randomized),
    NullSpec("gaussian_random_walk", _random_walk),
    NullSpec("matched_linear_drift", _linear_drift),
    NullSpec("polynomial_trend", _polynomial),
    NullSpec("matched_ar1", _ar1),
    NullSpec("randomized_boundary", _boundary_bridge),
)


def null_audit(
    x: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    observed = spectral_concentration(x)
    rows.append({"null": "observed", "draw": 0, **observed})
    for spec in NULL_SPECS:
        for draw in range(draws):
            rows.append(
                {
                    "null": spec.name,
                    "draw": draw,
                    **spectral_concentration(spec.generator(x, rng)),
                }
            )
    return rows


def preprocessing_sensitivity(x: np.ndarray) -> list[dict[str, float | str]]:
    variants = {
        "raw_centered": center_trace(x)[0],
        "linear_detrend": detrend(x, 1),
        "quadratic_detrend": detrend(x, 2),
        "hann_window": window_trace(x, "hann"),
        "sine_window": window_trace(x, "sine"),
    }
    rows = [{"method": name, **spectral_concentration(value)} for name, value in variants.items()]
    centered, _ = center_trace(x)
    dct_projection = temporal_project(centered, dct_basis(len(centered), 1))
    rows.append(
        {
            "method": "nonperiodic_dct_k1",
            "e_k": energy(dct_projection),
            "e_total_non_dc": energy(centered),
            "concentration": energy(dct_projection) / max(energy(centered), EPS),
            "residual_energy": energy(centered - dct_projection),
        }
    )
    reflected = boundary_reflect(x)
    rows.append({"method": "reflected_boundary", **spectral_concentration(reflected)})
    return rows

