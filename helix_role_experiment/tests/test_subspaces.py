import unittest

import numpy as np

from helix_role_experiment.subspaces import (
    fit_complex_coefficient_model,
    generalized_spectral_plane,
    generalized_spectral_plane_iterative,
    grassmann_mean,
    principal_angles,
    projector_similarity,
    projected_features,
    trace_harmonic_plane,
)


class SubspaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.normal(size=(18, 2)))
        self.plane = q
        self.rng = rng

    def make_trace(self, length, phase=0.0, scales=(2.0, 0.7)):
        t = np.arange(length)
        coords = np.column_stack(
            (
                scales[0] * np.cos(2 * np.pi * t / length + phase),
                scales[1] * np.sin(2 * np.pi * t / length + phase),
            )
        )
        return coords @ self.plane.T

    def test_grassmann_ignores_frame_gauge(self):
        frames = []
        for _ in range(20):
            rotation, _ = np.linalg.qr(self.rng.normal(size=(2, 2)))
            frames.append(self.plane @ rotation)
        estimate, _ = grassmann_mean(frames)
        self.assertGreater(projector_similarity(estimate, self.plane), 1 - 1e-12)

    def test_trace_plane_recovery(self):
        estimate, _ = trace_harmonic_plane(self.make_trace(73))
        self.assertLess(np.max(principal_angles(estimate, self.plane)), 1e-7)

    def test_complex_model_recovers_shared_plane_under_phase_gauge(self):
        traces = [
            self.make_trace(64 + index, phase=float(self.rng.uniform(-np.pi, np.pi)))
            for index in range(30)
        ]
        model = fit_complex_coefficient_model(traces, complex_rank=2)
        self.assertGreater(projector_similarity(model.real_plane, self.plane), 0.99)

    def test_iterative_generalized_solver_matches_dense_solver(self):
        traces = [
            self.make_trace(48 + index)
            + self.rng.normal(scale=0.08, size=(48 + index, 18))
            for index in range(8)
        ]
        dense, _ = generalized_spectral_plane(traces, ridge=0.01)
        iterative, _, iterations = generalized_spectral_plane_iterative(
            traces, ridge=0.01, tolerance=1e-6, max_iterations=500
        )
        self.assertLess(iterations, 500)
        self.assertGreater(projector_similarity(dense, iterative), 0.999)

    def test_low_radius_marks_phase_uncertain(self):
        trace = np.zeros((5, 18))
        trace[2] = self.plane[:, 0] * 2
        features = projected_features(
            trace,
            self.plane,
            np.zeros(18),
            np.eye(2),
            radius_threshold=0.5,
        )
        self.assertFalse(features["phase_reliable"][0])
        self.assertTrue(features["phase_reliable"][2])
        self.assertTrue(np.isnan(features["unwrapped_angle"][0]))


if __name__ == "__main__":
    unittest.main()
