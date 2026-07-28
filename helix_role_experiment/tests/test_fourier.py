import unittest

import numpy as np

from helix_role_experiment.fourier import (
    harmonic_basis,
    spectral_concentration,
    tautology_audit,
    temporal_project,
)


class FourierTests(unittest.TestCase):
    def test_exact_ellipse_phase_is_normalized_position(self):
        length, dimension = 127, 11
        rng = np.random.default_rng(1)
        a = rng.normal(size=dimension)
        b = rng.normal(size=dimension)
        t = np.arange(length)
        trace = (
            np.cos(2 * np.pi * t / length)[:, None] * a
            + np.sin(2 * np.pi * t / length)[:, None] * b
            + rng.normal(size=dimension)
        )
        result = tautology_audit(trace)
        self.assertAlmostEqual(result["slope"], 1.0, places=10)
        self.assertGreater(result["r2"], 1 - 1e-12)
        self.assertLess(result["circular_mae"], 1e-10)

    def test_harmonic_projection_is_idempotent(self):
        rng = np.random.default_rng(2)
        trace = rng.normal(size=(61, 9))
        basis = harmonic_basis(len(trace), 1)
        once = temporal_project(trace, basis)
        twice = temporal_project(once, basis)
        np.testing.assert_allclose(once, twice, atol=1e-12)

    def test_pure_k1_has_unit_concentration(self):
        basis = harmonic_basis(64, 1)
        trace = basis @ np.array([[2.0, 0.5, -1.0], [1.0, 3.0, 0.2]])
        result = spectral_concentration(trace)
        self.assertAlmostEqual(result["concentration"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()

