import unittest

import numpy as np

from helix_role_experiment.eos_controls import (
    orthogonalize_to_direction,
    subspace_direction_overlap,
)
from helix_role_experiment.interventions import (
    donor_transplant,
    within_plane_rotation,
)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.plane, _ = np.linalg.qr(rng.normal(size=(23, 2)))
        self.center = rng.normal(size=23)
        self.target = rng.normal(size=23)
        self.donor = rng.normal(size=23)

    def test_rotation_preserves_orthogonal_residual_and_radius(self):
        changed = within_plane_rotation(
            self.target, self.plane, self.center, delta=0.7
        )
        orthogonal = np.eye(23) - self.plane @ self.plane.T
        np.testing.assert_allclose(
            (changed - self.center) @ orthogonal,
            (self.target - self.center) @ orthogonal,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            np.linalg.norm((changed - self.center) @ self.plane),
            np.linalg.norm((self.target - self.center) @ self.plane),
            places=12,
        )

    def test_donor_transplant_sets_only_plane_coordinates(self):
        changed = donor_transplant(self.target, self.donor, self.plane, self.center)
        np.testing.assert_allclose(
            (changed - self.center) @ self.plane,
            (self.donor - self.center) @ self.plane,
            atol=1e-12,
        )

    def test_eos_orthogonalization(self):
        rng = np.random.default_rng(44)
        outside = rng.normal(size=23)
        outside -= (outside @ self.plane) @ self.plane.T
        direction = self.plane[:, 0] + outside
        no_eos = orthogonalize_to_direction(self.plane, direction)
        self.assertLess(subspace_direction_overlap(no_eos, direction), 1e-12)

    def test_eos_orthogonalization_reports_rank_collapse(self):
        direction = self.plane[:, 0] + 0.2 * self.plane[:, 1]
        with self.assertRaisesRegex(ValueError, "rank 1"):
            orthogonalize_to_direction(self.plane, direction)


if __name__ == "__main__":
    unittest.main()
