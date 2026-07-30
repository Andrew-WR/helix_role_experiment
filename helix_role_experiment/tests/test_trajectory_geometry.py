import unittest

import numpy as np

from helix_role_experiment.trajectory_geometry import (
    design_matrix,
    fit_trajectory_models,
    match_norm,
    normalized_positions,
)


class TrajectoryGeometryTests(unittest.TestCase):
    def test_generalized_helix_recovers_synthetic_curve(self):
        progress = normalized_positions(120)
        design = design_matrix(
            "generalized_helix",
            progress,
            omega=1.5 * np.pi,
            radius_slope=0.375,
        )
        coefficients = np.asarray(
            [
                [0.2, -0.1, 0.4],
                [1.0, 0.5, -0.2],
                [0.7, -0.8, 0.3],
                [-0.2, 0.6, 0.9],
            ]
        )
        activations = design @ coefficients
        curves, rows = fit_trajectory_models(
            activations,
            ridge=1e-8,
            fold_count=5,
        )
        selected = next(
            row
            for row in rows
            if row["model"] == "generalized_helix"
            and row["selected_within_model"]
        )
        self.assertAlmostEqual(selected["turns"], 0.75)
        self.assertAlmostEqual(selected["radius_slope"], 0.375)
        prediction = curves["generalized_helix"].value(progress)
        self.assertLess(float(np.mean((prediction - activations) ** 2)), 1e-8)

    def test_local_k1_step_is_nonzero_and_norm_matched(self):
        progress = normalized_positions(80)
        activations = design_matrix(
            "linear_plus_closed_k1",
            progress,
        ) @ np.eye(4)
        curves, _ = fit_trajectory_models(
            activations,
            ridge=1e-8,
            fold_count=4,
        )
        helix_delta = curves["generalized_helix"].local_delta(0.2, 0.05)
        k1_delta = curves["linear_plus_closed_k1"].local_delta(0.2, 0.05)
        self.assertGreater(float(np.linalg.norm(k1_delta)), 0.0)
        matched = match_norm(k1_delta, helix_delta)
        self.assertAlmostEqual(
            float(np.linalg.norm(matched)),
            float(np.linalg.norm(helix_delta)),
        )


if __name__ == "__main__":
    unittest.main()
