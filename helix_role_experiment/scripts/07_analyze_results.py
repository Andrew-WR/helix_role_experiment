from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import read_csv, write_csv
from helix_role_experiment.config import ensure_output_dirs, load_config, seed_everything
from helix_role_experiment.plotting import (
    heatmap_svg,
    line_svg,
    paired_effect_svg,
    scatter_svg,
)
from helix_role_experiment.statistics import (
    grouped_bootstrap_mean,
    predictor_block_comparison,
)


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def paired_coordinate_displacements(rows, treatment, control):
    effects, problem_ids = [], []
    problems = sorted({row["problem_id"] for row in rows})
    for problem_id in problems:
        problem_rows = [row for row in rows if row["problem_id"] == problem_id]
        treated = [
            (float(row["coordinate_1"]), float(row["coordinate_2"]))
            for row in problem_rows
            if row["condition"] == treatment
        ]
        controlled = [
            (float(row["coordinate_1"]), float(row["coordinate_2"]))
            for row in problem_rows
            if row["condition"] == control
        ]
        if treated and controlled:
            left = np.mean(np.asarray(treated), axis=0)
            right = np.mean(np.asarray(controlled), axis=0)
            effects.append(float(np.linalg.norm(left - right)))
            problem_ids.append(problem_id)
    return np.asarray(effects), np.asarray(problem_ids)


def write_unavailable_figure(path: Path, title: str, reason: str) -> None:
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;")
    safe_reason = reason.replace("&", "&amp;").replace("<", "&lt;")
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="380" y="70" text-anchor="middle" font-family="Arial" font-size="20">{safe_title}</text>'
        f'<text x="380" y="155" text-anchor="middle" font-family="Arial" font-size="15">{safe_reason}</text>'
        '<text x="380" y="195" text-anchor="middle" font-family="Arial" font-size="13">'
        "No values are fabricated.</text></svg>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped analysis and required figures")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    rng = seed_everything(int(config["study"]["seed"]) + 707)
    observational = read_csv(paths["tables"] / "observational_cross.csv")
    causal = read_csv(paths["tables"] / "causal_interventions.csv")
    subspace = read_csv(paths["tables"] / "shared_subspace_evaluation.csv")

    selected = [row for row in subspace if row["estimator"] != "random_plane"]
    scatter_svg(
        paths["figures"] / "04_shared_subspace_heldout_energy_by_layer.svg",
        numeric(selected, "layer"),
        numeric(selected, "test_selectivity_mean"),
        [row["estimator"] for row in selected],
        "Held-out first-harmonic selectivity by layer",
        "layer",
        "k=1 / residual energy",
    )
    conditions = {"concise", "verbose_paraphrase", "loop", "teleport", "rollback"}
    trajectories = [row for row in observational if row["condition"] in conditions]
    scatter_svg(
        paths["figures"] / "05_raw_2d_counterfactual_trajectories.svg",
        numeric(trajectories, "coordinate_1"),
        numeric(trajectories, "coordinate_2"),
        [row["condition"] for row in trajectories],
        "Frozen-plane coordinates under progress-position conflicts",
        "whitened coordinate 1",
        "whitened coordinate 2",
    )
    series = {
        "token position proxy": (
            numeric(observational, "token_count_proxy"),
            numeric(observational, "raw_angle"),
        ),
        "structural progress": (
            numeric(observational, "structural_progress"),
            numeric(observational, "raw_angle"),
        ),
        "EOS logit": (
            numeric(observational, "eos_logit"),
            numeric(observational, "raw_angle"),
        ),
        "confidence": (
            numeric(observational, "confidence"),
            numeric(observational, "raw_angle"),
        ),
    }
    # These predictors have different scales, so Figure 6 uses four panels
    # represented as separate color-coded line clouds after sorting each x.
    sorted_series = {
        key: (x[np.argsort(x)], y[np.argsort(x)]) for key, (x, y) in series.items()
    }
    line_svg(
        paths["figures"] / "06_candidate_state_vs_competing_variables.svg",
        sorted_series,
        "Candidate angle versus competing variables (descriptive)",
        "predictor-specific scale",
        "candidate raw angle",
    )

    effect_specs = [
        ("progress", "teleport", "concise"),
        ("position", "verbose_paraphrase", "concise"),
        ("termination", "complete_answer_allowed", "complete_answer_forbidden"),
    ]
    effect_rows = []
    estimates, lowers, uppers, labels = [], [], [], []
    for label, treatment, control in effect_specs:
        effects, ids = paired_coordinate_displacements(
            observational, treatment, control
        )
        if len(ids) >= 2:
            result = grouped_bootstrap_mean(
                effects,
                ids,
                int(config["analysis"]["bootstrap_draws"]),
                rng,
            )
            effect_rows.append({"contrast": label, **result})
            labels.append(label)
            estimates.append(result["estimate"])
            lowers.append(result["ci_lower"])
            uppers.append(result["ci_upper"])
    write_csv(paths["tables"] / "paired_counterfactual_effects.csv", effect_rows)
    paired_effect_svg(
        paths["figures"] / "07_progress_position_crossed_effects.svg",
        labels,
        np.asarray(estimates),
        np.asarray(lowers),
        np.asarray(uppers),
        "Matched progress, position, and termination effects",
    )
    event_rows = [
        row for row in observational if row["condition"] in {"teleport", "rollback", "loop"}
    ]
    scatter_svg(
        paths["figures"] / "08_rollback_teleport_events.svg",
        numeric(event_rows, "structural_progress"),
        numeric(event_rows, "raw_angle"),
        [row["condition"] for row in event_rows],
        "Rollback and teleport event states",
        "exact structural progress",
        "candidate raw angle",
    )

    y = np.column_stack(
        (numeric(observational, "coordinate_1"), numeric(observational, "coordinate_2"))
    )
    operation_columns = [
        key for key in observational[0] if key.startswith("operation_")
    ]
    blocks = {
        "position": np.column_stack(
            (
                numeric(observational, "token_count_proxy"),
                numeric(observational, "sentence_count_proxy"),
            )
        ),
        "termination": np.column_stack(
            (
                numeric(observational, "eos_logit"),
                numeric(observational, "termination_allowed"),
            )
        ),
        "procedure": np.column_stack(
            [numeric(observational, key) for key in operation_columns]
        ),
        "confidence": numeric(observational, "confidence")[:, None],
        "semantic": np.column_stack(
            (
                numeric(observational, "structural_progress"),
                numeric(observational, "remaining_distance"),
            )
        ),
    }
    block_rows = predictor_block_comparison(
        blocks,
        y,
        np.asarray([row["problem_id"] for row in observational]),
        int(config["analysis"]["cv_folds"]),
        float(config["analysis"]["ridge"]),
        int(config["study"]["seed"]),
    )
    write_csv(paths["tables"] / "predictor_block_comparison.csv", block_rows)
    scatter_svg(
        paths["figures"] / "09_predictor_unique_contributions.svg",
        np.arange(len(block_rows)),
        np.asarray([float(row["partial_r2"]) if np.isfinite(float(row["partial_r2"])) else 0.0 for row in block_rows]),
        [row["added_block"] for row in block_rows],
        "Incremental held-out contribution by predictor block",
        "pre-registered block order",
        "partial R²",
    )

    layers = sorted({int(row["layer"]) for row in causal})
    controls = sorted({row["control"] for row in causal})
    matrix = np.full((len(layers), len(controls)), np.nan)
    for i, layer in enumerate(layers):
        for j, control in enumerate(controls):
            values = [
                float(row["observed_progress_shift"])
                for row in causal
                if int(row["layer"]) == layer and row["control"] == control
            ]
            if values:
                matrix[i, j] = np.mean(values)
    heatmap_svg(
        paths["figures"] / "10_layer_by_intervention_effect.svg",
        matrix,
        [str(layer) for layer in layers],
        controls,
        "Layer-by-intervention progress-shift proxy",
    )
    eos_rows = [
        row
        for row in causal
        if row["control"] in {"candidate_transplant", "eos_orthogonal_transplant", "norm_matched_random"}
    ]
    scatter_svg(
        paths["figures"] / "11_eos_matched_intervention_comparison.svg",
        numeric(eos_rows, "eos_logit_change_proxy"),
        numeric(eos_rows, "causal_abstraction_direction_accuracy"),
        [row["control"] for row in eos_rows],
        "EOS overlap/control comparison under fixed length",
        "EOS-logit change proxy",
        "causal-abstraction direction accuracy",
    )
    write_unavailable_figure(
        paths["figures"] / "12_base_vs_finetuned_category_duration.svg",
        "Base versus helix-fine-tuned reasoning categories",
        "Fine-tuned checkpoint and paired traces were not supplied.",
    )
    write_unavailable_figure(
        paths["figures"] / "13_accuracy_token_pareto.svg",
        "Accuracy-token Pareto frontier",
        "Objective base/fine-tuned evaluations were not supplied.",
    )
    print(f"Wrote analysis tables and 13 figure slots to {paths['figures']}")


if __name__ == "__main__":
    main()
