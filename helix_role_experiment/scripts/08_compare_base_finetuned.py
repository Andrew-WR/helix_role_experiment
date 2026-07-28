from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import write_csv
from helix_role_experiment.config import read_jsonl
from helix_role_experiment.fine_tuning import (
    category_durations,
    removed_content_categories,
)
from helix_role_experiment.plotting import line_svg, scatter_svg
from helix_role_experiment.subspaces import principal_angles, projector_similarity


REQUIRED = {"problem_id", "text", "correct", "token_count"}


def indexed_generations(path: str) -> dict[str, dict]:
    rows = read_jsonl(path)
    for row in rows:
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"{path} row {row.get('problem_id')} missing {sorted(missing)}")
    return {row["problem_id"]: row for row in rows}


def load_selected_planes(output_root: Path) -> dict[int, np.ndarray]:
    with (output_root / "models" / "subspace_index.json").open(
        "r", encoding="utf-8"
    ) as handle:
        index = json.load(handle)
    planes = {}
    for layer_text, estimator in index["selected_by_layer"].items():
        layer = int(layer_text)
        with np.load(
            output_root
            / "models"
            / f"subspace_layer_{layer}_{estimator}.npz",
            allow_pickle=False,
        ) as data:
            planes[layer] = np.asarray(data["basis"])
    return planes


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired base/fine-tuned trace analysis")
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--tuned-output", required=True)
    parser.add_argument("--base-generations", required=True)
    parser.add_argument("--tuned-generations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    base = indexed_generations(args.base_generations)
    tuned = indexed_generations(args.tuned_generations)
    common = sorted(set(base) & set(tuned))
    if not common:
        raise ValueError("base and tuned generation files share no problem IDs")
    paired = []
    category_rows = []
    duration_series: dict[str, tuple[list[float], list[float]]] = {}
    category_aggregate = defaultdict(lambda: [0, 0])
    for problem_id in common:
        left, right = base[problem_id], tuned[problem_id]
        removed = removed_content_categories(left["text"], right["text"])
        base_categories = category_durations(left["text"])
        tuned_categories = category_durations(right["text"])
        paired.append(
            {
                "problem_id": problem_id,
                "base_tokens": left["token_count"],
                "tuned_tokens": right["token_count"],
                "token_change": int(right["token_count"]) - int(left["token_count"]),
                "base_correct": int(bool(left["correct"])),
                "tuned_correct": int(bool(right["correct"])),
                "accuracy_change": int(bool(right["correct"])) - int(bool(left["correct"])),
                "removed_categories": removed,
            }
        )
        for category in sorted(set(base_categories) | set(tuned_categories) | set(removed)):
            base_count = base_categories.get(category, 0)
            tuned_count = tuned_categories.get(category, 0)
            category_aggregate[category][0] += base_count
            category_aggregate[category][1] += tuned_count
            category_rows.append(
                {
                    "problem_id": problem_id,
                    "category": category,
                    "base_sentences": base_count,
                    "tuned_sentences": tuned_count,
                    "removed_sentences": removed.get(category, 0),
                }
            )
    write_csv(tables / "paired_base_finetuned.csv", paired)
    write_csv(tables / "reasoning_category_durations.csv", category_rows)
    categories = sorted(category_aggregate)
    line_svg(
        figures / "12_base_vs_finetuned_category_duration.svg",
        {
            "base": (
                np.arange(len(categories)),
                np.asarray([category_aggregate[value][0] for value in categories]),
            ),
            "fine-tuned": (
                np.arange(len(categories)),
                np.asarray([category_aggregate[value][1] for value in categories]),
            ),
        },
        "Base versus fine-tuned reasoning-category duration",
        "category index: " + ", ".join(f"{i}={value}" for i, value in enumerate(categories)),
        "sentence count",
    )
    scatter_svg(
        figures / "13_accuracy_token_pareto.svg",
        np.asarray(
            [int(base[value]["token_count"]) for value in common]
            + [int(tuned[value]["token_count"]) for value in common]
        ),
        np.asarray(
            [int(bool(base[value]["correct"])) for value in common]
            + [int(bool(tuned[value]["correct"])) for value in common]
        ),
        ["base"] * len(common) + ["fine-tuned"] * len(common),
        "Paired accuracy-token frontier",
        "output tokens",
        "objective correctness",
    )
    base_planes = load_selected_planes(Path(args.base_output))
    tuned_planes = load_selected_planes(Path(args.tuned_output))
    alignment = []
    for layer in sorted(set(base_planes) & set(tuned_planes)):
        angles = principal_angles(base_planes[layer], tuned_planes[layer])
        alignment.append(
            {
                "layer": layer,
                "projector_similarity": projector_similarity(
                    base_planes[layer], tuned_planes[layer]
                ),
                "principal_angle_1_radians": float(angles[0]),
                "principal_angle_2_radians": float(angles[1]),
            }
        )
    write_csv(tables / "base_finetuned_plane_alignment.csv", alignment)
    print(f"Compared {len(common)} paired problems into {output}")


if __name__ == "__main__":
    main()
