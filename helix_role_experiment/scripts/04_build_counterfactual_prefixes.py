from __future__ import annotations

import argparse
from collections import Counter

from _common import write_csv
from helix_role_experiment.config import ensure_output_dirs, load_config, write_jsonl
from helix_role_experiment.controlled_tasks import generate_suite
from helix_role_experiment.counterfactuals import build_all_counterfactuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact-state crossed prefixes")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    problems = generate_suite(
        int(config["tasks"]["problems_per_family"]), int(config["study"]["seed"])
    )
    rows = build_all_counterfactuals(problems)
    write_jsonl(paths["root"] / "counterfactual_prefixes.jsonl", rows)
    counts = Counter((row["family"], row["condition"], row["exact_state_valid"]) for row in rows)
    summary = [
        {
            "family": family,
            "condition": condition,
            "exact_state_valid": valid,
            "count": count,
        }
        for (family, condition, valid), count in sorted(counts.items())
    ]
    write_csv(paths["tables"] / "counterfactual_summary.csv", summary)
    invalid = sum(not row["exact_state_valid"] for row in rows)
    print(f"Wrote {len(rows)} counterfactual prefixes ({invalid} excluded)")


if __name__ == "__main__":
    main()

