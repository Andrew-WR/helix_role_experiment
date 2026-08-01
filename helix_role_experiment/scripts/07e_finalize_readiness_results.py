from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import write_csv
from helix_role_experiment.config import atomic_json, ensure_output_dirs, load_config, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge deferred HumanEval scores and apply commercial success gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-code", action="store_true", help="Diagnostic summaries only; cannot pass a cross-domain gate")
    parser.add_argument(
        "--replication-gates", action="append", default=[],
        help="readiness_success_gates.json from a different base model; repeat for multiple replications",
    )
    return parser.parse_args()


def load_code_scores(paths: dict[str, Path], condition: str) -> dict[str, bool]:
    source = paths["tables"] / f"humaneval_{condition}.jsonl_results.jsonl"
    if not source.exists():
        return {}
    return {str(row["task_id"]): bool(row["passed"]) for row in read_jsonl(source)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    observations = []
    for source in sorted((paths["traces"] / "readiness_baseline").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        if row["split"] == "test":
            observations.append({
                "task_id": row["task_id"], "domain": row["domain"], "condition": "baseline",
                "tokens": row["output_token_count"], "correct": row["math_correct"],
            })
    for source in sorted((paths["traces"] / "readiness_steering").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        observations.append({
            "task_id": row["task_id"], "domain": row["domain"], "condition": row["condition"],
            "tokens": row["output_token_count"], "correct": row["math_correct"],
        })
    conditions = sorted({row["condition"] for row in observations})
    missing = []
    for condition in conditions:
        scores = load_code_scores(paths, condition)
        for row in observations:
            if row["condition"] == condition and row["domain"] == "code":
                if row["task_id"] in scores:
                    row["correct"] = scores[row["task_id"]]
                else:
                    missing.append((condition, row["task_id"]))
    if missing and not args.allow_missing_code:
        examples = ", ".join(f"{condition}/{task}" for condition, task in missing[:4])
        raise RuntimeError(
            "HumanEval remains deferred. In an isolated evaluator, run "
            "`evaluate_functional_correctness humaneval_<condition>.jsonl` for each condition, "
            f"copy the *_results.jsonl files back to tables, then rerun. Missing: {examples}"
        )
    summaries = []
    for condition in conditions:
        for domain in ("math", "code", "overall"):
            selected = [row for row in observations if row["condition"] == condition and (domain == "overall" or row["domain"] == domain)]
            scored = [row for row in selected if row["correct"] is not None]
            summaries.append({
                "condition": condition, "domain": domain, "tasks": len(selected),
                "scored_tasks": len(scored), "accuracy": float(np.mean([row["correct"] for row in scored])) if scored else None,
                "mean_output_tokens": float(np.mean([row["tokens"] for row in selected])) if selected else None,
            })
    baseline = next(row for row in summaries if row["condition"] == "baseline" and row["domain"] == "overall")
    gate_rows = []
    for row in summaries:
        if row["domain"] != "overall" or row["condition"] == "baseline":
            continue
        complete = row["scored_tasks"] == row["tasks"] and baseline["scored_tasks"] == baseline["tasks"]
        token_saving = 1.0 - row["mean_output_tokens"] / baseline["mean_output_tokens"]
        accuracy_gain = (row["accuracy"] - baseline["accuracy"]) if complete else None
        efficient = complete and token_saving >= 0.10 and accuracy_gain >= -0.01
        accurate = complete and accuracy_gain >= 0.05 and row["mean_output_tokens"] <= baseline["mean_output_tokens"]
        # Generalization is explicit: the same condition must not regress either domain by >1 pp.
        domains_pass = True
        for domain in ("math", "code"):
            base_domain = next(value for value in summaries if value["condition"] == "baseline" and value["domain"] == domain)
            candidate = next(value for value in summaries if value["condition"] == row["condition"] and value["domain"] == domain)
            domains_pass &= candidate["accuracy"] is not None and base_domain["accuracy"] is not None and candidate["accuracy"] >= base_domain["accuracy"] - 0.01
        within_model = bool((efficient or accurate) and domains_pass)
        gate_rows.append({
            "condition": row["condition"], "token_saving_fraction": token_saving,
            "accuracy_gain_fraction": accuracy_gain, "efficiency_gate": efficient,
            "accuracy_gate": accurate, "cross_domain_noninferiority": domains_pass,
            "within_model_success": within_model,
            "cross_model_replication": False,
            "commercial_success": False,
        })
    replication_rows = []
    for source in args.replication_gates:
        replication_rows.extend(json.loads(Path(source).read_text(encoding="utf-8")))
    for row in gate_rows:
        row["cross_model_replication"] = any(
            other.get("condition") == row["condition"]
            and bool(other.get("within_model_success"))
            for other in replication_rows
        )
        row["commercial_success"] = bool(
            row["within_model_success"] and row["cross_model_replication"]
        )
    write_csv(paths["tables"] / "readiness_condition_summary.csv", summaries)
    write_csv(paths["tables"] / "readiness_success_gates.csv", gate_rows)
    atomic_json(paths["tables"] / "readiness_success_gates.json", gate_rows)
    passed = [row["condition"] for row in gate_rows if row["commercial_success"]]
    print(f"Commercial success gate: {'PASS ' + ', '.join(passed) if passed else 'not passed (a second-model replication is required)'}")


if __name__ == "__main__":
    main()
