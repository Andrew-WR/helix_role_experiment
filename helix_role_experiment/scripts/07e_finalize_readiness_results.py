from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import write_csv
from helix_role_experiment.config import atomic_json, ensure_output_dirs, load_config, read_jsonl
from helix_role_experiment.steering_artifacts import (
    READINESS_STOP_REGEX,
    steering_run_identity,
    valid_steering_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge deferred HumanEval scores and apply commercial success gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-code", action="store_true", help="Diagnostic summaries only; cannot pass a cross-domain gate")
    parser.add_argument(
        "--replication-gates", action="append", default=[],
        help="readiness_success_gates.json from a different base model; repeat for multiple replications",
    )
    return parser.parse_args()


def load_code_scores(paths: dict[str, Path], condition: str) -> dict[str, dict]:
    source = paths["tables"] / f"humaneval_{condition}.jsonl_results.jsonl"
    if not source.exists():
        return {}
    return {
        str(row["task_id"]): {
            "strict": bool(row["passed"]),
            "functional": bool(row.get("functional_passed", row["passed"])),
            "format_valid": bool(row.get("format_valid", True)),
        }
        for row in read_jsonl(source)
    }


def metric(value: object, percent: bool = False) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number * 100:.1f}%" if percent else f"{number:.1f}"


def print_diagnostics(summaries: list[dict], gate_rows: list[dict]) -> None:
    print("\nCondition results", flush=True)
    print(
        "condition  domain   scored/tasks  strict_acc  functional_acc  mean_tokens",
        flush=True,
    )
    for row in summaries:
        print(
            f"{row['condition']:<10} {row['domain']:<8} "
            f"{row['scored_tasks']:>3}/{row['tasks']:<3}       "
            f"{metric(row['accuracy'], percent=True):>10}  "
            f"{metric(row['functional_accuracy'], percent=True):>14}  "
            f"{metric(row['mean_output_tokens']):>11}",
            flush=True,
        )
    print("\nWithin-model gates", flush=True)
    for row in gate_rows:
        role = "candidate" if row["candidate_method"] else "control"
        print(
            f"{row['condition']}: role={role}; "
            f"token_saving={metric(row['token_saving_fraction'], percent=True)}; "
            f"accuracy_gain={metric(row['accuracy_gain_fraction'], percent=True)}; "
            f"efficiency_gate={row['efficiency_gate']}; "
            f"accuracy_gate={row['accuracy_gate']}; "
            f"paired_tasks={row['paired_task_set_complete']}; "
            f"cross_domain_noninferiority={row['cross_domain_noninferiority']}; "
            f"within_model_success={row['within_model_success']}",
            flush=True,
        )


def commercial_status(gate_rows: list[dict]) -> str:
    candidate = next(
        (row for row in gate_rows if row.get("candidate_method")), None
    )
    if candidate is None:
        return "Commercial success gate: not evaluated (gated candidate is missing)"
    if not candidate["within_model_success"]:
        return (
            "Commercial success gate: FAIL — gated did not pass the "
            "within-model efficiency/accuracy and cross-domain requirements"
        )
    if not candidate["cross_model_replication"]:
        return (
            "Commercial success gate: NOT YET — gated passed within-model, "
            "but a second-model replication is required"
        )
    return "Commercial success gate: PASS gated"


def partition_steering_sources(
    sources: list[Path], fingerprint: str,
) -> tuple[list[Path], list[Path]]:
    compatible = [
        source for source in sources
        if valid_steering_artifact(source, fingerprint)
    ]
    stale = [source for source in sources if source not in compatible]
    return compatible, stale


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    identity = steering_run_identity(
        config,
        paths["models"] / "readiness_survival_probe.npz",
        READINESS_STOP_REGEX,
    )
    fingerprint = str(identity["steering_run_fingerprint"])
    observations = []
    for source in sorted((paths["traces"] / "readiness_baseline").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        if row["split"] == "test":
            observations.append({
                "task_id": row["task_id"], "domain": row["domain"], "condition": "baseline",
                "tokens": row["output_token_count"], "correct": row["math_correct"],
                "functional_correct": row["math_correct"],
            })
    steering_sources = sorted(
        (paths["traces"] / "readiness_steering").glob("*.json")
    )
    steering_sources, stale = partition_steering_sources(
        steering_sources, fingerprint
    )
    if stale:
        print(
            f"Ignoring {len(stale)} steering traces generated by an old probe "
            "or configuration.",
            flush=True,
        )
    for source in steering_sources:
        row = json.loads(source.read_text(encoding="utf-8"))
        observations.append({
            "task_id": row["task_id"], "domain": row["domain"], "condition": row["condition"],
            "tokens": row["output_token_count"], "correct": row["math_correct"],
            "functional_correct": row["math_correct"],
        })
    conditions = sorted({row["condition"] for row in observations})
    missing = []
    for condition in conditions:
        scores = load_code_scores(paths, condition)
        for row in observations:
            if row["condition"] == condition and row["domain"] == "code":
                if row["task_id"] in scores:
                    score = scores[row["task_id"]]
                    row["correct"] = score["strict"]
                    row["functional_correct"] = score["functional"]
                else:
                    missing.append((condition, row["task_id"]))
    if missing and not args.allow_missing_code:
        examples = ", ".join(f"{condition}/{task}" for condition, task in missing[:4])
        raise RuntimeError(
            "HumanEval remains deferred. In an isolated Kaggle session, run "
            "`python scripts/evaluate_humaneval_subset.py --config <config>` "
            f"and then rerun 07e. Missing: {examples}"
        )
    summaries = []
    for condition in conditions:
        for domain in ("math", "code", "overall"):
            selected = [row for row in observations if row["condition"] == condition and (domain == "overall" or row["domain"] == domain)]
            scored = [row for row in selected if row["correct"] is not None]
            functional_scored = [
                row for row in selected if row["functional_correct"] is not None
            ]
            summaries.append({
                "condition": condition, "domain": domain, "tasks": len(selected),
                "scored_tasks": len(scored), "accuracy": float(np.mean([row["correct"] for row in scored])) if scored else None,
                "functional_accuracy": (
                    float(np.mean([
                        row["functional_correct"] for row in functional_scored
                    ])) if functional_scored else None
                ),
                "mean_output_tokens": float(np.mean([row["tokens"] for row in selected])) if selected else None,
            })
    baseline = next(row for row in summaries if row["condition"] == "baseline" and row["domain"] == "overall")
    gate_rows = []
    for row in summaries:
        if row["domain"] != "overall" or row["condition"] == "baseline":
            continue
        paired_task_set = all(
            {
                value["task_id"] for value in observations
                if value["condition"] == row["condition"]
                and value["domain"] == domain
            }
            == {
                value["task_id"] for value in observations
                if value["condition"] == "baseline"
                and value["domain"] == domain
            }
            for domain in ("math", "code")
        )
        complete = (
            paired_task_set
            and row["scored_tasks"] == row["tasks"]
            and baseline["scored_tasks"] == baseline["tasks"]
        )
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
            "paired_task_set_complete": paired_task_set,
            "within_model_success": within_model,
            "candidate_method": row["condition"] == "gated",
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
            row["candidate_method"]
            and row["within_model_success"]
            and row["cross_model_replication"]
        )
    write_csv(paths["tables"] / "readiness_condition_summary.csv", summaries)
    write_csv(paths["tables"] / "readiness_success_gates.csv", gate_rows)
    atomic_json(paths["tables"] / "readiness_success_gates.json", gate_rows)
    print_diagnostics(summaries, gate_rows)
    print("\n" + commercial_status(gate_rows), flush=True)


if __name__ == "__main__":
    main()
