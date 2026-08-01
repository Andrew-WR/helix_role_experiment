from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from helix_role_experiment.behavioral import final_answer_is_correct
from helix_role_experiment.config import atomic_json, deterministic_id, ensure_output_dirs, load_config, read_jsonl, seed_everything
from helix_role_experiment.models import huggingface_collector_from_config
from helix_role_experiment.readiness import ExponentialProbe, SentenceSteeringController
from helix_role_experiment.reasoning_benchmarks import ReadinessTask, extract_humaneval_completion, readiness_prompt

from _common import write_csv


STOP_REGEX = (
    r"(?is)</think>\s*(?:FINAL:\s*\S[^\r\n]*(?:\r?\n|<\|im_end\|>)|"
    r"FINAL_CODE:\s*```(?:python|py)?.*?```)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gated readiness steering and controls")
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args()


def worker(args: argparse.Namespace, config: dict, paths: dict[str, Path]) -> None:
    rows = [row for row in read_jsonl(paths["tables"] / "readiness_tasks.jsonl") if row["split"] == "test"]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    backend = huggingface_collector_from_config(config["model"], config["collection"])
    probe = ExponentialProbe.load(paths["models"] / "readiness_survival_probe.npz")
    intervention = config["intervention"]
    conditions = list(intervention.get("conditions", ["gated", "always", "random"]))
    batch_size = int(config["collection"].get("batch_size", 2))
    output_dir = paths["traces"] / "readiness_steering"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(config["study"]["seed"])
    for condition in conditions:
        pending = []
        for row in rows:
            trace_id = deterministic_id(row["task_id"], condition, config["study"]["seed"])
            destination = output_dir / f"{trace_id}.json"
            if not destination.exists():
                pending.append((trace_id, row, destination))
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            controllers = [
                SentenceSteeringController(
                    backend.tokenizer, probe,
                    alpha=float(intervention.get("alpha", 0.5)),
                    pulse_tokens=int(intervention.get("pulse_tokens", 4)),
                    mode=condition,
                    random_seed=int(deterministic_id(row["task_id"], "random")[:8], 16),
                ) for _, row, _ in chunk
            ]
            seeds = [seed_base + int(deterministic_id(row["task_id"], "paired")[:8], 16) % 1_000_000 for _, row, _ in chunk]
            generations = backend.collect_batch(
                [readiness_prompt(ReadinessTask.from_dict(row)) for _, row, _ in chunk],
                [probe.layer], int(config["collection"]["max_new_tokens"]), seeds,
                temperature=float(config["collection"].get("temperature", 0.6)),
                top_p=float(config["collection"].get("top_p", 0.95)),
                top_k=int(config["collection"].get("top_k", 20)),
                interventions=controllers, capture_activations=False,
                capture_eos_logits=True,
                capture_token_entropies=bool(config["collection"].get("capture_token_entropies", False)),
                stop_regex=STOP_REGEX,
                stop_check_interval=int(config["collection"].get("stop_check_interval", 8)),
            )
            for (trace_id, row, destination), controller, generation in zip(chunk, controllers, generations, strict=True):
                atomic_json(destination, {
                    "trace_id": trace_id, "task_id": row["task_id"], "domain": row["domain"],
                    "split": "test", "condition": condition, "text": generation.text,
                    "output_token_count": len(generation.token_ids), "reached_eos": generation.reached_eos,
                    "math_correct": (
                        final_answer_is_correct(generation.text, row["reference_answer"])
                        if row["domain"] == "math" else None
                    ),
                    "humaneval_completion": (
                        extract_humaneval_completion(generation.text) if row["domain"] == "code" else None
                    ),
                    "trigger_count": controller.trigger_count,
                    "steered_steps": controller.steered_steps, "readiness_scores": controller.scores,
                    "mean_eos_logit": sum(generation.eos_logits) / max(len(generation.eos_logits), 1),
                    "mean_token_entropy": (
                        sum(generation.token_entropies) / len(generation.token_entropies)
                        if generation.token_entropies else None
                    ),
                })
    atomic_json(paths["logs"] / f"07d_worker_{args.shard_index}.json", {"assigned": len(rows), "conditions": conditions})


def export_humaneval(paths: dict[str, Path]) -> None:
    steering = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((paths["traces"] / "readiness_steering").glob("*.json"))]
    baseline = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((paths["traces"] / "readiness_baseline").glob("*.json"))]
    rows = []
    for row in baseline:
        if row["split"] == "test" and row["domain"] == "code":
            rows.append({"condition": "baseline", "task_id": row["task_id"], "completion": row["humaneval_completion"]})
    for row in steering:
        if row["domain"] == "code":
            rows.append({"condition": row["condition"], "task_id": row["task_id"], "completion": row["humaneval_completion"]})
    for condition in sorted({row["condition"] for row in rows}):
        write_csv(paths["tables"] / f"humaneval_{condition}_display.csv", [row for row in rows if row["condition"] == condition])
        destination = paths["tables"] / f"humaneval_{condition}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["condition"] == condition:
                    handle.write(json.dumps({"task_id": row["task_id"], "completion": row["completion"]}, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    seed_everything(int(config["study"]["seed"]))
    if args.worker:
        worker(args, config, paths)
        return
    processes = []
    for shard in range(2):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        processes.append(subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()), "--config", args.config,
            "--worker", "--shard-index", str(shard), "--num-shards", "2",
        ], env=environment))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"steering workers failed: {codes}")
    export_humaneval(paths)
    print("Steering complete. HumanEval samples were exported without executing generated code.")


if __name__ == "__main__":
    main()
