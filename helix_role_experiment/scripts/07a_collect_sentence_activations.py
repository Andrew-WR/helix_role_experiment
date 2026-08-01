from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import parse_layer_spec
from helix_role_experiment.behavioral import final_answer_is_correct
from helix_role_experiment.config import (
    atomic_json, config_hash, deterministic_id, ensure_output_dirs,
    environment_record, load_config, seed_everything, write_jsonl,
)
from helix_role_experiment.models import huggingface_collector_from_config
from helix_role_experiment.readiness import assign_group_splits, sentence_boundaries
from helix_role_experiment.reasoning_benchmarks import (
    extract_humaneval_completion, load_mixed_readiness_tasks, readiness_prompt,
)


STOP_REGEX = (
    r"(?is)</think>\s*(?:FINAL:\s*\S[^\r\n]*(?:\r?\n|<\|im_end\|>)|"
    r"FINAL_CODE:\s*```(?:python|py)?.*?```)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sentence-boundary activations on two GPUs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def prepare_tasks(config: dict, paths: dict[str, Path]) -> list[dict]:
    task_config = config["tasks"]
    tasks = load_mixed_readiness_tasks(
        int(task_config["math500_count"]), int(task_config["humaneval_count"]),
        int(config["study"]["seed"]), task_config["math500_path"],
        task_config["humaneval_path"], set(task_config.get("math_levels", [1, 2, 3, 4, 5])),
    )
    rows = [task.to_dict() for task in tasks]
    split = assign_group_splits(rows, int(config["study"]["seed"]))
    for row in rows:
        row["split"] = split[row["task_id"]]
    write_jsonl(paths["tables"] / "readiness_tasks.jsonl", rows)
    return rows


def worker(args: argparse.Namespace, config: dict, paths: dict[str, Path]) -> None:
    from helix_role_experiment.config import read_jsonl

    rows = read_jsonl(paths["tables"] / "readiness_tasks.jsonl")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    backend = huggingface_collector_from_config(config["model"], config["collection"])
    layers = parse_layer_spec(str(config["collection"]["layers"]), list(range(len(backend.layers))))
    batch_size = int(config["collection"].get("batch_size", 2))
    trace_dir = paths["traces"] / "readiness_baseline"
    activation_dir = paths["traces"] / "readiness_activations"
    trace_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["study"]["seed"])
    pending = []
    for index, row in enumerate(rows):
        trace_id = deterministic_id(config_hash(config), row["task_id"], "baseline")
        if not (trace_dir / f"{trace_id}.json").exists():
            pending.append((index, trace_id, row))
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        generations = backend.collect_batch(
            [readiness_prompt_from_row(row) for _, _, row in chunk], layers,
            int(config["collection"]["max_new_tokens"]),
            [seed + args.shard_index * 100_000 + index for index, _, _ in chunk],
            temperature=float(config["collection"].get("temperature", 0.6)),
            top_p=float(config["collection"].get("top_p", 0.95)),
            top_k=int(config["collection"].get("top_k", 20)),
            capture_activations=True, capture_eos_logits=True,
            capture_token_entropies=bool(config["collection"].get("capture_token_entropies", False)),
            stop_regex=STOP_REGEX,
            stop_check_interval=int(config["collection"].get("stop_check_interval", 8)),
        )
        for (_, trace_id, row), generation in zip(chunk, generations, strict=True):
            boundaries = sentence_boundaries(
                backend.tokenizer, generation.text, generation.token_ids,
                generation.eos_logits, generation.token_entropies,
            )
            valid = [value for value in boundaries if value.activation_index < len(generation.token_ids)]
            arrays = {
                f"layer_{layer}": generation.activations_by_layer[layer][
                    [value.activation_index for value in valid]
                ].astype(np.float16)
                for layer in layers
            }
            npz_path = activation_dir / f"{trace_id}.npz"
            temporary = npz_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(npz_path)
            correct = (
                final_answer_is_correct(generation.text, row["reference_answer"])
                if row["domain"] == "math" else None
            )
            atomic_json(trace_dir / f"{trace_id}.json", {
                "trace_id": trace_id, "task_id": row["task_id"],
                "domain": row["domain"], "split": row["split"],
                "prompt": row["prompt"], "reference_answer": row["reference_answer"],
                "metadata": row.get("metadata", {}), "text": generation.text,
                "prompt_token_count": generation.prompt_token_count,
                "output_token_count": len(generation.token_ids),
                "reached_eos": generation.reached_eos, "layers": layers,
                "sentences": [value.to_dict() for value in valid],
                "math_correct": correct,
                "humaneval_completion": (
                    extract_humaneval_completion(generation.text) if row["domain"] == "code" else None
                ),
                "activation_file": str(npz_path),
            })
    atomic_json(paths["logs"] / f"07a_worker_{args.shard_index}.json", {
        "shard_index": args.shard_index, "assigned": len(rows), "completed": len(rows)
    })


def readiness_prompt_from_row(row: dict) -> str:
    from helix_role_experiment.reasoning_benchmarks import ReadinessTask
    return readiness_prompt(ReadinessTask.from_dict(row))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    seed_everything(int(config["study"]["seed"]))
    if args.worker:
        worker(args, config, paths)
        return
    atomic_json(paths["root"] / "environment.json", environment_record(config))
    rows = prepare_tasks(config, paths)
    if args.limit is not None:
        rows = rows[: args.limit]
        write_jsonl(paths["tables"] / "readiness_tasks.jsonl", rows)
    processes = []
    for shard in range(2):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--config", args.config,
            "--worker", "--shard-index", str(shard), "--num-shards", "2",
        ]
        if args.limit is not None:
            command.extend(["--limit", str((args.limit + 1) // 2)])
        processes.append(subprocess.Popen(command, env=environment))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"collection workers failed: {codes}")
    completed = len(list((paths["traces"] / "readiness_baseline").glob("*.json")))
    if completed != len(rows):
        raise RuntimeError(f"expected {len(rows)} traces, found {completed}")
    print(f"Collected {completed} traces with one model replica on each T4.")


if __name__ == "__main__":
    main()
