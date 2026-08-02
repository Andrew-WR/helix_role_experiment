from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    write_jsonl,
)
from helix_role_experiment.readiness import annotation_json_schema, validate_annotations


SCRIPT_DIR = Path(__file__).resolve().parent
API_LABELER_PATH = SCRIPT_DIR / "07b_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location("label_subgoal_events_07b", API_LABELER_PATH)
API_LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(API_LABELER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume missing sentence labels with one 4-bit vLLM Qwen replica per GPU"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Defaults to model.id in the config")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Label only the first N missing chunks (for a smoke test)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_requests(
    requests: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    """Assign whole trajectories to replicas so their common prefix is reusable."""
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    trace_ids = sorted({str(row["trace_id"]) for row in requests})
    selected = {
        trace_id
        for index, trace_id in enumerate(trace_ids)
        if index % shard_count == shard_index
    }
    return [row for row in requests if str(row["trace_id"]) in selected]


def grouped_by_trace(
    requests: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        groups[str(request["trace_id"])].append(request)
    return [
        (trace_id, sorted(rows, key=lambda row: int(row["chunk_index"])))
        for trace_id, rows in sorted(groups.items())
    ]


def build_prompt(tokenizer: Any, request: dict[str, Any], correction: str | None = None) -> str:
    messages = list(request["body"]["input"])
    if correction:
        messages.append({
            "role": "user",
            "content": (
                "Your previous response failed deterministic local validation: "
                f"{correction}. Return a corrected JSON object for the same target chunk. "
                "Keep every target sentence ID exactly once and in the requested order."
            ),
        })
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def sampling_params(max_output_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # A generic schema can be compiled once and shared by a batch. Exact IDs and
    # counts are enforced by validate_annotations before anything is checkpointed.
    structured = StructuredOutputsParams(json=annotation_json_schema())
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_output_tokens,
        seed=seed,
        structured_outputs=structured,
    )


def output_usage(output: Any) -> dict[str, int]:
    input_tokens = len(output.prompt_token_ids or [])
    output_tokens = len(output.outputs[0].token_ids or [])
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def target_for(request: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, Any]]:
    return trace["sentences"][
        int(request["sentence_start"]) : int(request["sentence_end"])
    ]


def save_output(
    output: Any,
    request: dict[str, Any],
    target: list[dict[str, Any]],
    model_id: str,
    paths: dict[str, Path],
) -> dict[str, int]:
    payload = json.loads(output.outputs[0].text)
    annotations = validate_annotations(target, payload)
    usage = output_usage(output)
    destination = API_LABELER.chunk_result_path(
        paths, str(request["trace_id"]), int(request["chunk_index"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, {
        "trace_id": request["trace_id"],
        "chunk_index": request["chunk_index"],
        "chunk_count": request["chunk_count"],
        "sentence_ids": [row["sentence_id"] for row in target],
        "source": "local_qwen_vllm",
        "response_id": getattr(output, "request_id", None),
        "request_id": None,
        "model": model_id,
        "usage": usage,
        "annotations": annotations,
    })
    return usage


def still_missing(
    request: dict[str, Any], trace: dict[str, Any], paths: dict[str, Path]
) -> bool:
    target = target_for(request, trace)
    destination = API_LABELER.chunk_result_path(
        paths, str(request["trace_id"]), int(request["chunk_index"])
    )
    return API_LABELER.valid_result(destination, target) is None


def run_batch(
    llm: Any,
    tokenizer: Any,
    params: Any,
    batch: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    model_id: str,
    paths: dict[str, Path],
    attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    prompts = [build_prompt(tokenizer, request) for request in batch]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    failures: list[dict[str, Any]] = []
    successes: dict[str, dict[str, int]] = {}
    for request, output in zip(batch, outputs, strict=True):
        key = f"{request['trace_id']}:{request['chunk_index']}"
        trace = traces[str(request["trace_id"])]
        target = target_for(request, trace)
        last_error = "unknown failure"
        for attempt in range(1, attempts + 1):
            try:
                successes[key] = save_output(output, request, target, model_id, paths)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == attempts:
                    failures.append({
                        "trace_id": request["trace_id"],
                        "chunk_index": request["chunk_index"],
                        "error": last_error,
                    })
                    break
                correction = build_prompt(tokenizer, request, last_error)
                output = llm.generate([correction], params, use_tqdm=False)[0]
    return failures, successes


def run_worker(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    if args.batch_size <= 0 or args.attempts <= 0:
        raise ValueError("batch-size and attempts must be positive")
    if not args.plan:
        raise ValueError("worker requires --plan")

    from vllm import LLM

    plan = [
        json.loads(line)
        for line in Path(args.plan).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assigned = selected_requests(plan, args.shard_index, args.replicas)
    traces = {
        source.stem: json.loads(source.read_text(encoding="utf-8"))
        for source in API_LABELER.trace_files(paths)
    }
    assigned = [
        request
        for request in assigned
        if still_missing(request, traces[str(request["trace_id"])], paths)
    ]
    if not assigned:
        print(f"[replica {args.shard_index}] nothing missing", flush=True)
        return

    model_id = args.model or config["model"]["id"]
    print(
        f"[replica {args.shard_index}] loading {model_id}; "
        f"{len(assigned)} chunks assigned",
        flush=True,
    )
    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        dtype="float16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=False,
        seed=int(config["study"].get("seed", 0)),
    )
    tokenizer = llm.get_tokenizer()
    params = sampling_params(
        args.max_output_tokens,
        int(config["study"].get("seed", 0)) + args.shard_index,
    )
    completed = 0
    failures: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for trace_id, trace_requests in grouped_by_trace(assigned):
        # Warm the repeated full-trajectory prefix with one request before the
        # remaining chunks enter a batch. Subsequent chunks can hit vLLM's APC.
        batches = [trace_requests[:1]] + [
            trace_requests[index : index + args.batch_size]
            for index in range(1, len(trace_requests), args.batch_size)
        ]
        for batch in batches:
            batch_failures, successes = run_batch(
                llm, tokenizer, params, batch, traces, model_id, paths, args.attempts
            )
            failures.extend(batch_failures)
            for request in batch:
                key = f"{request['trace_id']}:{request['chunk_index']}"
                if key not in successes:
                    print(
                        f"[replica {args.shard_index}] FAILED {trace_id} chunk "
                        f"{int(request['chunk_index']) + 1}/{request['chunk_count']}",
                        flush=True,
                    )
                    continue
                completed += 1
                for field, value in successes[key].items():
                    totals[field] += value
                print(
                    f"[replica {args.shard_index}] [{completed}/{len(assigned)}] "
                    f"saved {trace_id} chunk {int(request['chunk_index']) + 1}/"
                    f"{request['chunk_count']}",
                    flush=True,
                )
        API_LABELER.materialize_chunked_result(traces[trace_id], config, paths)

    atomic_json(paths["logs"] / f"07b_local_qwen_replica_{args.shard_index}.json", {
        "replica": args.shard_index,
        "assigned_chunks": len(assigned),
        "completed_chunks": completed,
        "failed": failures,
        "usage": totals,
        "model": model_id,
    })
    if failures:
        raise RuntimeError(
            f"replica {args.shard_index}: {len(failures)} chunks failed; rerun to resume"
        )


def prepare_plan(
    config: dict[str, Any], paths: dict[str, Path], limit: int | None
) -> tuple[Path, list[dict[str, Any]]]:
    requests, report = API_LABELER.estimate_requests(config, paths)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        requests = requests[:limit]
    plan_path = paths["tables"] / "local_qwen_label_requests.jsonl"
    write_jsonl(plan_path, requests)
    atomic_json(paths["tables"] / "local_qwen_label_plan.json", {
        **report,
        "selected_pending_chunks": len(requests),
        "labeler": "local_qwen_vllm",
    })
    return plan_path, requests


def launch_replicas(
    args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]
) -> None:
    if args.replicas <= 0:
        raise ValueError("replicas must be positive")
    plan_path, requests = prepare_plan(config, paths, args.limit)
    if not requests:
        print("No missing chunks; rebuilding the annotation table.", flush=True)
        API_LABELER.validate_cached(config, paths)
        return
    print(
        f"Preserved all valid whole/chunk results; {len(requests)} chunks remain. "
        f"Launching {args.replicas} replicas.",
        flush=True,
    )
    children = []
    for shard_index in range(args.replicas):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard_index)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config", args.config,
            "--replicas", str(args.replicas),
            "--batch-size", str(args.batch_size),
            "--max-model-len", str(args.max_model_len),
            "--max-output-tokens", str(args.max_output_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--attempts", str(args.attempts),
            "--worker",
            "--shard-index", str(shard_index),
            "--plan", str(plan_path),
        ]
        if args.model:
            command.extend(["--model", args.model])
        children.append(subprocess.Popen(command, env=environment))
    return_codes = [child.wait() for child in children]
    API_LABELER.validate_cached(
        config,
        paths,
        allow_missing=args.limit is not None or any(return_codes),
    )
    if any(return_codes):
        raise RuntimeError(
            f"one or more replicas failed ({return_codes}); rerun the same command to resume"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.worker:
        run_worker(args, config, paths)
    else:
        launch_replicas(args, config, paths)


if __name__ == "__main__":
    main()
