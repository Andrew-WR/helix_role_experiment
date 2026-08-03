from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

from helix_role_experiment.config import atomic_json, ensure_output_dirs, load_config, write_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_LABELER_PATH = SCRIPT_DIR / "07b_gemini_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location(
    "compact_label_helpers_07b", GEMINI_LABELER_PATH
)
COMPACT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(COMPACT)
API_LABELER = COMPACT.API_LABELER


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label missing complete trajectories with a compact 30B-A3B "
            "judge tensor-parallel across both T4s"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--audit-passes", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Label only the first N missing trajectories for a smoke test",
    )
    return parser.parse_args()


def generic_compact_schema() -> dict[str, Any]:
    """One reusable grammar; exact lengths and bounds are checked locally."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels", "review"],
        "properties": {
            "labels": {
                "type": "string",
                "description": "One F/B/N/R/I/A code per input sentence, in order.",
            },
            "review": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
        },
    }


def configure_environment() -> None:
    temporary = Path("/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", str(temporary / "compact_label_triton"))
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", str(temporary / "compact_label_inductor")
    )
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Kaggle's two T4s are not guaranteed to expose peer-to-peer access. NCCL
    # can still tensor-parallelize through host shared memory when P2P is off.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def build_prompt(tokenizer: Any, trace: dict[str, Any], prior: dict[str, Any] | None,
                 correction: str | None = None) -> str:
    messages = [
        {"role": "system", "content": COMPACT.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPACT.trajectory_prompt(
                trace, prior=prior, correction=correction
            ),
        },
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def sampling_params(max_output_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_output_tokens,
        seed=int(seed),
        structured_outputs=StructuredOutputsParams(json=generic_compact_schema()),
    )


def parse_output(trace: dict[str, Any], output: Any) -> dict[str, Any]:
    if not output.outputs:
        raise ValueError("vLLM returned no candidate")
    payload = json.loads(output.outputs[0].text)
    return COMPACT.validate_compact_payload(trace, payload)


def output_usage(output: Any) -> dict[str, int]:
    input_tokens = len(output.prompt_token_ids or [])
    output_tokens = len(output.outputs[0].token_ids or []) if output.outputs else 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def generate_batch(
    llm: Any, tokenizer: Any, params: Any,
    items: list[tuple[dict[str, Any], dict[str, Any] | None]], attempts: int,
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, int], dict[str, Any]]],
           list[dict[str, Any]]]:
    prompts = [build_prompt(tokenizer, trace, prior) for trace, prior in items]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    successes: dict[str, tuple[dict[str, Any], dict[str, int], dict[str, Any]]] = {}
    failures = []
    for (trace, prior), output in zip(items, outputs, strict=True):
        trace_id = str(trace["trace_id"])
        last_error = "unknown validation failure"
        candidate = output
        for attempt in range(1, attempts + 1):
            try:
                payload = parse_output(trace, candidate)
                usage = output_usage(candidate)
                successes[trace_id] = (
                    payload,
                    usage,
                    {
                        "attempt": attempt,
                        "payload": payload,
                        "usage": usage,
                        "request_id": getattr(candidate, "request_id", None),
                    },
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= attempts:
                    failures.append({"trace_id": trace_id, "error": last_error})
                    break
                retry_prompt = build_prompt(
                    tokenizer, trace, prior, correction=last_error
                )
                candidate = llm.generate(
                    [retry_prompt], params, use_tqdm=False
                )[0]
    return successes, failures


def write_review_queue(paths: dict[str, Path]) -> list[dict[str, Any]]:
    traces = {
        source.stem: json.loads(source.read_text(encoding="utf-8"))
        for source in API_LABELER.trace_files(paths)
    }
    rows = []
    directory = API_LABELER.result_directory(paths)
    for source in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            record = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("source") != "local_compact_qwen_moe":
            continue
        trace_id = str(record.get("trace_id", source.stem))
        trace = traces.get(trace_id)
        compact = record.get("compact_labels") or {}
        labels = compact.get("labels")
        if trace is None or not isinstance(labels, str):
            continue
        review = set(compact.get("review") or ()) | set(
            record.get("audit_disagreement_positions") or ()
        )
        pass_labels = [
            ((item.get("payload") or {}).get("labels"))
            for item in record.get("passes", [])
        ]
        for index in sorted(review):
            if not 0 <= index < len(trace["sentences"]):
                continue
            code = labels[index]
            if code == "A":
                continue
            rows.append({
                "trace_id": trace_id,
                "task_id": trace["task_id"],
                "domain": trace["domain"],
                "split": trace["split"],
                "sentence_index": index,
                "sentence_id": trace["sentences"][index]["sentence_id"],
                "text": trace["sentences"][index]["text"],
                "final_code": code,
                "final_label": COMPACT.CODE_TO_LABEL[code],
                "pass_codes": [
                    value[index] for value in pass_labels
                    if isinstance(value, str) and index < len(value)
                ],
            })
    write_jsonl(paths["tables"] / "local_compact_review_queue.jsonl", rows)
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.attempts <= 0 or args.audit_passes < 0:
        raise ValueError("batch-size/attempts must be positive; audit-passes nonnegative")
    if args.max_model_len <= args.max_output_tokens:
        raise ValueError("max-model-len must exceed max-output-tokens")
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    pending, preserved = COMPACT.pending_traces(config, paths)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        pending = pending[:args.limit]
    print(
        f"Preserved {preserved} complete trajectories; selected "
        f"{len(pending)} missing trajectories.",
        flush=True,
    )
    if not pending:
        API_LABELER.validate_cached(config, paths)
        write_review_queue(paths)
        return

    configure_environment()
    import torch
    import vllm
    from vllm import EngineArgs, LLM

    print(
        f"Loading {args.model} tensor-parallel across 2 GPUs; "
        f"vLLM={vllm.__version__}; torch={torch.__version__}; "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}; "
        f"GPU0={torch.cuda.get_device_name(0)}; GPU1={torch.cuda.get_device_name(1)}",
        flush=True,
    )
    engine_kwargs = dict(
        model=args.model,
        tokenizer=args.model,
        dtype="float16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
        seed=int(config["study"].get("seed", 0)),
    )
    if "language_model_only" in inspect.signature(EngineArgs).parameters:
        engine_kwargs["language_model_only"] = True
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    params = sampling_params(
        args.max_output_tokens, int(config["study"].get("seed", 0))
    )
    total = len(pending)
    completed = 0
    failures: list[dict[str, Any]] = []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        for start in range(0, total, args.batch_size):
            batch = pending[start : start + args.batch_size]
            active = {str(trace["trace_id"]): trace for trace in batch}
            current: dict[str, dict[str, Any] | None] = {
                trace_id: None for trace_id in active
            }
            histories: dict[str, list[dict[str, Any]]] = {
                trace_id: [] for trace_id in active
            }
            disagreements: dict[str, set[int]] = {
                trace_id: set() for trace_id in active
            }
            usages: dict[str, dict[str, int]] = {
                trace_id: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                for trace_id in active
            }

            for pass_index in range(1 + args.audit_passes):
                items = [
                    (trace, current[trace_id])
                    for trace_id, trace in active.items()
                ]
                successes, pass_failures = generate_batch(
                    llm, tokenizer, params, items, args.attempts
                )
                failures.extend({**row, "pass_index": pass_index} for row in pass_failures)
                for trace_id in list(active):
                    if trace_id not in successes:
                        active.pop(trace_id)
                        continue
                    previous = current[trace_id]
                    payload, usage, metadata = successes[trace_id]
                    metadata["pass_index"] = pass_index
                    current[trace_id] = payload
                    histories[trace_id].append(metadata)
                    for key in usages[trace_id]:
                        usages[trace_id][key] += usage[key]
                    if previous is not None:
                        disagreements[trace_id].update(
                            index for index, (before, after) in enumerate(
                                zip(previous["labels"], payload["labels"], strict=True)
                            ) if before != after
                        )

            for trace_id, trace in active.items():
                payload = current[trace_id]
                assert payload is not None
                annotations = COMPACT.compact_to_annotations(
                    trace, payload, disagreement=disagreements[trace_id]
                )
                destination = API_LABELER.full_result_path(paths, trace_id)
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(destination, {
                    "trace_id": trace_id,
                    "source": "local_compact_qwen_moe",
                    "model": args.model,
                    "usage": usages[trace_id],
                    "compact_labels": payload,
                    "audit_disagreement_positions": sorted(disagreements[trace_id]),
                    "agreement_rate": (
                        1.0 - len(disagreements[trace_id]) / len(annotations)
                    ),
                    "passes": histories[trace_id],
                    "annotations": annotations,
                })
                completed += 1
                for key in totals:
                    totals[key] += usages[trace_id][key]
                print(
                    f"[{completed}/{total}] saved {trace_id}: "
                    f"{len(annotations)} sentences, "
                    f"{len(disagreements[trace_id])} audit disagreements, "
                    f"{usages[trace_id]['output_tokens']} output tokens",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(
            "Paused safely. Atomically saved trajectories will be skipped when "
            "the same command is rerun.",
            flush=True,
        )

    atomic_json(paths["logs"] / "07b_local_compact_qwen_moe.json", {
        "selected_trajectories": total,
        "completed_trajectories": completed,
        "failed": failures,
        "usage": totals,
        "model": args.model,
        "tensor_parallel_size": 2,
        "audit_passes": args.audit_passes,
    })
    API_LABELER.validate_cached(
        config, paths,
        allow_missing=args.limit is not None or completed < total or bool(failures),
    )
    review = write_review_queue(paths)
    print(
        f"Review queue: {len(review)} reasoning sentences at "
        f"{paths['tables'] / 'local_compact_review_queue.jsonl'}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} pass requests failed; rerun to resume incomplete traces"
        )


if __name__ == "__main__":
    main()
