"""Minimal vLLM request-ID / decoder-row mapping probe.

Run this on Kaggle before running router.py:

    !python probe_vllm_request_ids.py

It uses one GPU and a 135M model. The output report is written to:

    /kaggle/working/vllm_request_id_probe.json

The probe does not import router.py and does not build Helix calibration data.
Its only purpose is to discover which vLLM runtime object owns the request IDs
corresponding to rows observed by a decoder-layer forward hook.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
EXPECTED_IDS = (
    "probe-000-alpha",
    "probe-001-beta",
    "probe-002-gamma",
    "probe-003-delta",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-hook-samples", type=int, default=30)
    return parser.parse_args()


def locate_runner_and_layer(engine, layer_index):
    candidates = (
        (
            "engine.model_executor.driver_worker.model_runner",
            lambda: engine.model_executor.driver_worker.model_runner,
        ),
        (
            "engine.model_executor.driver_worker.worker.model_runner",
            lambda: engine.model_executor.driver_worker.worker.model_runner,
        ),
        (
            "engine.engine_core.model_executor.driver_worker.model_runner",
            lambda: engine.engine_core.model_executor.driver_worker.model_runner,
        ),
    )
    errors = []
    for path, getter in candidates:
        try:
            runner = getter()
            layer = runner.model.model.layers[layer_index]
            return runner, layer, path
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not locate model runner / decoder layer:\n" + "\n".join(errors)
    )


def normalize_request_id(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    value = getattr(value, "request_id", value)
    return str(value)


def compact_ids(values):
    return [
        normalized
        for normalized in (normalize_request_id(value) for value in values)
        if normalized is not None
    ]


def resolve_probe_id(owner, expected_ids):
    if owner in expected_ids:
        return owner
    prefix, separator, suffix = owner.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(char in "0123456789abcdefABCDEF" for char in suffix)
        and prefix in expected_ids
    ):
        return prefix
    return None


def candidate_owner_lists(model_runner):
    """Return every plausible row-ordered request-ID list."""
    candidates = {}

    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is not None:
        req_ids = getattr(input_batch, "req_ids", None)
        if req_ids is not None:
            try:
                num_reqs = int(
                    getattr(input_batch, "num_reqs", len(req_ids))
                )
                candidates["input_batch.req_ids"] = compact_ids(
                    list(req_ids[:num_reqs])
                )
            except Exception as exc:
                candidates["input_batch.req_ids"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

        mapping = getattr(input_batch, "req_id_to_index", None)
        if isinstance(mapping, dict):
            try:
                candidates["input_batch.req_id_to_index"] = compact_ids([
                    req_id for req_id, _ in sorted(
                        mapping.items(), key=lambda pair: int(pair[1])
                    )
                ])
            except Exception as exc:
                candidates["input_batch.req_id_to_index"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

        previous_mapping = getattr(
            input_batch, "prev_req_id_to_index", None
        )
        if isinstance(previous_mapping, dict):
            try:
                candidates[
                    "input_batch.prev_req_id_to_index"
                ] = compact_ids([
                    req_id for req_id, _ in sorted(
                        previous_mapping.items(),
                        key=lambda pair: int(pair[1]),
                    )
                ])
            except Exception as exc:
                candidates["input_batch.prev_req_id_to_index"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

    req_states = getattr(model_runner, "req_states", None)
    if req_states is not None:
        mapping = getattr(req_states, "req_id_to_index", None)
        if isinstance(mapping, dict):
            try:
                candidates["req_states.req_id_to_index"] = compact_ids([
                    req_id for req_id, _ in sorted(
                        mapping.items(), key=lambda pair: int(pair[1])
                    )
                ])
            except Exception as exc:
                candidates["req_states.req_id_to_index"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

    requests = getattr(model_runner, "requests", None)
    if isinstance(requests, dict):
        candidates["model_runner.requests.keys"] = compact_ids(requests.keys())

    previous_mapping = getattr(
        model_runner, "prev_req_id_to_index", None
    )
    if isinstance(previous_mapping, dict):
        try:
            candidates["model_runner.prev_req_id_to_index"] = compact_ids([
                req_id for req_id, _ in sorted(
                    previous_mapping.items(), key=lambda pair: int(pair[1])
                )
            ])
        except Exception as exc:
            candidates["model_runner.prev_req_id_to_index"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

    return candidates


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams

    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"vLLM: {vllm.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine_args = EngineArgs(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=len(EXPECTED_IDS),
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    model_runner, layer, runner_path = locate_runner_and_layer(
        engine, args.layer
    )
    print(f"Runner path: {runner_path}")
    print(f"Runner type: {type(model_runner).__module__}.{type(model_runner).__name__}")
    print(f"Has input_batch: {hasattr(model_runner, 'input_batch')}")
    print(f"Has req_states: {hasattr(model_runner, 'req_states')}")

    expected_set = set(EXPECTED_IDS)
    stats = defaultdict(lambda: {
        "available_calls": 0,
        "row_count_matches": 0,
        "recognized_row_matches": 0,
        "exact_expected_subsets": 0,
        "examples": [],
    })
    hook_shapes = defaultdict(int)
    total_hook_calls = 0
    printed_samples = 0

    def hook_fn(module, inputs, output):
        nonlocal total_hook_calls, printed_samples
        hidden = output[0] if isinstance(output, tuple) else output
        total_hook_calls += 1
        shape = tuple(int(value) for value in hidden.shape)
        hook_shapes[str(shape)] += 1
        row_count = int(hidden.shape[0]) if hidden.ndim >= 2 else -1
        candidates = candidate_owner_lists(model_runner)

        printable = {}
        for source, owners in candidates.items():
            if not isinstance(owners, list):
                printable[source] = owners
                continue
            source_stats = stats[source]
            source_stats["available_calls"] += 1
            resolved = [
                resolve_probe_id(owner, expected_set) for owner in owners
            ]
            recognized = [owner for owner in resolved if owner is not None]
            row_match = len(owners) == row_count
            recognized_match = row_match and len(recognized) == row_count
            exact_subset = (
                recognized_match and set(recognized).issubset(expected_set)
            )
            source_stats["row_count_matches"] += int(row_match)
            source_stats["recognized_row_matches"] += int(recognized_match)
            source_stats["exact_expected_subsets"] += int(exact_subset)
            if len(source_stats["examples"]) < 5:
                source_stats["examples"].append({
                    "hook_shape": shape,
                    "row_count": row_count,
                    "owners": owners,
                    "resolved_external_ids": resolved,
                    "row_count_match": row_match,
                    "recognized_row_match": recognized_match,
                })
            printable[source] = owners

        should_print = (
            printed_samples < args.max_hook_samples
            and any(
                isinstance(owners, list)
                and len(owners) == row_count
                and any(
                    resolve_probe_id(owner, expected_set) is not None
                    for owner in owners
                )
                for owners in candidates.values()
            )
        )
        if should_print:
            printed_samples += 1
            print(
                "HOOK",
                json.dumps({
                    "call": total_hook_calls,
                    "shape": shape,
                    "candidates": printable,
                }, sort_keys=True),
            )
        return output

    handle = layer.register_forward_hook(hook_fn)
    prompts = (
        "Count slowly from one to twenty and explain each number.",
        "Write a detailed paragraph about four kinds of clouds.",
        "List practical steps for debugging a Python program.",
        "Explain how continuous batching works in language-model serving.",
    )
    output_limits = (48, 64, 80, 96)

    try:
        for request_id, prompt, max_tokens in zip(
            EXPECTED_IDS, prompts, output_limits
        ):
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
            engine.add_request(request_id, prompt, params)

        finished = {}
        steps = 0
        started = time.time()
        while engine.has_unfinished_requests():
            steps += 1
            for request_output in engine.step():
                if request_output.finished:
                    finished[str(request_output.request_id)] = len(
                        request_output.outputs[0].token_ids
                    )
            if steps > 400:
                raise RuntimeError("Probe exceeded 400 engine steps")
        elapsed = time.time() - started
    finally:
        handle.remove()

    ranked_sources = sorted(
        stats,
        key=lambda source: (
            stats[source]["exact_expected_subsets"],
            stats[source]["recognized_row_matches"],
            stats[source]["row_count_matches"],
        ),
        reverse=True,
    )
    winner = ranked_sources[0] if ranked_sources else None
    winner_valid_calls = (
        stats[winner]["exact_expected_subsets"] if winner else 0
    )
    passed = bool(winner and winner_valid_calls > 0)

    report = {
        "passed": passed,
        "recommended_source": winner,
        "recommended_source_valid_decode_calls": winner_valid_calls,
        "model": args.model,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "runner_path": runner_path,
        "runner_type": (
            f"{type(model_runner).__module__}.{type(model_runner).__name__}"
        ),
        "has_input_batch": hasattr(model_runner, "input_batch"),
        "has_req_states": hasattr(model_runner, "req_states"),
        "expected_request_ids": list(EXPECTED_IDS),
        "finished_output_lengths": finished,
        "engine_steps": steps,
        "elapsed_sec": elapsed,
        "total_hook_calls": total_hook_calls,
        "hook_shapes": dict(hook_shapes),
        "source_stats": dict(stats),
        "runner_diagnostics": {
            "runner_attributes": sorted(
                name for name in dir(model_runner)
                if any(
                    term in name.lower()
                    for term in ("req", "batch", "input")
                )
            ),
            "input_batch_type": (
                type(model_runner.input_batch).__name__
                if getattr(model_runner, "input_batch", None) is not None
                else None
            ),
            "input_batch_attributes": (
                sorted(
                    name for name in dir(model_runner.input_batch)
                    if any(
                        term in name.lower()
                        for term in ("req", "index", "batch")
                    )
                )
                if getattr(model_runner, "input_batch", None) is not None
                else []
            ),
        },
    }
    output_dir = (
        Path("/kaggle/working")
        if Path("/kaggle/working").is_dir()
        else Path.cwd()
    )
    output_path = output_dir / "vllm_request_id_probe.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"PASSED: {passed}")
    print(f"RECOMMENDED SOURCE: {winner}")
    print(f"VALID DECODE-SHAPED CALLS: {winner_valid_calls}")
    print(f"REPORT: {output_path}")
    print("=" * 80)
    if not passed:
        raise SystemExit(
            "No candidate source mapped decoder rows to the submitted IDs. "
            "Attach vllm_request_id_probe.json."
        )


if __name__ == "__main__":
    main()
