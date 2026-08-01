from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    read_jsonl,
    write_jsonl,
)
from helix_role_experiment.readiness import (
    annotation_json_schema,
    parse_response_output,
    validate_annotations,
)


SYSTEM_PROMPT = """You label immutable, pre-segmented reasoning sentences. Never split, merge, renumber, invent, or omit a sentence. A reference answer is evidence, not the only permitted method. For code tasks, mathematically_correct means locally correct as program reasoning.

Label forward_progress only when the sentence makes a correct, novel state change that advances any valid path to the solution. Planning, restatement, local algebra with no useful state change, and unsupported claims are not progress. Label productive_backtrack for an explicit useful correction or return from a failed path. Label final_answer only for the submitted answer/code outside the reasoning section. Copy evidence exactly from that sentence; use an empty string when no exact evidence is appropriate. Every forward_progress item must have mathematically_correct=yes, novel=yes, and advances_valid_path=yes. Every uncertain field requires needs_review=true. Return only the requested schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label trajectories immediately with concurrent Luna Responses API calls"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["prepare", "run", "validate"])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum simultaneous Responses API calls; defaults to labeling.concurrency",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Label only the first N pending traces for a smoke test",
    )
    return parser.parse_args()


def api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY")
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret("OPENAI_API_KEY")
        except Exception as exc:
            raise RuntimeError(
                "Set OPENAI_API_KEY or create the Kaggle secret OPENAI_API_KEY"
            ) from exc
    os.environ["OPENAI_API_KEY"] = value
    return value


def trace_files(paths: dict[str, Path]) -> list[Path]:
    return sorted((paths["traces"] / "readiness_baseline").glob("*.json"))


def request_for_trace(trace: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    sentence_text = "\n".join(
        f"[{row['sentence_id']}] {row['text']}" for row in trace["sentences"]
    )
    user = (
        f"DOMAIN: {trace['domain']}\nTASK:\n{trace['prompt']}\n\n"
        f"REFERENCE (non-exclusive):\n{trace['reference_answer']}\n\n"
        f"IMMUTABLE SENTENCES:\n{sentence_text}"
    )
    labeling = config["labeling"]
    return {
        "model": labeling.get("model", "gpt-5.6-luna"),
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "reasoning": {"effort": labeling.get("reasoning_effort", "low")},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sentence_annotations",
                "strict": True,
                "schema": annotation_json_schema(),
            }
        },
        "max_output_tokens": int(labeling.get("max_output_tokens", 10000)),
        "store": False,
    }


def estimate_requests(config: dict[str, Any], paths: dict[str, Path]) -> tuple[list[dict], dict]:
    requests = []
    estimated_input = 0.0
    estimated_output = 0
    for source in trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        body = request_for_trace(trace, config)
        requests.append({"trace_id": trace["trace_id"], "body": body})
        estimated_input += len(json.dumps(body["input"], ensure_ascii=False)) / 4
        estimated_output += 55 * len(trace["sentences"])
    rates = config["labeling"].get(
        "prices_per_million", {"input": 0.2, "output": 1.2}
    )
    estimate = (
        estimated_input / 1e6 * float(rates["input"])
        + estimated_output / 1e6 * float(rates["output"])
    )
    report = {
        "requests": len(requests),
        "estimated_input_tokens": round(estimated_input),
        "estimated_output_tokens": estimated_output,
        "estimated_immediate_cost_usd": estimate,
        "hard_budget_usd": float(config["labeling"].get("max_cost_usd", 5.6)),
    }
    return requests, report


def prepare(config: dict[str, Any], paths: dict[str, Path]) -> list[dict]:
    requests, report = estimate_requests(config, paths)
    destination = paths["tables"] / "luna_label_requests.jsonl"
    write_jsonl(destination, requests)
    atomic_json(paths["tables"] / "luna_cost_estimate.json", report)
    if report["estimated_immediate_cost_usd"] > report["hard_budget_usd"]:
        raise RuntimeError(
            f"estimated immediate label cost ${report['estimated_immediate_cost_usd']:.2f} "
            f"exceeds ${report['hard_budget_usd']:.2f} guard"
        )
    print(
        f"Prepared {len(requests)} immediate requests; conservative estimate "
        f"${report['estimated_immediate_cost_usd']:.2f}.",
        flush=True,
    )
    return requests


def response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    raise TypeError("OpenAI response cannot be serialized")


def usage_from_response(value: dict[str, Any]) -> dict[str, int]:
    usage = value.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


async def run_immediate(
    config: dict[str, Any],
    paths: dict[str, Path],
    concurrency: int,
    limit: int | None,
) -> None:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the labeling extra: pip install -e '.[labeling]'"
        ) from exc

    requests = prepare(config, paths)
    traces = {
        source.stem: json.loads(source.read_text(encoding="utf-8"))
        for source in trace_files(paths)
    }
    result_dir = paths["traces"] / "luna_sentence_labels"
    result_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        request
        for request in requests
        if not (result_dir / f"{request['trace_id']}.json").exists()
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No pending trajectories; rebuilding the validated annotation table.")
        validate_cached(paths, allow_missing=limit is not None)
        return

    labeling = config["labeling"]
    attempts = int(labeling.get("validation_attempts", 3))
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    failed: list[dict[str, str]] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    lock = asyncio.Lock()

    async with AsyncOpenAI(
        api_key=api_key(),
        max_retries=int(labeling.get("sdk_retries", 3)),
        timeout=float(labeling.get("timeout_seconds", 600)),
    ) as client:

        async def label_one(request: dict[str, Any]) -> None:
            nonlocal completed
            trace_id = request["trace_id"]
            body = dict(request["body"])
            last_error = "unknown failure"
            for attempt in range(1, attempts + 1):
                retry_body = dict(body)
                if attempt > 1:
                    retry_body["input"] = list(body["input"]) + [{
                        "role": "user",
                        "content": (
                            "Your previous response failed deterministic local validation: "
                            f"{last_error}. Re-label the same immutable sentences and obey every "
                            "schema and consistency rule exactly."
                        ),
                    }]
                try:
                    async with semaphore:
                        response = await client.responses.create(**retry_body)
                    raw = response_dict(response)
                    usage = usage_from_response(raw)
                    async with lock:
                        for key in usage_totals:
                            usage_totals[key] += usage[key]
                    payload = parse_response_output(raw)
                    annotations = validate_annotations(
                        traces[trace_id]["sentences"], payload
                    )
                    atomic_json(
                        result_dir / f"{trace_id}.json",
                        {
                            "trace_id": trace_id,
                            "response_id": raw.get("id"),
                            "request_id": getattr(response, "_request_id", None),
                            "model": raw.get("model"),
                            "usage": usage,
                            "annotations": annotations,
                        },
                    )
                    async with lock:
                        completed += 1
                        print(
                            f"[{completed}/{len(pending)}] labeled {trace_id} "
                            f"({usage['input_tokens']} in, {usage['output_tokens']} out)",
                            flush=True,
                        )
                    return
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
            async with lock:
                failed.append({"trace_id": trace_id, "error": last_error})
                print(f"FAILED {trace_id}: {last_error}", flush=True)

        await asyncio.gather(*(label_one(request) for request in pending))

    rates = labeling.get("prices_per_million", {"input": 0.2, "output": 1.2})
    actual_cost = (
        usage_totals["input_tokens"] / 1e6 * float(rates["input"])
        + usage_totals["output_tokens"] / 1e6 * float(rates["output"])
    )
    atomic_json(
        paths["tables"] / "luna_immediate_run.json",
        {
            "newly_completed": completed,
            "failed": failed,
            "usage": usage_totals,
            "actual_new_cost_usd": actual_cost,
            "concurrency": concurrency,
        },
    )
    validate_cached(paths, allow_missing=limit is not None)
    if failed:
        raise RuntimeError(
            f"{len(failed)} trajectories failed after {attempts} attempts; rerun to resume"
        )


def validate_cached(paths: dict[str, Path], allow_missing: bool = False) -> None:
    traces = {
        source.stem: json.loads(source.read_text(encoding="utf-8"))
        for source in trace_files(paths)
    }
    result_dir = paths["traces"] / "luna_sentence_labels"
    rows = []
    failures = []
    for trace_id, trace in traces.items():
        source = result_dir / f"{trace_id}.json"
        if not source.exists():
            failures.append({"trace_id": trace_id, "error": "missing immediate result"})
            continue
        try:
            record = json.loads(source.read_text(encoding="utf-8"))
            annotations = validate_annotations(
                trace["sentences"], {"annotations": record["annotations"]}
            )
            rows.append({"trace_id": trace_id, "annotations": annotations})
        except Exception as exc:
            failures.append({"trace_id": trace_id, "error": str(exc)})
    write_jsonl(paths["tables"] / "sentence_annotations.jsonl", rows)
    write_jsonl(paths["tables"] / "sentence_annotation_failures.jsonl", failures)
    print(f"Validated {len(rows)} of {len(traces)} trajectories.", flush=True)
    if failures and not allow_missing:
        raise RuntimeError(
            f"{len(failures)} trajectories are missing or invalid; rerun `run` to resume"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.command == "prepare":
        prepare(config, paths)
    elif args.command == "run":
        concurrency = args.concurrency or int(config["labeling"].get("concurrency", 8))
        asyncio.run(run_immediate(config, paths, concurrency, args.limit))
    else:
        validate_cached(paths)


if __name__ == "__main__":
    main()
