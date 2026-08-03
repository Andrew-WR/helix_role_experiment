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
    write_jsonl,
)
from helix_role_experiment.readiness import (
    annotation_json_schema,
    parse_response_output,
    validate_annotations,
)


SYSTEM_PROMPT = """You label immutable, pre-segmented reasoning sentences. You receive the complete trajectory as read-only context and a smaller target chunk. Return exactly one annotation for each target sentence, in target order. Never label a context-only sentence, and never split, merge, renumber, invent, duplicate, or omit a target sentence. A reference answer is evidence, not the only permitted method. For code tasks, mathematically_correct means locally correct as program reasoning.

Label forward_progress only when the sentence makes a correct, novel state change that advances any valid path to the solution. Judge novelty relative to all earlier sentences in the complete trajectory, not merely the target chunk. Planning, restatement, local algebra with no useful state change, and unsupported claims are not progress. Label productive_backtrack for an explicit useful correction or return from a failed path. Label final_answer only for the submitted answer/code outside the reasoning section. Copy evidence exactly from that target sentence; use an empty string when no exact evidence is appropriate. Every forward_progress item must have mathematically_correct=yes, novel=yes, and advances_valid_path=yes. Every uncertain field requires needs_review=true. Return only the requested schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label missing trajectories in resumable Luna sentence chunks"
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
        help="Process only the first N pending chunks for a smoke test",
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


def result_directory(paths: dict[str, Path]) -> Path:
    return paths["traces"] / "luna_sentence_labels"


def chunk_directory(paths: dict[str, Path], trace_id: str) -> Path:
    return result_directory(paths) / "chunks" / trace_id


def sentence_chunks(sentences: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("labeling.chunk_sentences must be positive")
    return [sentences[index : index + size] for index in range(0, len(sentences), size)]


def valid_result(path: Path, sentences: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["annotations"] = validate_annotations(
            sentences, {"annotations": record["annotations"]}
        )
        return record
    except Exception:
        return None


def full_result_path(paths: dict[str, Path], trace_id: str) -> Path:
    return result_directory(paths) / f"{trace_id}.json"


def chunk_result_path(paths: dict[str, Path], trace_id: str, chunk_index: int) -> Path:
    return chunk_directory(paths, trace_id) / f"{chunk_index:04d}.json"


def request_for_chunk(
    trace: dict[str, Any],
    target: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    complete_text = "\n".join(
        f"[{row['sentence_id']}] {row['text']}" for row in trace["sentences"]
    )
    target_text = "\n".join(
        f"[{row['sentence_id']}] {row['text']}" for row in target
    )
    target_ids = [str(row["sentence_id"]) for row in target]
    user = (
        f"DOMAIN: {trace['domain']}\nTASK:\n{trace['prompt']}\n\n"
        f"REFERENCE (non-exclusive):\n{trace['reference_answer']}\n\n"
        "COMPLETE IMMUTABLE TRAJECTORY — CONTEXT ONLY:\n"
        f"{complete_text}\n\n"
        f"TARGET CHUNK {chunk_index + 1}/{chunk_count} — LABEL ONLY THESE "
        f"{len(target)} SENTENCES:\n{target_text}\n\n"
        f"Return these sentence IDs exactly once and in this exact order: "
        f"{', '.join(target_ids)}"
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
                "schema": annotation_json_schema(target_ids),
            }
        },
        "max_output_tokens": int(labeling.get("chunk_max_output_tokens", 5000)),
        "store": False,
    }


def materialize_chunked_result(
    trace: dict[str, Any], config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any] | None:
    trace_id = trace["trace_id"]
    destination = full_result_path(paths, trace_id)
    existing = valid_result(destination, trace["sentences"])
    if existing is not None:
        return existing

    size = int(config["labeling"].get("chunk_sentences", 24))
    chunks = sentence_chunks(trace["sentences"], size)
    records = []
    annotations = []
    for chunk_index, sentences in enumerate(chunks):
        record = valid_result(
            chunk_result_path(paths, trace_id, chunk_index), sentences
        )
        if record is None:
            return None
        records.append(record)
        annotations.extend(record["annotations"])
    annotations = validate_annotations(
        trace["sentences"], {"annotations": annotations}
    )
    usage = {
        key: sum(int(record.get("usage", {}).get(key, 0)) for record in records)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    combined = {
        "trace_id": trace_id,
        "source": "chunked",
        "model": records[0].get("model") if records else None,
        "response_ids": [record.get("response_id") for record in records],
        "request_ids": [record.get("request_id") for record in records],
        "chunk_sentences": size,
        "chunk_count": len(chunks),
        "usage": usage,
        "annotations": annotations,
    }
    atomic_json(destination, combined)
    return combined


def estimate_requests(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_directory(paths).mkdir(parents=True, exist_ok=True)
    size = int(config["labeling"].get("chunk_sentences", 24))
    requests = []
    estimated_input = 0.0
    estimated_output = 0
    preserved_complete = 0
    cached_chunks = 0
    for source in trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        if materialize_chunked_result(trace, config, paths) is not None:
            preserved_complete += 1
            continue
        chunks = sentence_chunks(trace["sentences"], size)
        for chunk_index, target in enumerate(chunks):
            chunk_directory(paths, trace["trace_id"]).mkdir(parents=True, exist_ok=True)
            if valid_result(
                chunk_result_path(paths, trace["trace_id"], chunk_index), target
            ) is not None:
                cached_chunks += 1
                continue
            body = request_for_chunk(
                trace, target, chunk_index, len(chunks), config
            )
            requests.append({
                "trace_id": trace["trace_id"],
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "sentence_start": chunk_index * size,
                "sentence_end": chunk_index * size + len(target),
                "body": body,
            })
            estimated_input += len(json.dumps(body["input"], ensure_ascii=False)) / 4
            estimated_output += 60 * len(target)
    rates = config["labeling"].get(
        "prices_per_million", {"input": 0.2, "output": 1.2}
    )
    estimate = (
        estimated_input / 1e6 * float(rates["input"])
        + estimated_output / 1e6 * float(rates["output"])
    )
    report = {
        "pending_chunk_requests": len(requests),
        "preserved_complete_trajectories": preserved_complete,
        "cached_valid_chunks": cached_chunks,
        "chunk_sentences": size,
        "estimated_input_tokens": round(estimated_input),
        "estimated_output_tokens": estimated_output,
        "estimated_immediate_cost_usd": estimate,
        "hard_budget_usd": float(config["labeling"].get("max_cost_usd", 5.6)),
    }
    return requests, report


def prepare(config: dict[str, Any], paths: dict[str, Path]) -> list[dict[str, Any]]:
    requests, report = estimate_requests(config, paths)
    write_jsonl(paths["tables"] / "luna_label_requests.jsonl", requests)
    atomic_json(paths["tables"] / "luna_cost_estimate.json", report)
    if report["estimated_immediate_cost_usd"] > report["hard_budget_usd"]:
        raise RuntimeError(
            f"estimated immediate label cost ${report['estimated_immediate_cost_usd']:.2f} "
            f"exceeds ${report['hard_budget_usd']:.2f} guard"
        )
    print(
        f"Preserved {report['preserved_complete_trajectories']} complete results; "
        f"prepared {len(requests)} missing chunks of at most "
        f"{report['chunk_sentences']} sentences; conservative estimate "
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

    pending = prepare(config, paths)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No pending chunks; rebuilding the validated annotation table.")
        validate_cached(config, paths, allow_missing=limit is not None)
        return

    traces = {
        source.stem: json.loads(source.read_text(encoding="utf-8"))
        for source in trace_files(paths)
    }
    labeling = config["labeling"]
    attempts = int(labeling.get("validation_attempts", 3))
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    failed: list[dict[str, Any]] = []
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
            chunk_index = int(request["chunk_index"])
            trace = traces[trace_id]
            target = trace["sentences"][
                int(request["sentence_start"]) : int(request["sentence_end"])
            ]
            body = dict(request["body"])
            last_error = "unknown failure"
            for attempt in range(1, attempts + 1):
                retry_body = dict(body)
                if attempt > 1:
                    retry_body["input"] = list(body["input"]) + [{
                        "role": "user",
                        "content": (
                            "Your previous response failed deterministic local validation: "
                            f"{last_error}. Re-label only the same target chunk and obey every "
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
                    annotations = validate_annotations(
                        target, parse_response_output(raw)
                    )
                    destination = chunk_result_path(paths, trace_id, chunk_index)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(destination, {
                        "trace_id": trace_id,
                        "chunk_index": chunk_index,
                        "chunk_count": request["chunk_count"],
                        "sentence_ids": [row["sentence_id"] for row in target],
                        "response_id": raw.get("id"),
                        "request_id": getattr(response, "_request_id", None),
                        "model": raw.get("model"),
                        "usage": usage,
                        "annotations": annotations,
                    })
                    async with lock:
                        completed += 1
                        print(
                            f"[{completed}/{len(pending)}] labeled {trace_id} "
                            f"chunk {chunk_index + 1}/{request['chunk_count']} "
                            f"({usage['input_tokens']} in, {usage['output_tokens']} out)",
                            flush=True,
                        )
                    return
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
            async with lock:
                failed.append({
                    "trace_id": trace_id,
                    "chunk_index": chunk_index,
                    "error": last_error,
                })
                print(
                    f"FAILED {trace_id} chunk {chunk_index + 1}/"
                    f"{request['chunk_count']}: {last_error}",
                    flush=True,
                )

        await asyncio.gather(*(label_one(request) for request in pending))

    rates = labeling.get("prices_per_million", {"input": 0.2, "output": 1.2})
    actual_cost = (
        usage_totals["input_tokens"] / 1e6 * float(rates["input"])
        + usage_totals["output_tokens"] / 1e6 * float(rates["output"])
    )
    atomic_json(paths["tables"] / "luna_immediate_run.json", {
        "newly_completed_chunks": completed,
        "failed": failed,
        "usage": usage_totals,
        "actual_new_cost_usd": actual_cost,
        "concurrency": concurrency,
    })
    validate_cached(
        config,
        paths,
        allow_missing=limit is not None or bool(failed),
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} chunks failed after {attempts} attempts; rerun to resume"
        )


def validate_cached(
    config: dict[str, Any],
    paths: dict[str, Path],
    allow_missing: bool = False,
) -> None:
    rows = []
    failures = []
    for source in trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        record = materialize_chunked_result(trace, config, paths)
        if record is None:
            failures.append({
                "trace_id": trace["trace_id"],
                "error": "missing immediate result or one or more chunks",
            })
            continue
        rows.append({
            "trace_id": trace["trace_id"],
            "source": record.get("source", "unknown"),
            "annotations": record["annotations"],
        })
    write_jsonl(paths["tables"] / "sentence_annotations.jsonl", rows)
    write_jsonl(paths["tables"] / "sentence_annotation_failures.jsonl", failures)
    print(
        f"Validated {len(rows)} of {len(rows) + len(failures)} trajectories.",
        flush=True,
    )
    if failures and not allow_missing:
        raise RuntimeError(
            f"{len(failures)} trajectories are missing results; rerun `run` to resume"
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
        validate_cached(config, paths)


if __name__ == "__main__":
    main()
