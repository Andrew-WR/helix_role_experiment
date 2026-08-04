from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any

from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    write_jsonl,
)
from helix_role_experiment.readiness import validate_annotations


SCRIPT_DIR = Path(__file__).resolve().parent
API_LABELER_PATH = SCRIPT_DIR / "07b_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location(
    "label_subgoal_events_07b_for_inkling", API_LABELER_PATH
)
API_LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(API_LABELER)


CODE_TO_LABEL = {
    "F": "forward_progress",
    "B": "productive_backtrack",
    "N": "neutral_support",
    "R": "redundant",
    "I": "incorrect",
    "A": "final_answer",
}

SYSTEM_PROMPT = """You label immutable, pre-segmented reasoning trajectories.
Return one compact code for every sentence, in exact input order.

F = forward_progress: correct and novel state change that advances any valid solution path.
B = productive_backtrack: explicit useful correction or abandonment of a failed path that leaves the reasoning in a more valid state.
N = neutral_support: useful planning, explanation, bookkeeping, or restatement without a novel state change.
R = redundant: unnecessary repetition or re-derivation of an already established state.
I = incorrect: false, invalid, unsupported, or path-diverging reasoning.
A = final_answer: submitted answer or code outside the reasoning section.

Judge correctness from the task, reference, and complete trajectory. The reference is evidence, not the only valid method. Judge novelty against every earlier sentence. Ordinary planning, restatement, algebra that establishes no useful new state, and unsupported claims are not F. Use B only when the recovery is productive, not for hesitation. Use A only outside reasoning. Prefer N over F unless correctness, novelty, and advancement are all satisfied. Add genuinely ambiguous zero-based positions to review. Return only the strict schema, with no prose."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing labels with the Modal Inkling endpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["prepare", "run", "validate"])
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N missing trajectories for a smoke test",
    )
    return parser.parse_args()


def secret(*names: str) -> str:
    if not names:
        raise ValueError("at least one secret name is required")
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        value = None
        for name in names:
            try:
                value = client.get_secret(name)
            except Exception:
                continue
            if value:
                os.environ[name] = value
                return value
    except Exception as exc:
        raise RuntimeError(
            f"Set one of {', '.join(names)} or create the matching Kaggle secret"
        ) from exc
    raise RuntimeError(
        f"None of the Kaggle secrets were found: {', '.join(names)}"
    )


def compact_schema(sentence_count: int) -> dict[str, Any]:
    if sentence_count <= 0:
        raise ValueError("a trajectory must contain at least one sentence")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels", "review"],
        "properties": {
            "labels": {
                "type": "string",
                "minLength": sentence_count,
                "maxLength": sentence_count,
                "pattern": "^[FBNRIA]+$",
                "description": (
                    f"Exactly {sentence_count} characters in sentence order."
                ),
            },
            "review": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "integer", "minimum": 0,
                    "maximum": sentence_count - 1,
                },
            },
        },
    }


def validate_compact_payload(
    trace: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("compact response must be an object")
    labels = payload.get("labels")
    count = len(trace["sentences"])
    if not isinstance(labels, str) or len(labels) != count:
        observed = len(labels) if isinstance(labels, str) else "non-string"
        raise ValueError(
            f"labels must contain exactly {count} characters; observed {observed}"
        )
    unknown = sorted(set(labels) - set(CODE_TO_LABEL))
    if unknown:
        raise ValueError(f"unknown compact label codes: {unknown}")
    review = payload.get("review")
    if not isinstance(review, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in review
    ):
        raise ValueError("review must be a list of integer positions")
    if len(set(review)) != len(review):
        raise ValueError("review positions must be unique")
    if any(value < 0 or value >= count for value in review):
        raise ValueError("review position is outside the trajectory")
    for index, (sentence, code) in enumerate(
        zip(trace["sentences"], labels, strict=True)
    ):
        reasoning = bool(sentence.get("is_reasoning", False))
        if code == "A" and reasoning:
            raise ValueError(f"reasoning sentence {index} cannot be final_answer")
        if code != "A" and not reasoning:
            raise ValueError(
                f"outside-reasoning sentence {index} must be final_answer"
            )
    return {"labels": labels, "review": sorted(review)}


def compact_to_annotations(
    trace: dict[str, Any], payload: dict[str, Any],
    disagreement: set[int] | None = None,
) -> list[dict[str, Any]]:
    compact = validate_compact_payload(trace, payload)
    review = set(compact["review"]) | set(disagreement or ())
    annotations = []
    for index, (sentence, code) in enumerate(
        zip(trace["sentences"], compact["labels"], strict=True)
    ):
        label = CODE_TO_LABEL[code]
        if code in {"F", "B"}:
            correct, novel, advances = "yes", "yes", "yes"
        elif code in {"N", "R"}:
            correct, novel, advances = "yes", "no", "no"
        elif code == "I":
            correct, novel, advances = "no", "no", "no"
        else:
            correct, novel, advances = "uncertain", "no", "no"
            review.add(index)
        annotations.append({
            "sentence_id": sentence["sentence_id"],
            "mathematically_correct": correct,
            "novel": novel,
            "advances_valid_path": advances,
            "primary_label": label,
            "evidence": sentence["text"] if code == "F" else "",
            "state_change": label if code in {"F", "B"} else "",
            "needs_review": index in review,
        })
    return validate_annotations(trace["sentences"], {"annotations": annotations})


def trajectory_prompt(
    trace: dict[str, Any], prior: dict[str, Any] | None = None,
    correction: str | None = None,
) -> str:
    lines = []
    for index, sentence in enumerate(trace["sentences"]):
        section = "R" if sentence.get("is_reasoning", False) else "O"
        lines.append(f"[{index:04d} {section}] {sentence['text']}")
    value = (
        f"DOMAIN: {trace['domain']}\nTASK:\n{trace['prompt']}\n\n"
        f"REFERENCE (non-exclusive):\n{trace['reference_answer']}\n\n"
        f"COMPLETE IMMUTABLE TRAJECTORY ({len(lines)} sentences; "
        "R=reasoning, O=outside reasoning):\n" + "\n".join(lines)
    )
    if prior is not None:
        value += (
            "\n\nAUDIT PASS: Recheck every label, especially false F events, "
            "missed valid progress, productive corrections mislabeled I/N, and "
            "repetition mislabeled F. Return a complete replacement, not a diff."
            f"\nPRIOR LABELS: {prior['labels']}"
            f"\nPRIOR REVIEW POSITIONS: {prior['review']}"
        )
    if correction:
        value += (
            "\n\nThe previous response failed deterministic validation: "
            f"{correction}. Return a corrected complete response."
        )
    return value


def settings(config: dict[str, Any]) -> dict[str, Any]:
    result = {
        "base_url": (
            "https://andrewwafik350--ep-inkling-nvfp4-server.us-west."
            "modal.direct/v1"
        ),
        "model": "thinkingmachines/Inkling-NVFP4",
        "concurrency": 4,
        "validation_attempts": 3,
        "audit_passes": 1,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2048,
        "reasoning_effort": "none",
    }
    result.update(config.get("inkling_labeling", {}))
    return result


def pending_traces(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], int]:
    pending, preserved = [], 0
    for source in API_LABELER.trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        if API_LABELER.materialize_chunked_result(trace, config, paths) is not None:
            preserved += 1
        else:
            pending.append(trace)
    return pending, preserved


def prepare(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending, preserved = pending_traces(config, paths)
    values = settings(config)
    calls = 1 + int(values["audit_passes"])
    plan = []
    estimated_input = 0
    for trace in pending:
        characters = len(SYSTEM_PROMPT) + len(trajectory_prompt(trace))
        estimated_input += characters / 4
        plan.append({
            "trace_id": trace["trace_id"], "task_id": trace["task_id"],
            "domain": trace["domain"], "split": trace["split"],
            "sentence_count": len(trace["sentences"]),
            "estimated_input_tokens": round(characters / 4), "calls": calls,
        })
    report = {
        "pending_trajectories": len(pending),
        "preserved_complete_trajectories": preserved,
        "calls_per_trajectory": calls,
        "estimated_input_tokens_first_pass": round(estimated_input),
        "model": values["model"], "base_url": values["base_url"],
    }
    write_jsonl(paths["tables"] / "inkling_label_plan.jsonl", plan)
    atomic_json(paths["tables"] / "inkling_label_plan.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return pending, report


def response_payload(completion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    choice = completion.choices[0]
    content = choice.message.content
    if not content:
        raise ValueError("Inkling returned no message content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Inkling response is not a JSON object")
    usage = getattr(completion, "usage", None)
    usage_row = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    metadata = {
        "response_id": getattr(completion, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None),
    }
    return payload, {"usage": usage_row, "metadata": metadata}


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
        if record.get("source") != "inkling_compact_trajectory":
            continue
        trace = traces.get(str(record.get("trace_id", source.stem)))
        compact = record.get("compact_labels") or {}
        labels = compact.get("labels")
        if trace is None or not isinstance(labels, str):
            continue
        review = set(compact.get("review") or ()) | set(
            record.get("audit_disagreement_positions") or ()
        )
        for index in sorted(review):
            if 0 <= index < len(trace["sentences"]) and labels[index] != "A":
                rows.append({
                    "trace_id": trace["trace_id"], "task_id": trace["task_id"],
                    "domain": trace["domain"], "split": trace["split"],
                    "sentence_index": index,
                    "sentence_id": trace["sentences"][index]["sentence_id"],
                    "text": trace["sentences"][index]["text"],
                    "final_code": labels[index],
                    "final_label": CODE_TO_LABEL[labels[index]],
                })
    write_jsonl(paths["tables"] / "inkling_review_queue.jsonl", rows)
    return rows


async def run_immediate(
    config: dict[str, Any], paths: dict[str, Path], concurrency: int,
    limit: int | None,
) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install labeling support: pip install -U openai") from exc
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    pending, _ = prepare(config, paths)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No missing trajectories; rebuilding validated tables.", flush=True)
        API_LABELER.validate_cached(config, paths)
        return

    values = settings(config)
    attempts = int(values["validation_attempts"])
    audit_passes = int(values["audit_passes"])
    if attempts <= 0 or audit_passes < 0:
        raise ValueError("validation_attempts must be positive and audit_passes nonnegative")
    client = AsyncOpenAI(
        base_url=str(values["base_url"]), api_key="unused",
        default_headers={
            "Modal-Key": secret("Modal-Key", "MODAL_PROXY_TOKEN_ID"),
            "Modal-Secret": secret(
                "Modal-Secre", "Modal-Secret", "MODAL_PROXY_TOKEN_SECRET"
            ),
        },
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0
    failures = []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def request_pass(
        trace: dict[str, Any], prior: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error = "unknown failure"
        for attempt in range(1, attempts + 1):
            try:
                prompt = trajectory_prompt(
                    trace, prior=prior,
                    correction=last_error if attempt > 1 else None,
                )
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=str(values["model"]),
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=float(values["temperature"]),
                        max_tokens=int(values["max_tokens"]),
                        top_p=float(values["top_p"]), stream=False,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "trajectory_labels", "strict": True,
                                "schema": compact_schema(len(trace["sentences"])),
                            },
                        },
                        extra_body={
                            "reasoning_effort": str(values["reasoning_effort"])
                        },
                    )
                payload, response_info = response_payload(completion)
                return validate_compact_payload(trace, payload), response_info
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    await asyncio.sleep(min(2 ** attempt + random.random(), 10))
        raise RuntimeError(last_error)

    async def label_one(trace: dict[str, Any]) -> None:
        nonlocal completed
        trace_id = str(trace["trace_id"])
        try:
            passes = []
            disagreement: set[int] = set()
            current = None
            local_usage = {key: 0 for key in totals}
            for _ in range(1 + audit_passes):
                previous = current
                current, info = await request_pass(trace, previous)
                passes.append({**info["metadata"], "payload": current,
                               "usage": info["usage"]})
                for key in local_usage:
                    local_usage[key] += info["usage"][key]
                if previous is not None:
                    disagreement.update(
                        index for index, (before, after) in enumerate(
                            zip(previous["labels"], current["labels"], strict=True)
                        ) if before != after
                    )
            assert current is not None
            annotations = compact_to_annotations(trace, current, disagreement)
            destination = API_LABELER.full_result_path(paths, trace_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, {
                "trace_id": trace_id, "source": "inkling_compact_trajectory",
                "model": values["model"], "usage": local_usage,
                "compact_labels": current,
                "audit_disagreement_positions": sorted(disagreement),
                "agreement_rate": 1.0 - len(disagreement) / len(annotations),
                "passes": passes, "annotations": annotations,
            })
            async with lock:
                completed += 1
                for key in totals:
                    totals[key] += local_usage[key]
                print(
                    f"[{completed}/{len(pending)}] saved {trace_id}: "
                    f"{len(annotations)} sentences, {len(disagreement)} audit "
                    f"disagreements ({local_usage['input_tokens']} in, "
                    f"{local_usage['output_tokens']} out)", flush=True,
                )
        except Exception as exc:
            async with lock:
                failures.append({
                    "trace_id": trace_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"FAILED {trace_id}: {type(exc).__name__}: {exc}", flush=True)

    try:
        await asyncio.gather(*(label_one(trace) for trace in pending))
    finally:
        await client.close()
    atomic_json(paths["tables"] / "inkling_immediate_run.json", {
        "newly_completed_trajectories": completed, "failed": failures,
        "usage": totals, "concurrency": concurrency,
        "audit_passes": audit_passes, "model": values["model"],
    })
    API_LABELER.validate_cached(
        config, paths, allow_missing=limit is not None or bool(failures)
    )
    rows = write_review_queue(paths)
    print(f"Inkling review queue: {len(rows)} rows", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} trajectories failed; rerun the command to resume"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.command == "prepare":
        prepare(config, paths)
    elif args.command == "validate":
        API_LABELER.validate_cached(config, paths)
        write_review_queue(paths)
    else:
        concurrency = args.concurrency or int(settings(config)["concurrency"])
        try:
            asyncio.run(run_immediate(config, paths, concurrency, args.limit))
        except KeyboardInterrupt:
            print(
                "Paused safely. Completed trajectories were saved atomically; "
                "rerun the same command to resume.", flush=True,
            )


if __name__ == "__main__":
    main()
