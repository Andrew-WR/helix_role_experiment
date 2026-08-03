from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from helix_role_experiment.config import (
    atomic_json,
    deterministic_id,
    ensure_output_dirs,
    load_config,
    write_jsonl,
)
from helix_role_experiment.readiness import validate_annotations


SCRIPT_DIR = Path(__file__).resolve().parent
API_LABELER_PATH = SCRIPT_DIR / "07b_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location(
    "label_subgoal_events_07b_for_gemini", API_LABELER_PATH
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
LABEL_TO_CODE = {value: key for key, value in CODE_TO_LABEL.items()}

SYSTEM_PROMPT = """You label immutable, pre-segmented reasoning trajectories.
Return one compact code for every sentence, in exact input order.

F = forward_progress: the sentence makes a correct, novel state change that advances any valid solution path.
B = productive_backtrack: the sentence explicitly and usefully corrects, abandons, or returns from a failed path, leaving the reasoning in a more valid state.
N = neutral_support: locally useful explanation, planning, bookkeeping, or restatement without a novel state change.
R = redundant: unnecessarily repeats or re-derives a state already established earlier.
I = incorrect: a false, invalid, unsupported, or path-diverging reasoning step.
A = final_answer: submitted answer or code outside the reasoning section.

Judge correctness using the task, reference, and complete trajectory. The reference is evidence, not the only permitted method. For code, correctness means locally valid program reasoning. Judge novelty relative to every earlier sentence. Ordinary planning, restatement, algebra that establishes nothing useful, and unsupported claims are not F. Use B only for a productive recovery, not hesitation. Use A only outside reasoning. Prefer N over F when the three F requirements are not all satisfied. Put genuinely ambiguous sentence positions in review. Obey the response schema and return no prose."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing trajectory labels with resumable compact Gemini requests"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["prepare", "run", "validate"])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Simultaneous standard API calls; defaults to gemini_labeling.concurrency",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N missing trajectories for a smoke test",
    )
    return parser.parse_args()


def gemini_api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret("GEMINI_API_KEY")
        except Exception as exc:
            raise RuntimeError(
                "Set GEMINI_API_KEY/GOOGLE_API_KEY or create the Kaggle secret "
                "GEMINI_API_KEY"
            ) from exc
    os.environ["GEMINI_API_KEY"] = value
    return value


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
                "description": (
                    f"Exactly {sentence_count} characters, one per sentence in "
                    "order; every character is one of F, B, N, R, I, A."
                ),
            },
            "review": {
                "type": "array",
                "description": (
                    "Zero-based positions whose semantic label remains genuinely "
                    "ambiguous after considering the complete trajectory."
                ),
                "items": {
                    "type": "integer",
                    "minimum": 0,
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
    for index, (sentence, code) in enumerate(zip(trace["sentences"], labels, strict=True)):
        if code == "A" and sentence.get("is_reasoning", False):
            raise ValueError(f"reasoning sentence {index} cannot be final_answer")
        if code in {"F", "B"} and not sentence.get("is_reasoning", False):
            raise ValueError(
                f"non-reasoning sentence {index} cannot be a progress event"
            )
    return {"labels": labels, "review": sorted(review)}


def compact_to_annotations(
    trace: dict[str, Any], payload: dict[str, Any], disagreement: set[int] | None = None
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
            # Final-answer correctness belongs to the benchmark evaluator, not
            # this semantic event labeler. Mark the compatibility field unknown.
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
    prompt = (
        f"DOMAIN: {trace['domain']}\n"
        f"TASK:\n{trace['prompt']}\n\n"
        f"REFERENCE (non-exclusive):\n{trace['reference_answer']}\n\n"
        f"COMPLETE IMMUTABLE TRAJECTORY ({len(lines)} sentences; R=reasoning, "
        f"O=outside reasoning):\n" + "\n".join(lines)
    )
    if prior is not None:
        prompt += (
            "\n\nAUDIT PASS: Inspect every sentence again. Correct any semantic "
            "mistakes in the prior compact labeling, especially false F events, "
            "missed valid progress, productive corrections mislabeled I/N, and "
            "repetition mislabeled progress. Return a complete replacement, not "
            "a diff.\n"
            f"PRIOR LABELS: {prior['labels']}\n"
            f"PRIOR REVIEW POSITIONS: {prior['review']}"
        )
    if correction:
        prompt += (
            "\n\nYour preceding response failed deterministic validation: "
            f"{correction}. Return a corrected complete labeling."
        )
    return prompt


def parsed_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump(mode="json")
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini response contains no structured text")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Gemini structured response is not an object")
    return value


def usage_from_response(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if hasattr(metadata, "model_dump"):
        values = metadata.model_dump(mode="json")
    elif isinstance(metadata, dict):
        values = metadata
    else:
        values = {}
    prompt = int(values.get("prompt_token_count") or 0)
    total = int(values.get("total_token_count") or 0)
    output = int(values.get("candidates_token_count") or 0) + int(
        values.get("thoughts_token_count") or 0
    )
    if not output and total >= prompt:
        output = total - prompt
    return {
        "input_tokens": prompt,
        "output_tokens": output,
        "total_tokens": total or prompt + output,
    }


def pending_traces(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], int]:
    pending = []
    preserved = 0
    for source in API_LABELER.trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        if API_LABELER.materialize_chunked_result(trace, config, paths) is not None:
            preserved += 1
        else:
            pending.append(trace)
    return pending, preserved


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
        if record.get("source") != "gemini_compact_trajectory":
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
                "final_label": CODE_TO_LABEL[code],
                "pass_codes": [
                    value[index] for value in pass_labels
                    if isinstance(value, str) and index < len(value)
                ],
            })
    write_jsonl(paths["tables"] / "gemini_review_queue.jsonl", rows)
    return rows


def prepare(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending, preserved = pending_traces(config, paths)
    settings = config.get("gemini_labeling", {})
    calls_per_trace = 1 + int(settings.get("audit_passes", 1))
    input_tokens = 0.0
    output_tokens = 0.0
    plan = []
    for trace in pending:
        first_chars = len(SYSTEM_PROMPT) + len(trajectory_prompt(trace))
        input_tokens += first_chars / 4
        for _ in range(calls_per_trace - 1):
            input_tokens += (first_chars + len(trace["sentences"]) + 200) / 4
        output_tokens += calls_per_trace * (len(trace["sentences"]) / 2 + 64)
        plan.append({
            "trace_id": trace["trace_id"],
            "task_id": trace["task_id"],
            "domain": trace["domain"],
            "split": trace["split"],
            "sentence_count": len(trace["sentences"]),
            "calls": calls_per_trace,
        })
    rates = settings.get("prices_per_million", {"input": 0.5, "output": 3.0})
    estimate = (
        input_tokens / 1e6 * float(rates["input"])
        + output_tokens / 1e6 * float(rates["output"])
    )
    report = {
        "pending_trajectories": len(pending),
        "preserved_complete_trajectories": preserved,
        "calls_per_trajectory": calls_per_trace,
        "estimated_input_tokens": round(input_tokens),
        "estimated_output_tokens": round(output_tokens),
        "estimated_cost_usd": estimate,
        "hard_budget_usd": float(settings.get("max_cost_usd", 3.0)),
        "model": settings.get("model", "gemini-3-flash-preview"),
    }
    write_jsonl(paths["tables"] / "gemini_label_plan.jsonl", plan)
    atomic_json(paths["tables"] / "gemini_cost_estimate.json", report)
    if estimate > report["hard_budget_usd"]:
        raise RuntimeError(
            f"estimated Gemini cost ${estimate:.2f} exceeds the configured "
            f"${report['hard_budget_usd']:.2f} guard"
        )
    print(
        f"Preserved {preserved} complete trajectories; {len(pending)} remain. "
        f"Planned {calls_per_trace} compact call(s) per trajectory; estimated "
        f"paid-tier cost ${estimate:.2f}.",
        flush=True,
    )
    return pending, report


def generation_config(
    types: Any, settings: dict[str, Any], sentence_count: int, seed: int
) -> Any:
    model = str(settings.get("model", "gemini-3-flash-preview"))
    level = str(settings.get("thinking_level", "minimal")).casefold()
    if model.startswith("gemini-2.5"):
        budget = {"none": 0, "minimal": 1024, "low": 1024,
                  "medium": 8192, "high": 24576}.get(level)
        if budget is None:
            raise ValueError(f"unsupported Gemini thinking level: {level}")
        thinking = types.ThinkingConfig(thinking_budget=budget)
    else:
        mapped = {"none": "MINIMAL", "minimal": "MINIMAL", "low": "LOW",
                  "medium": "MEDIUM", "high": "HIGH"}.get(level)
        if mapped is None:
            raise ValueError(f"unsupported Gemini thinking level: {level}")
        thinking = types.ThinkingConfig(thinking_level=mapped)
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        seed=int(seed),
        max_output_tokens=int(settings.get("max_output_tokens", 4096)),
        thinking_config=thinking,
        response_mime_type="application/json",
        response_json_schema=compact_schema(sentence_count),
    )


async def run_immediate(
    config: dict[str, Any], paths: dict[str, Path], concurrency: int,
    limit: int | None,
) -> None:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Install Gemini support: pip install -U google-genai"
        ) from exc

    pending, _ = prepare(config, paths)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No missing trajectories; rebuilding the validated table.", flush=True)
        API_LABELER.validate_cached(config, paths)
        return

    settings = config.get("gemini_labeling", {})
    model = str(settings.get("model", "gemini-3-flash-preview"))
    attempts = int(settings.get("validation_attempts", 3))
    audit_passes = int(settings.get("audit_passes", 1))
    if attempts <= 0 or audit_passes < 0:
        raise ValueError("validation_attempts must be positive and audit_passes nonnegative")
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0
    failures: list[dict[str, Any]] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    client_root = genai.Client(api_key=gemini_api_key())
    async with client_root.aio as client:

        async def request_pass(
            trace: dict[str, Any], pass_index: int,
            prior: dict[str, Any] | None,
        ) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
            last_error = "unknown failure"
            for attempt in range(1, attempts + 1):
                seed = int(deterministic_id(
                    trace["trace_id"], "gemini", pass_index, attempt
                )[:8], 16)
                prompt = trajectory_prompt(
                    trace, prior=prior,
                    correction=last_error if attempt > 1 else None,
                )
                try:
                    async with semaphore:
                        response = await client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=generation_config(
                                types, settings, len(trace["sentences"]), seed
                            ),
                        )
                    payload = validate_compact_payload(
                        trace, parsed_response(response)
                    )
                    usage = usage_from_response(response)
                    metadata = {
                        "response_id": getattr(response, "response_id", None),
                        "model_version": getattr(response, "model_version", None),
                        "pass_index": pass_index,
                        "attempt": attempt,
                        "payload": payload,
                        "usage": usage,
                    }
                    return payload, usage, metadata
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        await asyncio.sleep(min(2 ** attempt, 8))
            raise RuntimeError(last_error)

        async def label_one(trace: dict[str, Any]) -> None:
            nonlocal completed
            trace_id = str(trace["trace_id"])
            try:
                passes = []
                current = None
                disagreement: set[int] = set()
                local_usage = {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
                }
                for pass_index in range(1 + audit_passes):
                    previous = current
                    current, usage, metadata = await request_pass(
                        trace, pass_index, previous
                    )
                    passes.append(metadata)
                    for key in local_usage:
                        local_usage[key] += usage[key]
                    if previous is not None:
                        disagreement.update(
                            index for index, (before, after) in enumerate(
                                zip(previous["labels"], current["labels"], strict=True)
                            ) if before != after
                        )
                assert current is not None
                annotations = compact_to_annotations(
                    trace, current, disagreement=disagreement
                )
                destination = API_LABELER.full_result_path(paths, trace_id)
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(destination, {
                    "trace_id": trace_id,
                    "source": "gemini_compact_trajectory",
                    "model": model,
                    "usage": local_usage,
                    "compact_labels": current,
                    "audit_disagreement_positions": sorted(disagreement),
                    "agreement_rate": 1.0 - len(disagreement) / len(annotations),
                    "passes": passes,
                    "annotations": annotations,
                })
                async with lock:
                    completed += 1
                    for key in usage_totals:
                        usage_totals[key] += local_usage[key]
                    print(
                        f"[{completed}/{len(pending)}] saved {trace_id}: "
                        f"{len(annotations)} sentences, {len(disagreement)} audit "
                        f"disagreements ({local_usage['input_tokens']} in, "
                        f"{local_usage['output_tokens']} out)",
                        flush=True,
                    )
            except Exception as exc:
                async with lock:
                    failures.append({
                        "trace_id": trace_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(
                        f"FAILED {trace_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

        await asyncio.gather(*(label_one(trace) for trace in pending))

    rates = settings.get("prices_per_million", {"input": 0.5, "output": 3.0})
    cost = (
        usage_totals["input_tokens"] / 1e6 * float(rates["input"])
        + usage_totals["output_tokens"] / 1e6 * float(rates["output"])
    )
    atomic_json(paths["tables"] / "gemini_immediate_run.json", {
        "newly_completed_trajectories": completed,
        "failed": failures,
        "usage": usage_totals,
        "actual_new_cost_usd": cost,
        "concurrency": concurrency,
        "audit_passes": audit_passes,
        "model": model,
    })
    API_LABELER.validate_cached(
        config, paths, allow_missing=limit is not None or bool(failures)
    )
    review_rows = write_review_queue(paths)
    print(
        f"Gemini review queue: {len(review_rows)} reasoning sentences at "
        f"{paths['tables'] / 'gemini_review_queue.jsonl'}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} trajectories failed; rerun the same command to resume"
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
        concurrency = args.concurrency or int(
            config.get("gemini_labeling", {}).get("concurrency", 4)
        )
        try:
            asyncio.run(run_immediate(config, paths, concurrency, args.limit))
        except KeyboardInterrupt:
            print(
                "Paused safely. Completed trajectories were saved atomically; "
                "rerun the same command to resume.",
                flush=True,
            )


if __name__ == "__main__":
    main()
