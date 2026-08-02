from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import tempfile
from pathlib import Path
from typing import Any

from helix_role_experiment.config import ensure_output_dirs, load_config, read_jsonl


DEFAULT_CONDITIONS = ("baseline", "gated", "always", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the experiment's held-out HumanEval subset"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", action="append", choices=DEFAULT_CONDITIONS)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def build_program(task: dict[str, Any], completion: str) -> str:
    metadata = task.get("metadata") or {}
    return (
        str(task["prompt"])
        + str(completion)
        + "\n"
        + str(metadata["test"])
        + "\n"
        + f"check({metadata['entry_point']})"
    )


def completion_format(completion: str, entry_point: str) -> tuple[bool, str]:
    """Enforce the completion-only contract used in the generation prompt."""
    value = completion.lstrip("\r\n")
    if not value.strip():
        return False, "empty_completion"
    if any(marker in value for marker in ("```", "<|", "FINAL_CODE:")):
        return False, "generation_marker_or_fence"
    if not value[0].isspace():
        return False, "not_an_indented_prompt_continuation"
    definition = re.compile(
        rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    )
    if definition.search(value):
        return False, "repeats_supplied_function"
    return True, "valid_completion_only_format"


def _execute(program: str, result: Any) -> None:
    """Run one candidate inside a disposable child and temporary directory."""
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            import resource

            two_gib = 2 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (two_gib, two_gib))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
        except (ImportError, OSError, ValueError):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        exec(program, {})
            finally:
                os.chdir(previous)
        result.put({"passed": True, "result": "passed"})
    except BaseException as exc:
        result.put({
            "passed": False,
            "result": f"failed: {type(exc).__name__}: {exc}",
        })


def check_program(program: str, timeout: float) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    context = mp.get_context("spawn")
    result = context.Queue(maxsize=1)
    process = context.Process(target=_execute, args=(program, result))
    process.start()
    process.join(timeout + 1.0)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()
        process.join()
    try:
        return dict(result.get(timeout=0.2))
    except queue.Empty:
        return {"passed": False, "result": "timed out"}
    finally:
        result.close()


def atomic_jsonl(destination: Path, rows: list[dict[str, Any]]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def rebuild_sample_files(paths: dict[str, Path]) -> dict[str, int]:
    """Derive evaluator inputs from atomic trace checkpoints without loading Qwen."""
    grouped: dict[str, list[dict[str, str]]] = {
        condition: [] for condition in DEFAULT_CONDITIONS
    }
    for source in sorted((paths["traces"] / "readiness_baseline").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        if row.get("split") == "test" and row.get("domain") == "code":
            grouped["baseline"].append({
                "task_id": str(row["task_id"]),
                "completion": str(row.get("humaneval_completion") or ""),
            })
    for source in sorted((paths["traces"] / "readiness_steering").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8"))
        condition = str(row.get("condition"))
        if row.get("domain") == "code" and condition in grouped:
            grouped[condition].append({
                "task_id": str(row["task_id"]),
                "completion": str(row.get("humaneval_completion") or ""),
            })
    counts = {}
    for condition, rows in grouped.items():
        rows.sort(key=lambda row: row["task_id"])
        destination = paths["tables"] / f"humaneval_{condition}.jsonl"
        if rows:
            atomic_jsonl(destination, rows)
        else:
            destination.unlink(missing_ok=True)
            Path(str(destination) + "_results.jsonl").unlink(missing_ok=True)
        counts[condition] = len(rows)
    return counts


def evaluate_file(
    source: Path,
    tasks: dict[str, dict[str, Any]],
    timeout: float,
) -> tuple[Path, int, int]:
    samples = read_jsonl(source)
    if not samples:
        raise RuntimeError(f"{source} contains no samples")
    results = []
    passed = 0
    for index, sample in enumerate(samples, 1):
        task_id = str(sample["task_id"])
        if task_id not in tasks:
            raise RuntimeError(f"{source}: unknown or non-code task {task_id}")
        outcome = check_program(
            build_program(tasks[task_id], str(sample.get("completion", ""))),
            timeout,
        )
        format_valid, format_reason = completion_format(
            str(sample.get("completion", "")),
            str(tasks[task_id]["metadata"]["entry_point"]),
        )
        functional_passed = bool(outcome["passed"])
        strict_passed = bool(functional_passed and format_valid)
        result = {
            **sample,
            **outcome,
            "functional_passed": functional_passed,
            "format_valid": format_valid,
            "format_reason": format_reason,
            "passed": strict_passed,
            "completion_id": 0,
        }
        results.append(result)
        passed += int(strict_passed)
        print(
            f"[{source.stem} {index}/{len(samples)}] {task_id}: "
            f"functional={functional_passed}; strict={strict_passed}; "
            f"format={format_reason}; execution={outcome['result']}",
            flush=True,
        )
    destination = Path(str(source) + "_results.jsonl")
    atomic_jsonl(destination, results)
    return destination, passed, len(results)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    tasks = {
        str(row["task_id"]): row
        for row in read_jsonl(paths["tables"] / "readiness_tasks.jsonl")
        if row["domain"] == "code"
    }
    conditions = tuple(args.condition or DEFAULT_CONDITIONS)
    rebuilt = rebuild_sample_files(paths)
    print(f"Rebuilt HumanEval inputs from saved traces: {rebuilt}", flush=True)
    print(
        "SECURITY: this executes model-generated Python. Run only in an "
        "isolated Kaggle session with Internet disabled and no secrets attached.",
        flush=True,
    )
    for condition in conditions:
        source = paths["tables"] / f"humaneval_{condition}.jsonl"
        if not source.exists():
            raise RuntimeError(
                f"missing {source}; finish or safely pause 07d so it exports samples"
            )
        destination, passed, total = evaluate_file(source, tasks, args.timeout)
        print(
            f"{condition}: {passed}/{total} strict passes; wrote {destination}",
            flush=True,
        )


if __name__ == "__main__":
    main()
