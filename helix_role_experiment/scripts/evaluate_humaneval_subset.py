from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import io
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from helix_role_experiment.config import ensure_output_dirs, load_config, read_jsonl
from helix_role_experiment.steering_artifacts import (
    READINESS_STOP_REGEX,
    steering_run_identity,
    valid_steering_artifact,
)


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


class _RenameGeneratedEntry(ast.NodeTransformer):
    def __init__(self, original: str, replacement: str):
        self.original = original
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.original:
            return ast.copy_location(
                ast.Name(id=self.replacement, ctx=node.ctx), node
            )
        return node


def _entry_call_arguments(function: ast.FunctionDef) -> tuple[list[ast.expr], list[ast.keyword]]:
    arguments: list[ast.expr] = [
        ast.Name(id=value.arg, ctx=ast.Load())
        for value in (*function.args.posonlyargs, *function.args.args)
    ]
    if function.args.vararg is not None:
        arguments.append(ast.Starred(
            value=ast.Name(id=function.args.vararg.arg, ctx=ast.Load()),
            ctx=ast.Load(),
        ))
    keywords = [
        ast.keyword(arg=value.arg, value=ast.Name(id=value.arg, ctx=ast.Load()))
        for value in function.args.kwonlyargs
    ]
    if function.args.kwarg is not None:
        keywords.append(ast.keyword(
            arg=None,
            value=ast.Name(id=function.args.kwarg.arg, ctx=ast.Load()),
        ))
    return arguments, keywords


def normalize_standalone_completion(
    prompt: str, completion: str, entry_point: str,
) -> tuple[str | None, str]:
    """Convert a standalone function solution into a strict continuation.

    The generated entry function becomes a uniquely named nested helper. This
    preserves its own signature and recursion. Imports, constants, classes,
    and helper functions are moved into the supplied HumanEval function's
    scope, after which the helper is called with the supplied arguments.
    """
    valid, _ = completion_format(completion, entry_point)
    if valid:
        return completion, "already_valid_completion"
    try:
        prompt_tree = ast.parse(prompt)
        generated_tree = ast.parse(completion)
    except SyntaxError as exc:
        return None, f"unparseable_python:{exc.msg}"
    supplied = next((
        node for node in reversed(prompt_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ), None)
    generated = next((
        node for node in reversed(generated_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ), None)
    if supplied is None:
        return None, "supplied_entry_function_not_found"
    if generated is None:
        return None, "generated_entry_function_not_found"

    occupied = {
        node.name for node in generated_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    helper_name = f"_normalized_{entry_point}"
    suffix = 2
    while helper_name in occupied:
        helper_name = f"_normalized_{entry_point}_{suffix}"
        suffix += 1

    safe_support = (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign,
    )
    support = [
        copy.deepcopy(node) for node in generated_tree.body
        if isinstance(node, safe_support)
        and not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        )
    ]
    nested_entry = copy.deepcopy(generated)
    nested_entry.name = helper_name
    renamer = _RenameGeneratedEntry(entry_point, helper_name)
    support = [renamer.visit(node) for node in support]
    nested_entry = renamer.visit(nested_entry)
    call_args, call_keywords = _entry_call_arguments(supplied)
    invoke = ast.Return(value=ast.Call(
        func=ast.Name(id=helper_name, ctx=ast.Load()),
        args=call_args,
        keywords=call_keywords,
    ))
    module = ast.Module(body=[*support, nested_entry, invoke], type_ignores=[])
    ast.fix_missing_locations(module)
    normalized = textwrap.indent(ast.unparse(module), "    ") + "\n"
    valid, reason = completion_format(normalized, entry_point)
    if not valid:
        return None, f"normalizer_produced_invalid_format:{reason}"
    return normalized, "standalone_entry_wrapped"


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


def atomic_evaluator_input(destination: Path, rows: list[dict[str, Any]]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    atomic_jsonl(temporary, rows)
    changed = (
        not destination.exists()
        or destination.read_bytes() != temporary.read_bytes()
    )
    temporary.replace(destination)
    if changed:
        Path(str(destination) + "_results.jsonl").unlink(missing_ok=True)


def rebuild_sample_files(
    paths: dict[str, Path], fingerprint: str | None = None
) -> dict[str, int]:
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
        if fingerprint is not None and not valid_steering_artifact(
            source, fingerprint
        ):
            continue
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
            atomic_evaluator_input(destination, rows)
        else:
            destination.unlink(missing_ok=True)
            Path(str(destination) + "_results.jsonl").unlink(missing_ok=True)
        counts[condition] = len(rows)
    return counts


def evaluate_file(
    source: Path,
    tasks: dict[str, dict[str, Any]],
    timeout: float,
) -> tuple[Path, int, int, int]:
    samples = read_jsonl(source)
    if not samples:
        raise RuntimeError(f"{source} contains no samples")
    results = []
    passed = 0
    normalized_passed = 0
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
        normalized_completion, normalization_reason = normalize_standalone_completion(
            str(tasks[task_id]["prompt"]),
            str(sample.get("completion", "")),
            str(tasks[task_id]["metadata"]["entry_point"]),
        )
        if normalized_completion is None:
            normalized_outcome = {
                "passed": False,
                "result": "normalization unavailable",
            }
            normalized_format_valid = False
            normalized_format_reason = normalization_reason
        elif normalization_reason == "already_valid_completion":
            normalized_outcome = outcome
            normalized_format_valid = format_valid
            normalized_format_reason = format_reason
        else:
            normalized_outcome = check_program(
                build_program(tasks[task_id], normalized_completion),
                timeout,
            )
            normalized_format_valid, normalized_format_reason = completion_format(
                normalized_completion,
                str(tasks[task_id]["metadata"]["entry_point"]),
            )
        normalized_functional = bool(normalized_outcome["passed"])
        normalized_strict = bool(
            normalized_functional and normalized_format_valid
        )
        result = {
            **sample,
            **outcome,
            "functional_passed": functional_passed,
            "format_valid": format_valid,
            "format_reason": format_reason,
            "passed": strict_passed,
            "normalized_completion": normalized_completion,
            "normalization_applied": bool(
                normalized_completion is not None
                and normalization_reason != "already_valid_completion"
            ),
            "normalization_reason": normalization_reason,
            "normalized_functional_passed": normalized_functional,
            "normalized_format_valid": normalized_format_valid,
            "normalized_format_reason": normalized_format_reason,
            "normalized_passed": normalized_strict,
            "normalized_execution_result": normalized_outcome["result"],
            "completion_id": 0,
        }
        results.append(result)
        passed += int(strict_passed)
        normalized_passed += int(normalized_strict)
        print(
            f"[{source.stem} {index}/{len(samples)}] {task_id}: "
            f"functional={functional_passed}; strict={strict_passed}; "
            f"normalized={normalized_strict}; "
            f"format={format_reason}; execution={outcome['result']}",
            flush=True,
        )
    destination = Path(str(source) + "_results.jsonl")
    atomic_jsonl(destination, results)
    return destination, passed, normalized_passed, len(results)


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
    identity = steering_run_identity(
        config,
        paths["models"] / "readiness_survival_probe.npz",
        READINESS_STOP_REGEX,
    )
    rebuilt = rebuild_sample_files(
        paths, str(identity["steering_run_fingerprint"])
    )
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
        destination, passed, normalized_passed, total = evaluate_file(
            source, tasks, args.timeout
        )
        print(
            f"{condition}: {passed}/{total} strict passes; "
            f"{normalized_passed}/{total} normalized passes; wrote {destination}",
            flush=True,
        )


if __name__ == "__main__":
    main()
