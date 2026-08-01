from __future__ import annotations

import gzip
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmarks import load_math500_integer_problems


HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/"
    "data/HumanEval.jsonl.gz"
)
SWE_EVO_NOT_SUPPORTED_REASON = (
    "SWE-EVO requires an interactive OpenHands or SWE-agent scaffold over "
    "repository snapshots. A one-shot activation collector would not be a "
    "valid SWE-EVO evaluation; use HumanEval for the offline-scored coding "
    "domain in this resource-limited experiment."
)


@dataclass(frozen=True)
class ReadinessTask:
    task_id: str
    domain: str
    prompt: str
    reference_answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReadinessTask":
        return cls(
            task_id=str(value["task_id"]),
            domain=str(value["domain"]),
            prompt=str(value["prompt"]),
            reference_answer=str(value.get("reference_answer", "")),
            metadata=dict(value.get("metadata") or {}),
        )


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Enable Kaggle internet or place the "
            f"dataset at {destination}."
        ) from exc


def load_humaneval_tasks(
    count: int,
    seed: int,
    cache_path: str | Path,
) -> list[ReadinessTask]:
    if count <= 0:
        raise ValueError("count must be positive")
    source = Path(cache_path)
    if not source.is_file():
        _download(HUMANEVAL_URL, source)
    opener = gzip.open if source.suffix == ".gz" else open
    rows = []
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: str(row["task_id"]))
    if count > len(rows):
        raise ValueError(
            f"HumanEval contains {len(rows)} tasks; requested {count}"
        )
    rng = np.random.default_rng(int(seed))
    selected = [rows[index] for index in rng.permutation(len(rows))[:count]]
    return [
        ReadinessTask(
            task_id=str(row["task_id"]),
            domain="code",
            prompt=str(row["prompt"]),
            reference_answer=str(row["canonical_solution"]),
            metadata={
                "benchmark": "human_eval",
                "entry_point": str(row["entry_point"]),
                "test": str(row["test"]),
            },
        )
        for row in selected
    ]


def load_mixed_readiness_tasks(
    math_count: int,
    code_count: int,
    seed: int,
    math500_path: str | Path,
    humaneval_path: str | Path,
    math_levels: set[int] | None = None,
) -> list[ReadinessTask]:
    levels = math_levels or {1, 2, 3, 4, 5}
    math = load_math500_integer_problems(
        math_count,
        levels,
        seed,
        Path(math500_path),
    )
    tasks = [
        ReadinessTask(
            task_id=problem.problem_id,
            domain="math",
            prompt=problem.prompt,
            reference_answer=problem.answer,
            metadata=dict(problem.metadata),
        )
        for problem in math
    ]
    tasks.extend(
        load_humaneval_tasks(code_count, seed + 1, humaneval_path)
    )
    tasks.sort(key=lambda task: (task.domain, task.task_id))
    return tasks


def readiness_prompt(task: ReadinessTask) -> str:
    common = (
        "Work through the task in the model-provided thinking section. "
        "Use concise, nonredundant sentences, with one operation, inference, "
        "or correction per sentence. Preserve mathematical and code syntax. "
    )
    if task.domain == "math":
        return (
            common
            + "After closing the thinking section, output exactly one line "
            "of the form `FINAL: <answer>` and nothing else.\n\n"
            + f"Question: {task.prompt}"
        )
    if task.domain == "code":
        return (
            common
            + "After closing the thinking section, write `FINAL_CODE:` and "
            "then exactly one Python code block containing only the completion "
            "that begins immediately after the supplied HumanEval prompt. Do "
            "not repeat the supplied prompt and do not execute the code.\n\n"
            + "HumanEval prompt:\n"
            + task.prompt
        )
    raise ValueError(f"unknown readiness task domain {task.domain!r}")


def extract_humaneval_completion(text: str) -> str:
    """Extract a HumanEval completion without executing model-generated code."""

    lower = text.casefold()
    marker = lower.rfind("final_code:")
    value = text[marker + len("final_code:") :] if marker >= 0 else text
    fenced = re.search(
        r"```(?:python|py)?\s*\n(?P<code>.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group("code")
    # Leading indentation is part of a HumanEval completion and must survive.
    value = value.replace("<|im_end|>", "").strip("\r\n")
    return value + ("\n" if value else "")
