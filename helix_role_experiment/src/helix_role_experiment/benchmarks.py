from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from .controlled_tasks import ControlledProblem


MATH500_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
    "resolve/main/test.jsonl"
)
INTEGER_ANSWER = re.compile(r"^-?\d+$")


def load_math500_integer_problems(
    count: int,
    levels: set[int],
    seed: int,
    cache_path: Path,
) -> list[ControlledProblem]:
    """Load deterministic, exactly scoreable MATH-500 problems."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not levels:
        raise ValueError("at least one MATH-500 level is required")
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(MATH500_URL, temporary)
            temporary.replace(cache_path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download MATH-500. Enable Kaggle internet or pass "
                "--math500-path pointing to its test.jsonl file."
            ) from exc

    rows = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    eligible = [
        row
        for row in rows
        if int(row["level"]) in levels
        and INTEGER_ANSWER.fullmatch(str(row["answer"]).strip())
    ]
    eligible.sort(
        key=lambda row: (
            str(row.get("unique_id", "")),
            str(row["problem"]),
        )
    )
    if len(eligible) < count:
        raise ValueError(
            f"MATH-500 has only {len(eligible)} integer-answer problems at "
            f"levels {sorted(levels)}; requested {count}"
        )
    rng = np.random.default_rng(seed)
    chosen = [eligible[index] for index in rng.permutation(len(eligible))[:count]]
    return [
        ControlledProblem(
            problem_id=f"math500-{row.get('unique_id', index)}",
            family="iterative_state_machine",
            prompt=str(row["problem"]).strip(),
            states=[],
            answer=str(row["answer"]).strip(),
            metadata={
                "benchmark": "math500",
                "level": int(row["level"]),
                "subject": str(row["subject"]),
                "unique_id": str(row.get("unique_id", index)),
                "numeric_answer": int(str(row["answer"]).strip()),
            },
        )
        for index, row in enumerate(chosen)
    ]
