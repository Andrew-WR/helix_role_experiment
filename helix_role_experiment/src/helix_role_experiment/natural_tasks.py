from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"problem_id", "prompt", "reference_answer", "task_family"}


def load_natural_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            tasks.append(row)
    return tasks

