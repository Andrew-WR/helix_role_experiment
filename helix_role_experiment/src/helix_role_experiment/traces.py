from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import deterministic_id


@dataclass
class TraceRecord:
    request_id: str
    problem_id: str
    task_family: str
    condition: str
    split: str
    layer: int
    prompt_token_count: int
    token_ids: list[int]
    tokens: list[str]
    activation_file: str
    generated_token_count: int
    reached_eos: bool
    truncated: bool
    model_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    seed: int
    state_ids: list[str] = field(default_factory=list)
    structural_progress: list[float] = field(default_factory=list)
    remaining_distance: list[float] = field(default_factory=list)
    operation: list[str] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    eos_logit: list[float] = field(default_factory=list)
    termination_allowed: list[bool] = field(default_factory=list)
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.generated_token_count != len(self.token_ids):
            raise ValueError("generated_token_count does not match token_ids")
        if self.tokens and len(self.tokens) != len(self.token_ids):
            raise ValueError("tokens and token_ids are misaligned")
        for name in (
            "state_ids",
            "structural_progress",
            "remaining_distance",
            "operation",
            "confidence",
            "eos_logit",
            "termination_allowed",
        ):
            values = getattr(self, name)
            if values and len(values) != self.generated_token_count:
                raise ValueError(f"{name} is not token aligned")


def split_for_problem(problem_id: str, study_seed: int = 0) -> str:
    digest = int(deterministic_id(study_seed, problem_id)[:12], 16) / float(16**12)
    if digest < 0.50:
        return "calibration"
    if digest < 0.70:
        return "train"
    if digest < 0.80:
        return "validation"
    return "test"


class TraceStore:
    """One compressed activation shard per trace plus an auditable JSONL manifest."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"

    def write(
        self,
        record: TraceRecord,
        activations: np.ndarray,
        overwrite: bool = False,
    ) -> TraceRecord:
        record.validate()
        array = np.asarray(activations)
        expected = (record.generated_token_count,)
        if array.ndim != 2 or array.shape[:1] != expected:
            raise ValueError(
                f"activations must be [tokens, hidden], got {array.shape}; "
                f"expected first dimension {record.generated_token_count}"
            )
        if not np.isfinite(array).all():
            raise ValueError("activations contain nonfinite values")
        shard = self.root / f"{record.request_id}.npz"
        if shard.exists() and not overwrite:
            return record
        temporary = shard.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, activations=array)
        temporary.replace(shard)
        record.activation_file = shard.name
        existing = {row["request_id"] for row in self.read_manifest()} if self.manifest_path.exists() else set()
        if record.request_id not in existing:
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return record

    def read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        rows = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"corrupt manifest line {line_number}: {self.manifest_path}"
                        ) from exc
        return rows

    def load_activations(self, row: TraceRecord | dict[str, Any]) -> np.ndarray:
        filename = row.activation_file if isinstance(row, TraceRecord) else row["activation_file"]
        with np.load(self.root / filename, allow_pickle=False) as data:
            return np.asarray(data["activations"], dtype=np.float64)

    def iter_traces(self) -> Iterable[tuple[dict[str, Any], np.ndarray]]:
        for row in self.read_manifest():
            yield row, self.load_activations(row)
