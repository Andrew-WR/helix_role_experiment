from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0.0"


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    config["_config_path"] = str(source.resolve())
    return config


def canonical_config(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    if isinstance(clean.get("model"), dict):
        model = dict(clean["model"])
        # Kaggle mount aliases are environment routing, not a scientific
        # configuration change. The resolved adapter remains recorded in
        # preflight and trace metadata.
        model.pop("adapter_fallback_paths", None)
        # Qwen3.6's template default is thinking mode. Recording that default
        # explicitly should not invalidate activations collected under the
        # identical implicit setting; disabling it remains hash-significant.
        if (
            model.get("id") == "Qwen/Qwen3.6-27B"
            and model.get("chat_template_kwargs")
            == {"enable_thinking": True}
        ):
            model.pop("chat_template_kwargs")
        clean["model"] = model
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()[:16]


def deterministic_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def environment_record(config: dict[str, Any]) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
    ):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = None
    try:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(config["_config_path"]).parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        repository_commit = None
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(config),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "pid": os.getpid(),
        "repository_commit": repository_commit,
    }


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output"]["root"])
    paths = {
        "root": root,
        "traces": root / "traces",
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with Path(path).open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows
