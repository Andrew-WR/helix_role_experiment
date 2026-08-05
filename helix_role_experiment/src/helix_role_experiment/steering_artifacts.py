from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FINGERPRINT_VERSION = 1
READINESS_STOP_REGEX = (
    r"(?is)</think>\s*(?:FINAL:\s*\S[^\r\n]*(?:\r?\n|<\|im_end\|>)|"
    r"FINAL_CODE:\s*```(?:python|py)?.*?```)"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def steering_run_identity(
    config: dict[str, Any], probe_path: str | Path, stop_regex: str,
) -> dict[str, Any]:
    probe_sha256 = file_sha256(probe_path)
    collection = config["collection"]
    model = config["model"]
    scientific_model = {
        key: model.get(key) for key in (
            "backend", "id", "revision", "adapter_id", "adapter_path",
            "dtype", "attn_implementation", "quantization",
            "chat_template_kwargs",
        ) if key in model
    }
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "probe_sha256": probe_sha256,
        "model": scientific_model,
        "intervention": config["intervention"],
        "generation": {
            key: collection.get(key) for key in (
                "max_new_tokens", "temperature", "top_p", "top_k",
                "stop_check_interval",
            )
        },
        "study_seed": int(config["study"]["seed"]),
        "stop_regex": stop_regex,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return {
        "steering_run_fingerprint": fingerprint,
        "probe_sha256": probe_sha256,
        "fingerprint_version": FINGERPRINT_VERSION,
        "fingerprint_payload": payload,
    }


def valid_steering_artifact(path: str | Path, fingerprint: str) -> bool:
    source = Path(path)
    if not source.exists():
        return False
    try:
        row = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(row.get("steering_run_fingerprint", "")) == str(fingerprint)
