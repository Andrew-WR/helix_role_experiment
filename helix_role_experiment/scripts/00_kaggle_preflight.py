from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
from pathlib import Path

from _common import PACKAGE_ROOT
from helix_role_experiment.config import atomic_json, load_config
from helix_role_experiment.hooks import infer_adapter_target_layers
from helix_role_experiment.models import resolve_adapter_path


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def numeric_version(version: str | None) -> tuple[int, ...]:
    if version is None:
        return ()
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast Kaggle/GPU/adapter/base-model validation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-hub-check",
        action="store_true",
        help="Do not resolve the base model config from Hugging Face/local storage",
    )
    parser.add_argument(
        "--output",
        default="/kaggle/working/helix_preflight.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "bitsandbytes",
            "safetensors",
            "sentence_transformers",
        )
    }
    missing = [
        name
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "bitsandbytes",
            "sentence_transformers",
        )
        if packages[name] is None
    ]
    if missing:
        raise RuntimeError(
            f"missing packages {missing}; run pip install -e '.[model,analysis]'"
        )
    if numeric_version(packages["transformers"]) < (5, 14, 1):
        raise RuntimeError(
            "Qwen3.5 requires transformers>=5.14.1 for this experiment, but "
            f"{packages['transformers']} is installed. From the repository's "
            "helix_role_experiment directory run "
            "`python -m pip install -U -e '.[model,analysis]'`, then rerun "
            "preflight in a fresh Python process."
        )
    if numeric_version(packages["peft"]) < (0, 18):
        raise RuntimeError(
            "Transformers 5.x adapter integration requires peft>=0.18, but "
            f"{packages['peft']} is installed. Run "
            "`python -m pip install -U -e '.[model,analysis]'`."
        )

    import torch
    from peft import PeftConfig

    adapter_path = resolve_adapter_path(model_config)
    if not adapter_path:
        raise ValueError("model.adapter_path is required for the Qwen Kaggle configs")
    adapter = Path(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(
            f"adapter directory does not exist: {adapter}. Attach the Kaggle model "
            "input and verify its versioned path."
        )
    adapter_config_path = adapter / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"adapter_config.json is missing from {adapter}; upload the complete "
            "PEFT save_pretrained directory"
        )
    adapter_config = PeftConfig.from_pretrained(str(adapter))
    configured_id = model_config.get("id")
    base_model_id = (
        adapter_config.base_model_name_or_path
        if configured_id in ("auto_from_adapter", "", None)
        else configured_id
    )
    if not base_model_id or str(base_model_id).startswith("REQUIRED_"):
        raise ValueError(
            "could not resolve the base model. Set model.id or correct "
            "base_model_name_or_path in adapter_config.json"
        )

    gpu_rows = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_rows.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(
                    properties.total_memory / (1024**3), 2
                ),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    if model_config.get("quantization", {}).get("load_in_4bit") and not gpu_rows:
        raise RuntimeError("4-bit config requested but CUDA sees no GPUs")

    hub_config = None
    if not args.skip_hub_check:
        from transformers import AutoConfig

        resolved = AutoConfig.from_pretrained(
            base_model_id,
            revision=model_config.get("revision"),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        hub_config = {
            "model_type": getattr(resolved, "model_type", None),
            "hidden_size": getattr(resolved, "hidden_size", None),
            "num_hidden_layers": getattr(resolved, "num_hidden_layers", None),
            "vocab_size": getattr(resolved, "vocab_size", None),
        }

    working = Path("/kaggle/working")
    disk_root = working if working.exists() else PACKAGE_ROOT
    disk = shutil.disk_usage(disk_root)
    report = {
        "config": args.config,
        "packages": packages,
        "adapter_path": str(adapter),
        "adapter_enabled": bool(model_config.get("adapter_enabled", True)),
        "adapter_target_layers": infer_adapter_target_layers(
            adapter_config, adapter
        ),
        "base_model_id": base_model_id,
        "base_model_config": hub_config,
        "gpus": gpu_rows,
        "cuda_available": torch.cuda.is_available(),
        "working_disk_free_gib": round(disk.free / (1024**3), 2),
        "quantization": model_config.get("quantization"),
        "max_memory": model_config.get("max_memory"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, report)
    if len(gpu_rows) < 2:
        print(
            "WARNING: fewer than two CUDA GPUs are visible. Confirm Kaggle "
            "Accelerator is set to GPU T4 x2."
        )
    if not report["adapter_target_layers"]:
        print(
            "WARNING: adapter_config.json does not declare layers_to_transform; "
            "layers='adapter_neighborhood' cannot be resolved."
        )


if __name__ == "__main__":
    main()
