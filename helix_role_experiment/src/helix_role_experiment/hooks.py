from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Callable, Iterator


def _decoder_layer_candidates(model: Any) -> list[Any]:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        current = model
        try:
            for name in path:
                current = getattr(current, name)
            layers = list(current)
            if layers:
                return layers
        except (AttributeError, TypeError):
            continue
    return []


def discover_decoder_layers(model: Any) -> list[Any]:
    queue = [model]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        layers = _decoder_layer_candidates(current)
        if layers:
            return layers
        for attribute in ("get_base_model", "base_model", "model"):
            try:
                value = getattr(current, attribute)
                value = value() if callable(value) and attribute == "get_base_model" else value
                if value is not current:
                    queue.append(value)
            except (AttributeError, TypeError):
                continue
    raise ValueError(
        "could not discover decoder layers after unwrapping common PEFT/base-model "
        "wrappers; supported paths include model.layers, transformer.h, "
        "gpt_neox.layers, and model.decoder.layers"
    )


def select_layer_indices(
    configured: str | list[int],
    total_layers: int,
    adapter_target_layers: list[int] | None = None,
) -> list[int]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if configured == "all":
        return list(range(total_layers))
    if configured == "adapter_neighborhood":
        targets = sorted(set(adapter_target_layers or []))
        if not targets:
            raise ValueError(
                "layers='adapter_neighborhood' requires a target layer declared "
                "in adapter_config.json or inferable from adapter weight keys"
            )
        sentinels = {
            0,
            total_layers // 4,
            total_layers // 2,
            (3 * total_layers) // 4,
            total_layers - 1,
        }
        neighborhood = {
            layer
            for target in targets
            for layer in range(target - 2, target + 3)
            if 0 <= layer < total_layers
        }
        return sorted(sentinels | neighborhood)
    if not isinstance(configured, list):
        raise ValueError(
            "collection.layers must be 'all', 'adapter_neighborhood', or a list"
        )
    indices = sorted(set(int(value) for value in configured))
    invalid = [value for value in indices if value < 0 or value >= total_layers]
    if invalid:
        raise ValueError(
            f"invalid configured layer indices {invalid}; model has {total_layers} layers"
        )
    if not indices:
        raise ValueError("at least one layer must be selected")
    return indices


def adapter_layers_from_config(peft_config: Any) -> list[int]:
    value = getattr(peft_config, "layers_to_transform", None)
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    try:
        return sorted(set(int(layer) for layer in value))
    except TypeError as exc:
        raise ValueError(
            f"unsupported layers_to_transform value in adapter config: {value!r}"
        ) from exc


def infer_adapter_target_layers(
    peft_config: Any,
    adapter_path: str | Path,
) -> list[int]:
    declared = adapter_layers_from_config(peft_config)
    if declared:
        return declared
    keys: list[str] = []
    directory = Path(adapter_path)
    safetensor_files = sorted(directory.glob("adapter_model*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "reading adapter layer names requires safetensors"
            ) from exc
        for filename in safetensor_files:
            with safe_open(str(filename), framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
    else:
        binary_files = sorted(directory.glob("adapter_model*.bin"))
        if binary_files:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("reading adapter .bin files requires torch") from exc
            for filename in binary_files:
                state = torch.load(
                    filename,
                    map_location="cpu",
                    weights_only=True,
                )
                keys.extend(state.keys())
    patterns = (
        re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)"),
        re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),
    )
    layers = set()
    for key in keys:
        for pattern in patterns:
            match = pattern.search(key)
            if match:
                layers.add(int(match.group(1)))
                break
    return sorted(layers)


def hidden_from_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    return hidden


@contextmanager
def registered_hooks(
    layers: list[Any],
    hook_factory: Callable[[int], Callable[..., Any]],
) -> Iterator[None]:
    handles = []
    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook_factory(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()
