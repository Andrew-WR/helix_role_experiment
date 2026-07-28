from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator


def discover_decoder_layers(model: Any) -> list[Any]:
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
    raise ValueError(
        "could not discover decoder layers; supported paths are "
        "model.layers, transformer.h, gpt_neox.layers, and model.decoder.layers"
    )


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

