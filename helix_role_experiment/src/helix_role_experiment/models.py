from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import deterministic_id
from .hooks import discover_decoder_layers, hidden_from_output, registered_hooks, replace_hidden
from .subspaces import orthonormalize


@dataclass
class CollectedGeneration:
    token_ids: list[int]
    tokens: list[str]
    activations_by_layer: dict[int, np.ndarray]
    eos_logits: list[float]
    reached_eos: bool
    prompt_token_count: int
    text: str


class SyntheticActivationBackend:
    """Deterministic no-download backend for algebraic end-to-end validation.

    Its latent role is configured and therefore cannot provide scientific
    evidence about a language model.
    """

    def __init__(
        self,
        hidden_size: int,
        layers: int,
        seed: int,
        role: str = "mixed",
        noise: float = 0.08,
    ):
        if hidden_size < 8 or layers < 1:
            raise ValueError("synthetic backend requires hidden_size>=8 and layers>=1")
        self.hidden_size = hidden_size
        self.layers = layers
        self.seed = seed
        self.role = role
        self.noise = noise
        rng = np.random.default_rng(seed)
        self.planes = [orthonormalize(rng.normal(size=(hidden_size, 2)), 2) for _ in range(layers)]
        self.position_directions = [
            orthonormalize(rng.normal(size=(hidden_size, 1)), 1)[:, 0] for _ in range(layers)
        ]
        self.eos_directions = [
            orthonormalize(rng.normal(size=(hidden_size, 1)), 1)[:, 0] for _ in range(layers)
        ]
        self.centers = rng.normal(scale=0.25, size=(layers, hidden_size))

    def _role_phase(self, position: np.ndarray, progress: np.ndarray) -> np.ndarray:
        if self.role == "semantic_progress":
            latent = progress
        elif self.role == "sequence_clock":
            latent = position
        elif self.role == "termination":
            latent = np.clip((position - 0.65) / 0.35, 0.0, 1.0)
        elif self.role == "mixed":
            latent = 0.55 * progress + 0.35 * position + 0.10 * np.sin(np.pi * progress)
        else:
            raise ValueError(f"unknown synthetic role: {self.role}")
        return 2.0 * np.pi * latent

    def generate(
        self,
        request_id: str,
        length: int,
        progress: np.ndarray,
        position: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        termination_allowed: np.ndarray | None = None,
    ) -> CollectedGeneration:
        if length < 8:
            raise ValueError("synthetic trace length must be at least eight")
        progress = np.asarray(progress, dtype=np.float64)
        if progress.shape != (length,):
            raise ValueError("progress must have shape [length]")
        confidence = (
            np.asarray(confidence, dtype=np.float64)
            if confidence is not None
            else 0.4 + 0.5 * progress
        )
        termination_allowed = (
            np.asarray(termination_allowed, dtype=bool)
            if termination_allowed is not None
            else np.zeros(length, dtype=bool)
        )
        position = (
            np.asarray(position, dtype=np.float64)
            if position is not None
            else np.arange(length, dtype=np.float64) / max(1, length - 1)
        )
        if position.shape != (length,):
            raise ValueError("position must have shape [length]")
        phase = self._role_phase(position, progress)
        rng = np.random.default_rng(int(deterministic_id(self.seed, request_id)[:16], 16))
        activations: dict[int, np.ndarray] = {}
        eos_logits = -5.0 + 7.0 * confidence + 2.0 * termination_allowed + 1.5 * position
        for layer in range(self.layers):
            strength = 0.4 + 1.6 * (layer + 1) / self.layers
            coords = np.column_stack((np.cos(phase), np.sin(phase))) * strength
            base = self.centers[layer] + coords @ self.planes[layer].T
            base += (0.15 + 0.1 * layer / self.layers) * position[:, None] * self.position_directions[layer]
            base += 0.05 * eos_logits[:, None] * self.eos_directions[layer]
            base += rng.normal(scale=self.noise, size=base.shape)
            activations[layer] = base.astype(np.float32)
        token_ids = list(range(1000, 1000 + length))
        return CollectedGeneration(
            token_ids=token_ids,
            tokens=[f"<synth-{token}>" for token in token_ids],
            activations_by_layer=activations,
            eos_logits=eos_logits.tolist(),
            reached_eos=True,
            prompt_token_count=16,
            text=" ".join(f"<synth-{token}>" for token in token_ids),
        )


class HuggingFaceTraceCollector:
    """Token-aligned causal-LM collector with optional activation intervention."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face collection requires the `model` extra: "
                "pip install -e '.[model]'"
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=self.tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if device == "auto":
            kwargs["device_map"] = "auto"
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.layers = discover_decoder_layers(self.model)
        self.input_device = next(self.model.parameters()).device

    def format_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def collect(
        self,
        prompt: str,
        layer_indices: list[int],
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
        disable_eos: bool = False,
        intervention: Callable[[int, int, Any], Any] | None = None,
    ) -> CollectedGeneration:
        torch = self.torch
        invalid = [index for index in layer_indices if index < 0 or index >= len(self.layers)]
        if invalid:
            raise ValueError(f"invalid layer indices {invalid}; model has {len(self.layers)} layers")
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.input_device)
        selected_layers = [self.layers[index] for index in layer_indices]
        captured: dict[int, list[Any]] = {index: [] for index in layer_indices}
        step = 0

        def hook_factory(local_index: int):
            layer_index = layer_indices[local_index]

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = hidden_from_output(output)
                last = hidden[:, -1, :]
                if intervention is not None:
                    changed = intervention(layer_index, step, last)
                    hidden = hidden.clone()
                    hidden[:, -1, :] = changed
                    last = changed
                    output = replace_hidden(output, hidden)
                captured[layer_index].append(last.detach().cpu().float().numpy()[0])
                return output

            return hook

        generator = torch.Generator(device=self.input_device)
        generator.manual_seed(seed)
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids or []}
        generated: list[int] = []
        eos_logits: list[float] = []
        past = None
        current_ids = input_ids
        current_mask = attention_mask
        reached_eos = False
        with registered_hooks(selected_layers, hook_factory), torch.inference_mode():
            for step in range(max_new_tokens):
                output = self.model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                logits = output.logits[:, -1, :]
                if eos_set:
                    eos_logits.append(float(logits[0, sorted(eos_set)[0]].item()))
                else:
                    eos_logits.append(float("nan"))
                if temperature > 0:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, 1, generator=generator)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                token = int(next_token.item())
                generated.append(token)
                past = output.past_key_values
                current_ids = next_token
                current_mask = torch.cat(
                    (
                        current_mask,
                        torch.ones((1, 1), device=current_mask.device, dtype=current_mask.dtype),
                    ),
                    dim=1,
                )
                if not disable_eos and token in eos_set:
                    reached_eos = True
                    break
        count = len(generated)
        arrays = {
            layer: np.asarray(values[:count], dtype=np.float32)
            for layer, values in captured.items()
        }
        for layer, array in arrays.items():
            if len(array) != count:
                raise RuntimeError(
                    f"hook/token misalignment at layer {layer}: {len(array)} activations "
                    f"for {count} generated tokens"
                )
        return CollectedGeneration(
            token_ids=generated,
            tokens=self.tokenizer.convert_ids_to_tokens(generated),
            activations_by_layer=arrays,
            eos_logits=eos_logits,
            reached_eos=reached_eos,
            prompt_token_count=int(input_ids.shape[1]),
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
        )

    def score_first_transition(
        self,
        prompt: str,
        candidate_transitions: list[str],
        layer_index: int,
        intervention: Callable[[int, int, Any], Any] | None = None,
    ) -> dict[str, Any]:
        """Score valid next-transition first tokens under a one-token budget.

        This is a conservative, architecture-independent causal outcome for
        controlled tasks. Multi-token transition scoring can be layered on top,
        but must preserve the same intervention window and problem-level unit.
        EOS probability is reported separately and is never counted as a valid
        transition.
        """

        if not candidate_transitions:
            raise ValueError("at least one candidate transition is required")
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"invalid layer {layer_index}")
        torch = self.torch
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(
            self.input_device
        )
        token_ids = []
        for transition in candidate_transitions:
            encoded = self.tokenizer(
                " " + transition.strip(),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if len(encoded) == 0:
                raise ValueError(f"candidate transition tokenized empty: {transition!r}")
            token_ids.append(int(encoded[0].item()))
        unique_ids = sorted(set(token_ids))

        def hook_factory(_local_index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = hidden_from_output(output)
                if intervention is not None:
                    changed = intervention(layer_index, 0, hidden[:, -1, :])
                    hidden = hidden.clone()
                    hidden[:, -1, :] = changed
                    output = replace_hidden(output, hidden)
                return output

            return hook

        with registered_hooks([self.layers[layer_index]], hook_factory), torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_list = [int(eos_ids)] if isinstance(eos_ids, int) else [int(x) for x in eos_ids or []]
        candidate_probabilities = {
            str(token_id): float(probabilities[token_id].item()) for token_id in unique_ids
        }
        return {
            "candidate_token_ids": unique_ids,
            "candidate_probabilities": candidate_probabilities,
            "valid_next_state_probability": float(
                probabilities[torch.tensor(unique_ids, device=probabilities.device)].sum().item()
            ),
            "eos_logit": (
                float(logits[eos_list[0]].item()) if eos_list else float("nan")
            ),
            "eos_probability": (
                float(probabilities[torch.tensor(eos_list, device=probabilities.device)].sum().item())
                if eos_list
                else float("nan")
            ),
            "fixed_continuation_budget": 1,
        }
