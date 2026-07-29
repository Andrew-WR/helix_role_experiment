from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import deterministic_id
from .hooks import (
    discover_decoder_layers,
    hidden_from_output,
    infer_adapter_target_layers,
    registered_hooks,
    replace_hidden,
)
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
        adapter_path: str | None = None,
        adapter_enabled: bool = True,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        max_memory: dict[str | int, str] | None = None,
        attn_implementation: str | None = None,
        low_cpu_mem_usage: bool = True,
        activation_dtype: str = "float16",
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face collection requires the `model` extra: "
                "pip install -e '.[model]'"
            ) from exc
        self.torch = torch
        self.adapter_path = adapter_path
        self.adapter_enabled = bool(adapter_path and adapter_enabled)
        self.adapter_target_layers: list[int] = []
        peft_config = None
        if adapter_path:
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "adapter_path requires PEFT: pip install peft"
                ) from exc
            peft_config = PeftConfig.from_pretrained(adapter_path)
            self.adapter_target_layers = infer_adapter_target_layers(
                peft_config, adapter_path
            )
            if model_id in ("auto_from_adapter", "", None) or str(model_id).startswith(
                "REQUIRED_"
            ):
                model_id = peft_config.base_model_name_or_path
            if not model_id:
                raise ValueError(
                    "adapter_config.json does not identify base_model_name_or_path; "
                    "set model.id explicitly"
                )
        if not model_id or str(model_id).startswith("REQUIRED_"):
            raise ValueError(
                "model.id is still a placeholder. Set a Hugging Face repo/local "
                "path, or use 'auto_from_adapter' with adapter_path."
            )
        if revision and str(revision).startswith("REQUIRED_"):
            raise ValueError(
                "model.revision is still a placeholder; use an immutable commit, "
                "'main', or null"
            )
        self.model_id = str(model_id)
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if device == "auto":
            kwargs["device_map"] = "auto"
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        if max_memory:
            kwargs["max_memory"] = {
                int(key) if str(key).isdigit() else key: value
                for key, value in max_memory.items()
            }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit bitsandbytes loading requires CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
            )
        base_model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter_enabled:
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False,
            )
        else:
            self.model = base_model
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.layers = discover_decoder_layers(self.model)
        self.input_device = self.model.get_input_embeddings().weight.device
        if activation_dtype not in ("float16", "float32"):
            raise ValueError("activation_dtype must be float16 or float32")
        self.activation_dtype = activation_dtype
        self.capture_torch_dtype = getattr(torch, activation_dtype)
        self.capture_numpy_dtype = getattr(np, activation_dtype)

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
        capture_activations: bool = True,
        capture_eos_logits: bool = True,
        stop_regex: str | None = None,
    ) -> CollectedGeneration:
        torch = self.torch
        invalid = [index for index in layer_indices if index < 0 or index >= len(self.layers)]
        if invalid:
            raise ValueError(f"invalid layer indices {invalid}; model has {len(self.layers)} layers")
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.input_device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.input_device)
        selected_layers = (
            [self.layers[index] for index in layer_indices]
            if capture_activations or intervention is not None
            else []
        )
        captured: dict[int, list[Any]] = {index: [] for index in layer_indices}
        stop_pattern = re.compile(stop_regex) if stop_regex else None
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
                if capture_activations:
                    captured[layer_index].append(
                        last.detach()
                        .to(dtype=self.capture_torch_dtype)
                        .cpu()
                        .numpy()[0]
                    )
                return output

            return hook

        generator = None
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
                if capture_eos_logits and eos_set:
                    eos_logits.append(float(logits[0, sorted(eos_set)[0]].item()))
                elif capture_eos_logits:
                    eos_logits.append(float("nan"))
                if temperature > 0:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    if generator is None:
                        generator = torch.Generator(device=probabilities.device)
                        generator.manual_seed(seed)
                    next_token = torch.multinomial(probabilities, 1, generator=generator)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                token = int(next_token.item())
                generated.append(token)
                past = output.past_key_values
                current_ids = next_token.to(self.input_device)
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
                if stop_pattern is not None and stop_pattern.search(
                    self.tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                    )
                ):
                    break
        count = len(generated)
        arrays = {}
        if capture_activations:
            arrays = {
                layer: np.asarray(
                    values[:count],
                    dtype=self.capture_numpy_dtype,
                )
                for layer, values in captured.items()
            }
            for layer, array in arrays.items():
                if len(array) != count:
                    raise RuntimeError(
                        f"hook/token misalignment at layer {layer}: "
                        f"{len(array)} activations for {count} generated tokens"
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

    def score_continuations(
        self,
        prompt: str,
        continuations: list[str],
        layer_index: int,
        intervention: Callable[[int, int, Any], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Teacher-force complete continuations under a persistent intervention.

        Unlike ``score_first_transition``, this avoids relying on one literal
        opening token. Each continuation is scored autoregressively, with the
        intervention applied at the selected layer while predicting every
        continuation token.
        """

        if not continuations:
            raise ValueError("at least one continuation is required")
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"invalid layer {layer_index}")
        torch = self.torch
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        prompt_ids = inputs["input_ids"].to(self.input_device)
        prompt_mask = inputs.get(
            "attention_mask",
            torch.ones_like(prompt_ids),
        ).to(self.input_device)
        results: list[dict[str, Any]] = []
        for continuation in continuations:
            target_ids = self.tokenizer(
                continuation,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if len(target_ids) == 0:
                raise ValueError(
                    f"continuation tokenized empty: {continuation!r}"
                )
            step = 0

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = hidden_from_output(output)
                if intervention is not None:
                    changed = intervention(
                        layer_index,
                        step,
                        hidden[:, -1, :],
                    )
                    hidden = hidden.clone()
                    hidden[:, -1, :] = changed
                    output = replace_hidden(output, hidden)
                return output

            current_ids = prompt_ids
            current_mask = prompt_mask
            past = None
            token_log_probabilities: list[float] = []
            with registered_hooks(
                [self.layers[layer_index]],
                lambda _index: hook,
            ), torch.inference_mode():
                for step, target_id in enumerate(target_ids.tolist()):
                    output = self.model(
                        input_ids=current_ids,
                        attention_mask=current_mask,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = output.logits[0, -1].float()
                    token_log_probabilities.append(
                        float(
                            torch.log_softmax(logits, dim=-1)[
                                int(target_id)
                            ].item()
                        )
                    )
                    past = output.past_key_values
                    current_ids = torch.as_tensor(
                        [[int(target_id)]],
                        device=self.input_device,
                    )
                    current_mask = torch.cat(
                        (
                            current_mask,
                            torch.ones(
                                (1, 1),
                                device=current_mask.device,
                                dtype=current_mask.dtype,
                            ),
                        ),
                        dim=1,
                    )
            results.append(
                {
                    "continuation": continuation,
                    "token_count": len(token_log_probabilities),
                    "total_log_probability": float(
                        sum(token_log_probabilities)
                    ),
                    "mean_log_probability": float(
                        np.mean(token_log_probabilities)
                    ),
                    "token_log_probabilities": token_log_probabilities,
                }
            )
        return results


def huggingface_collector_from_config(
    model_config: dict[str, Any],
    collection_config: dict[str, Any] | None = None,
) -> HuggingFaceTraceCollector:
    quantization = model_config.get("quantization") or {}
    collection = collection_config or {}
    return HuggingFaceTraceCollector(
        model_id=model_config.get("id"),
        revision=model_config.get("revision"),
        tokenizer_revision=model_config.get("tokenizer_revision"),
        device=model_config.get("device", "auto"),
        dtype=model_config.get("dtype", "auto"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        adapter_path=model_config.get("adapter_path"),
        adapter_enabled=bool(model_config.get("adapter_enabled", True)),
        load_in_4bit=bool(quantization.get("load_in_4bit", False)),
        bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            quantization.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=quantization.get(
            "bnb_4bit_compute_dtype", "float16"
        ),
        max_memory=model_config.get("max_memory"),
        attn_implementation=model_config.get("attn_implementation"),
        low_cpu_mem_usage=bool(model_config.get("low_cpu_mem_usage", True)),
        activation_dtype=collection.get("activation_dtype", "float16"),
    )
