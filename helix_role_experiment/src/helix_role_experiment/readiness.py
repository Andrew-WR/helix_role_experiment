from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .behavioral import split_sentence_spans


EVENT_LABELS = (
    "forward_progress",
    "productive_backtrack",
    "neutral_support",
    "redundant",
    "incorrect",
    "final_answer",
)


@dataclass(frozen=True)
class SentenceBoundary:
    sentence_id: str
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    activation_index: int
    is_reasoning: bool
    eos_logit: float | None = None
    token_entropy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _retokenized_char_owners(
    tokenizer: Any, text: str, token_ids: list[int]
) -> list[int]:
    """Fallback for tokenizers without DecodeStream.

    This path is exact only when encoding the decoded text reproduces the
    original segmentation. Modern Hugging Face tokenizers use the streaming
    decoder path below and do not require that invalid BPE assumption.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if [int(value) for value in ids] != [int(value) for value in token_ids]:
        raise ValueError(
            "this tokenizer lacks DecodeStream and decoded output uses a "
            "different valid BPE segmentation; install tokenizers>=0.21"
        )
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list) and len(offsets[0]) and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    owners = [-1] * len(text)
    for token_index, (left, right) in enumerate(offsets):
        for character in range(int(left), min(int(right), len(text))):
            owners[character] = token_index
    return owners


def _token_char_owners(
    tokenizer: Any, text: str, token_ids: list[int]
) -> list[int]:
    """Map decoded characters to the original generated-token indices.

    Decoding and then encoding is not an inverse operation for BPE: adjacent
    generated tokens may be merged on re-encoding. DecodeStream instead follows
    the original autoregressive ID stream and correctly buffers split UTF-8 and
    byte-fallback tokens.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return _retokenized_char_owners(tokenizer, text, token_ids)
    try:
        from tokenizers.decoders import DecodeStream
    except ImportError:
        return _retokenized_char_owners(tokenizer, text, token_ids)

    stream = DecodeStream(skip_special_tokens=False)
    chunks: list[str] = []
    owners: list[int] = []
    pending_start = 0
    for token_index, token_id in enumerate(token_ids):
        chunk = stream.step(backend, int(token_id))
        if chunk is None:
            continue
        chunks.append(chunk)
        owners.extend([pending_start] * len(chunk))
        pending_start = token_index + 1
    streamed = "".join(chunks)
    if streamed != text:
        # Wrapper decode settings can differ across Transformers releases. If
        # ordinary exact retokenization happens to work, it remains safe.
        try:
            return _retokenized_char_owners(tokenizer, text, token_ids)
        except ValueError as exc:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(streamed, text))
                    if left != right
                ),
                min(len(streamed), len(text)),
            )
            raise ValueError(
                "streaming decoder text differs from collected text at "
                f"character {mismatch}; streamed={len(streamed)} chars, "
                f"collected={len(text)} chars"
            ) from exc
    return owners


def sentence_boundaries(
    tokenizer: Any,
    text: str,
    token_ids: list[int],
    eos_logits: list[float] | None = None,
    token_entropies: list[float] | None = None,
) -> list[SentenceBoundary]:
    char_owners = _token_char_owners(tokenizer, text, token_ids)
    eos_logits = eos_logits or []
    token_entropies = token_entropies or []
    thinking_close = text.casefold().find("</think>")
    final_markers = [
        value for value in (
            text.casefold().find("final:"),
            text.casefold().find("final_code:"),
        ) if value >= 0
    ]
    reasoning_end = thinking_close if thinking_close >= 0 else (
        min(final_markers) if final_markers else len(text)
    )
    result: list[SentenceBoundary] = []
    for span in split_sentence_spans(text):
        overlapping = [
            owner for owner in char_owners[span.start : span.end] if owner >= 0
        ]
        if not overlapping:
            continue
        token_start = overlapping[0]
        token_end = overlapping[-1] + 1
        activation_index = token_start
        result.append(
            SentenceBoundary(
                sentence_id=f"S{len(result):04d}",
                text=span.text,
                char_start=span.start,
                char_end=span.end,
                token_start=token_start,
                token_end=token_end,
                activation_index=activation_index,
                is_reasoning=span.start < reasoning_end,
                eos_logit=(
                    float(eos_logits[activation_index])
                    if activation_index < len(eos_logits) else None
                ),
                token_entropy=(
                    float(token_entropies[activation_index])
                    if activation_index < len(token_entropies) else None
                ),
            )
        )
    return result


def annotation_json_schema(
    sentence_ids: list[str] | None = None,
) -> dict[str, Any]:
    tri = {"type": "string", "enum": ["yes", "no", "uncertain"]}
    sentence_id_schema: dict[str, Any] = {"type": "string"}
    if sentence_ids is not None:
        if not sentence_ids:
            raise ValueError("sentence_ids cannot be empty")
        sentence_id_schema["enum"] = list(sentence_ids)
    annotations_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sentence_id", "mathematically_correct", "novel",
                "advances_valid_path", "primary_label", "evidence",
                "state_change", "needs_review",
            ],
            "properties": {
                "sentence_id": sentence_id_schema,
                "mathematically_correct": tri,
                "novel": tri,
                "advances_valid_path": tri,
                "primary_label": {
                    "type": "string", "enum": list(EVENT_LABELS)
                },
                "evidence": {"type": "string"},
                "state_change": {"type": "string"},
                "needs_review": {"type": "boolean"},
            },
        },
    }
    if sentence_ids is not None:
        annotations_schema["minItems"] = len(sentence_ids)
        annotations_schema["maxItems"] = len(sentence_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["annotations"],
        "properties": {
            "annotations": {
                **annotations_schema,
            }
        },
    }


def validate_annotations(
    sentences: list[dict[str, Any]] | list[SentenceBoundary],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = [
        value.to_dict() if isinstance(value, SentenceBoundary) else value
        for value in sentences
    ]
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != len(expected):
        raise ValueError("annotation count must exactly match sentence count")
    validated = []
    for sentence, annotation in zip(expected, annotations, strict=True):
        if not isinstance(annotation, dict):
            raise ValueError("each annotation must be an object")
        if annotation.get("sentence_id") != sentence["sentence_id"]:
            raise ValueError("sentence IDs must be returned once, in exact order")
        if annotation.get("primary_label") not in EVENT_LABELS:
            raise ValueError("unknown primary label")
        for field in ("mathematically_correct", "novel", "advances_valid_path"):
            if annotation.get(field) not in {"yes", "no", "uncertain"}:
                raise ValueError(f"invalid {field}")
        evidence = annotation.get("evidence")
        if not isinstance(evidence, str) or (
            evidence and evidence not in sentence["text"]
        ):
            raise ValueError("evidence must be an exact substring of its sentence")
        if not isinstance(annotation.get("state_change"), str):
            raise ValueError("state_change must be text")
        if not isinstance(annotation.get("needs_review"), bool):
            raise ValueError("needs_review must be boolean")
        if annotation["primary_label"] == "forward_progress":
            if (
                annotation["mathematically_correct"] != "yes"
                or annotation["advances_valid_path"] != "yes"
                or annotation["novel"] != "yes"
            ):
                raise ValueError("forward progress must be correct, novel, and advance a valid path")
            if not evidence:
                raise ValueError("forward progress requires exact evidence")
        if annotation["primary_label"] == "final_answer" and sentence.get("is_reasoning", False):
            raise ValueError("a reasoning sentence cannot be labeled final_answer")
        if "uncertain" in (
            annotation["mathematically_correct"], annotation["novel"],
            annotation["advances_valid_path"],
        ) and not annotation["needs_review"]:
            raise ValueError("uncertain judgments must set needs_review")
        validated.append(dict(annotation))
    return validated


def assign_group_splits(
    task_rows: Iterable[dict[str, Any]], seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, str]:
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    rows = list(task_rows)
    rng = np.random.default_rng(int(seed))
    result: dict[str, str] = {}
    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        ids = sorted({str(row["task_id"]) for row in rows if str(row["domain"]) == domain})
        ids = [ids[index] for index in rng.permutation(len(ids))]
        n_train = int(round(len(ids) * fractions[0]))
        n_val = int(round(len(ids) * fractions[1]))
        for index, task_id in enumerate(ids):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            result[task_id] = split
    return result


def build_survival_rows(
    trace: dict[str, Any], annotations: list[dict[str, Any]], layer: int,
) -> list[dict[str, Any]]:
    sentences = trace["sentences"]
    if len(sentences) != len(annotations):
        raise ValueError("sentence and annotation lengths differ")
    reasoning = [index for index, value in enumerate(sentences) if value["is_reasoning"]]
    rows = []
    output_tokens = int(trace["output_token_count"])
    for position, index in enumerate(reasoning):
        current = sentences[index]
        future = next((
            candidate for candidate in reasoning[position:]
            if annotations[candidate]["primary_label"] == "forward_progress"
        ), None)
        event = int(future is not None)
        endpoint = sentences[future]["token_end"] if future is not None else output_tokens
        duration = max(1, int(endpoint) - int(current["token_start"]))
        previous_length = 0 if position == 0 else (
            int(sentences[reasoning[position - 1]]["token_end"])
            - int(sentences[reasoning[position - 1]]["token_start"])
        )
        rows.append({
            "trace_id": trace["trace_id"], "task_id": trace["task_id"],
            "domain": trace["domain"], "split": trace["split"],
            "layer": int(layer), "sentence_id": current["sentence_id"],
            "sentence_index": position, "activation_index": current["activation_index"],
            "duration": duration, "event": event,
            "token_start": current["token_start"],
            "previous_sentence_tokens": previous_length,
            "eos_logit": current.get("eos_logit"),
            "token_entropy": current.get("token_entropy"),
        })
    return rows


def concordance_index(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for i in range(len(duration)):
        for j in range(i + 1, len(duration)):
            if duration[i] == duration[j]:
                continue
            early, late = (i, j) if duration[i] < duration[j] else (j, i)
            if not event[early]:
                continue
            comparable += 1
            concordant += 1.0 if risk[early] > risk[late] else 0.5 if risk[early] == risk[late] else 0.0
    return float(concordant / comparable) if comparable else float("nan")


@dataclass
class ExponentialProbe:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    duration_scale: float
    layer: int = -1
    threshold: float = 0.0
    native_step_norm: float = 1.0

    def score(self, activation: np.ndarray) -> np.ndarray:
        values = np.asarray(activation, dtype=np.float64)
        return (values - self.mean) / self.scale @ self.weights + self.intercept

    @property
    def direction(self) -> np.ndarray:
        value = self.weights / self.scale
        norm = np.linalg.norm(value)
        return value / norm if norm else value

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, mean=self.mean, scale=self.scale, weights=self.weights,
            intercept=self.intercept, duration_scale=self.duration_scale,
            layer=self.layer, threshold=self.threshold,
            native_step_norm=self.native_step_norm,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExponentialProbe":
        value = np.load(path)
        return cls(
            mean=value["mean"], scale=value["scale"], weights=value["weights"],
            intercept=float(value["intercept"]), duration_scale=float(value["duration_scale"]),
            layer=int(value["layer"]), threshold=float(value["threshold"]),
            native_step_norm=float(value["native_step_norm"]),
        )


def fit_exponential_probe(
    activations: np.ndarray, duration: np.ndarray, event: np.ndarray,
    ridge: float = 1e-3, layer: int = -1,
) -> ExponentialProbe:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("probe fitting requires scipy") from exc
    x = np.asarray(activations, dtype=np.float64)
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    duration_scale = float(np.median(duration)) or 1.0
    t = duration / duration_scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, weights = parameters[0], parameters[1:]
        eta = np.clip(intercept + z @ weights, -20.0, 20.0)
        hazard_time = np.exp(eta) * t
        loss = np.mean(hazard_time - event * eta) + 0.5 * ridge * float(weights @ weights)
        residual = hazard_time - event
        gradient = np.r_[residual.mean(), z.T @ residual / len(z) + ridge * weights]
        return float(loss), gradient

    initial = np.zeros(z.shape[1] + 1)
    initial[0] = math.log(max(event.sum(), 0.5) / max(t.sum(), 1e-6))
    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(f"probe optimization failed: {fitted.message}")
    return ExponentialProbe(mean, scale, fitted.x[1:], float(fitted.x[0]), duration_scale, layer=layer)


class SentenceSteeringController:
    """Pulse a readiness direction only at safely parsed sentence starts."""

    def __init__(
        self, tokenizer: Any, probe: ExponentialProbe, alpha: float = 0.5,
        pulse_tokens: int = 4, mode: str = "gated", random_seed: int = 0,
    ) -> None:
        if mode not in {"gated", "always", "random"}:
            raise ValueError("mode must be gated, always, or random")
        self.tokenizer = tokenizer
        self.probe = probe
        self.alpha = float(alpha)
        self.pulse_tokens = int(pulse_tokens)
        self.mode = mode
        direction = probe.direction.astype(np.float32)
        if mode == "random":
            rng = np.random.default_rng(int(random_seed))
            direction = rng.normal(size=direction.shape).astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
        self.direction = direction
        self.generated_ids: list[int] = []
        self.generated_text = ""
        self._boundary_pending = True
        self._remaining = 0
        self._inside_reasoning = True
        self.trigger_count = 0
        self.steered_steps = 0
        self.scores: list[float] = []

    def _safe_completed_boundary(self, before: str, after: str) -> bool:
        if not after or after == before:
            return False
        # A period at the buffer edge may become a decimal after the next token.
        stripped = after.rstrip()
        if re.search(r"\d\.$", stripped):
            return False
        # A sentinel makes the parser distinguish a completed final sentence
        # from its always-emitted incomplete tail, while preserving its LaTeX,
        # abbreviation, decimal, and fenced-code protections.
        spans = split_sentence_spans(after + " X")
        return bool(
            len(spans) >= 2
            and spans[-2].end <= len(after)
            and not after[spans[-2].end :].strip()
        )

    def observe_token(self, token_id: int) -> None:
        before = self.generated_text
        self.generated_ids.append(int(token_id))
        self.generated_text = self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "</think>" in self.generated_text.casefold():
            self._inside_reasoning = False
        if self._inside_reasoning and self._safe_completed_boundary(before, self.generated_text):
            self._boundary_pending = True

    def __call__(self, layer_index: int, _step: int, hidden: Any) -> Any:
        if layer_index != self.probe.layer or not self._inside_reasoning:
            return hidden
        if self._boundary_pending:
            vector = hidden.detach().float().cpu().numpy()[0]
            score = float(self.probe.score(vector))
            self.scores.append(score)
            trigger = self.mode in {"always", "random"} or score >= self.probe.threshold
            if trigger:
                self._remaining = self.pulse_tokens
                self.trigger_count += 1
            self._boundary_pending = False
        if self._remaining <= 0:
            return hidden
        import torch
        direction = torch.as_tensor(self.direction, device=hidden.device, dtype=hidden.dtype)
        self._remaining -= 1
        self.steered_steps += 1
        return hidden + direction[None, :] * (self.alpha * self.probe.native_step_norm)


def parse_response_output(body: dict[str, Any]) -> dict[str, Any]:
    for output in body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return json.loads(content["text"])
    raise ValueError("response contains no output_text JSON")
