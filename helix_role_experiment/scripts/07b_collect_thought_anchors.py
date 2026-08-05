from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from helix_role_experiment.config import (
    atomic_json,
    ensure_output_dirs,
    load_config,
    read_jsonl,
    write_jsonl,
)
from helix_role_experiment.models import huggingface_collector_from_config
from helix_role_experiment.reasoning_benchmarks import ReadinessTask, readiness_prompt
from helix_role_experiment.thought_anchors import (
    forward_anchor_overlap,
    receiver_head_statistics,
    top_fraction_flags,
)


ACTIVE_ACCUMULATOR: "SentenceAttentionAccumulator | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute memory-bounded receiver-head Thought Anchor scores"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["collect", "finalize"])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def settings(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "top_k_heads": 16,
        "proximity_ignore": 1,
        "anchor_fraction": 0.10,
        "query_tokens_per_sentence": 4,
        "query_chunk_size": 32,
        "teacher_force_chunk_tokens": 256,
        "minimum_teacher_force_chunk_tokens": 64,
    }
    values.update(config.get("thought_anchors", {}))
    return values


def artifact_directory(paths: dict[str, Path]) -> Path:
    return paths["traces"] / "thought_anchor_attention"


def trace_files(paths: dict[str, Path]) -> list[Path]:
    return sorted((paths["traces"] / "readiness_baseline").glob("*.json"))


def valid_artifact(
    path: Path, trace: dict[str, Any], proximity_ignore: int
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            scores = data["vertical_scores"]
            kurtosis = data["head_kurtosis"]
            sentence_ids = data["sentence_ids"].astype(str).tolist()
            # Raw sentence attention is threshold-independent. Legacy artifacts
            # only retained statistics calculated with the original value of 4.
            reusable = "sentence_attention" in data.files
            stored_proximity = int(
                data["proximity_ignore"][0]
            ) if "proximity_ignore" in data.files else 4
            raw_shape_ok = True
            if reusable:
                raw = data["sentence_attention"]
                raw_shape_ok = (
                    raw.ndim == 4
                    and raw.shape[:2] == scores.shape[:2]
                    and raw.shape[-2:] == (len(sentence_ids), len(sentence_ids))
                )
            return (
                scores.ndim == 3
                and kurtosis.shape == scores.shape[:2]
                and len(sentence_ids) == scores.shape[-1]
                and raw_shape_ok
                and (reusable or stored_proximity == int(proximity_ignore))
                and sentence_ids == [
                    str(row["sentence_id"]) for row in trace["sentences"]
                    if row.get("is_reasoning", False)
                ]
            )
    except Exception:
        return False


def sampled_query_positions(
    token_sentence_ids: np.ndarray, sentence_count: int, per_sentence: int
) -> tuple[np.ndarray, np.ndarray]:
    if per_sentence <= 0:
        raise ValueError("query_tokens_per_sentence must be positive")
    positions, owners = [], []
    for sentence in range(sentence_count):
        local = np.flatnonzero(token_sentence_ids == sentence)
        if not len(local):
            raise ValueError(f"reasoning sentence {sentence} has no aligned tokens")
        count = min(per_sentence, len(local))
        chosen = local[
            np.unique(np.linspace(0, len(local) - 1, count).round().astype(int))
        ]
        positions.extend(chosen.tolist())
        owners.extend([sentence] * len(chosen))
    order = np.argsort(positions)
    return np.asarray(positions)[order], np.asarray(owners)[order]


class SentenceAttentionAccumulator:
    def __init__(
        self, token_sentence_ids: np.ndarray, sentence_count: int,
        per_sentence: int, chunk_size: int,
    ) -> None:
        self.token_sentence_ids = np.asarray(token_sentence_ids, dtype=np.int64)
        self.sentence_count = int(sentence_count)
        self.query_positions, self.query_owners = sampled_query_positions(
            self.token_sentence_ids, self.sentence_count, per_sentence
        )
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.key_starts = np.asarray([
            np.flatnonzero(self.token_sentence_ids == index)[0]
            for index in range(self.sentence_count)
        ], dtype=np.int64)
        self.key_ends = np.asarray([
            np.flatnonzero(self.token_sentence_ids == index)[-1] + 1
            for index in range(self.sentence_count)
        ], dtype=np.int64)
        self.sums: dict[int, np.ndarray] = {}
        self.counts: dict[int, np.ndarray] = {}
        self.query_offset = 0

    def query_window(self, query_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Return local query indices and owners for the current cached chunk."""
        stop = self.query_offset + int(query_length)
        selected = (
            (self.query_positions >= self.query_offset)
            & (self.query_positions < stop)
        )
        return (
            self.query_positions[selected] - self.query_offset,
            self.query_owners[selected],
        )

    def observe(
        self, module: Any, query: Any, key: Any, attention_mask: Any,
        scaling: float,
    ) -> None:
        torch = query.__class__.__module__.split(".")[0]
        if torch != "torch" or query.shape[0] != 1:
            raise RuntimeError("Thought Anchor collector requires torch batch size one")
        import torch as t

        layer = int(module.layer_idx)
        groups = int(getattr(module, "num_key_value_groups", 1))
        expanded_key = key.repeat_interleave(groups, dim=1) if groups > 1 else key
        heads = int(query.shape[1])
        local_sum = np.zeros(
            (heads, self.sentence_count, self.sentence_count), dtype=np.float64
        )
        local_count = np.zeros(self.sentence_count, dtype=np.int64)
        available_np = np.flatnonzero(self.key_ends <= int(key.shape[-2]))
        positions_all, owners_all = self.query_window(int(query.shape[-2]))
        if not len(positions_all) or not len(available_np):
            return
        starts = t.as_tensor(self.key_starts[available_np], device=query.device)
        ends = t.as_tensor(self.key_ends[available_np] - 1, device=query.device)
        lengths = t.as_tensor(
            self.key_ends[available_np] - self.key_starts[available_np],
            device=query.device, dtype=t.float32,
        )
        for start in range(0, len(positions_all), self.chunk_size):
            stop = min(start + self.chunk_size, len(positions_all))
            positions_np = positions_all[start:stop]
            owners_np = owners_all[start:stop]
            positions = t.as_tensor(positions_np, device=query.device)
            selected_query = query.index_select(2, positions)
            logits = t.matmul(
                selected_query, expanded_key.transpose(2, 3)
            ) * float(scaling)
            if attention_mask is not None:
                mask = attention_mask
                if mask.shape[-2] == query.shape[-2]:
                    mask = mask.index_select(-2, positions)
                if mask.dtype == t.bool:
                    logits = logits.masked_fill(~mask, float("-inf"))
                else:
                    logits = logits + mask
            else:
                keys = t.arange(key.shape[-2], device=query.device)
                global_positions = positions + self.query_offset
                causal = keys[None, :] <= global_positions[:, None]
                logits = logits.masked_fill(
                    ~causal[None, None, :, :], float("-inf")
                )
            probabilities = t.softmax(logits.float(), dim=-1)
            prefix = probabilities.cumsum(dim=-1)
            end_values = prefix.index_select(-1, ends)
            start_indices = starts - 1
            start_values = t.zeros_like(end_values)
            positive = start_indices >= 0
            if bool(positive.any()):
                start_values[..., positive] = prefix.index_select(
                    -1, start_indices[positive]
                )
            key_means = ((end_values - start_values) / lengths)[0]
            for owner in np.unique(owners_np):
                mask_np = owners_np == owner
                owner_mask = t.as_tensor(mask_np, device=query.device)
                local_sum[:, int(owner), available_np] += (
                    key_means[:, owner_mask, :].sum(dim=1).cpu().numpy()
                )
                local_count[int(owner)] += int(mask_np.sum())
            del logits, probabilities, prefix, key_means
        if layer not in self.sums:
            self.sums[layer] = local_sum
            self.counts[layer] = local_count
        else:
            self.sums[layer] += local_sum
            self.counts[layer] += local_count

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.sums:
            raise RuntimeError(
                "model exposed no full-attention layers to the custom backend"
            )
        layers = np.asarray(sorted(self.sums), dtype=np.int64)
        matrices = []
        for layer in layers:
            count = self.counts[int(layer)]
            matrix = np.full_like(self.sums[int(layer)], np.nan)
            valid = count > 0
            matrix[:, valid, :] = (
                self.sums[int(layer)][:, valid, :] / count[valid][None, :, None]
            )
            matrices.append(matrix)
        return layers, np.asarray(matrices, dtype=np.float32)


def register_attention_backend() -> None:
    import torch
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask

    def thought_anchor_attention(
        module: Any, query: Any, key: Any, value: Any,
        attention_mask: Any, scaling: float, dropout: float = 0.0, **kwargs: Any,
    ) -> tuple[Any, None]:
        output, _ = sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
        if ACTIVE_ACCUMULATOR is not None:
            with torch.no_grad():
                ACTIVE_ACCUMULATOR.observe(
                    module, query, key, attention_mask, scaling
                )
        return output, None

    AttentionInterface.register("thought_anchor_sdpa", thought_anchor_attention)
    AttentionMaskInterface.register("thought_anchor_sdpa", sdpa_mask)


def tokenized_trace(backend: Any, trace: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    task = ReadinessTask.from_dict(trace)
    prefix = backend.format_prompt(readiness_prompt(task))
    full_text = prefix + str(trace["text"])
    encoded = backend.tokenizer(
        full_text, return_tensors="pt", return_offsets_mapping=True
    )
    offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
    reasoning = [
        row for row in trace["sentences"] if row.get("is_reasoning", False)
    ]
    owners = np.full(len(offsets), -1, dtype=np.int64)
    prefix_length = len(prefix)
    for index, sentence in enumerate(reasoning):
        left = prefix_length + int(sentence["char_start"])
        right = prefix_length + int(sentence["char_end"])
        overlap = (offsets[:, 1] > left) & (offsets[:, 0] < right)
        owners[overlap] = index
    missing = [
        reasoning[index]["sentence_id"] for index in range(len(reasoning))
        if not np.any(owners == index)
    ]
    if missing:
        raise ValueError(f"could not align reasoning sentences: {missing[:5]}")
    return encoded, owners, [str(row["sentence_id"]) for row in reasoning]


def base_text_model(model: Any) -> Any:
    candidate = getattr(model, "model", None)
    if candidate is not None and hasattr(candidate, "layers"):
        return candidate
    language = getattr(candidate, "language_model", None)
    if language is not None:
        return language
    raise RuntimeError("could not locate the Qwen text backbone without its LM head")


def cached_teacher_force(
    model: Any, inputs: dict[str, Any], accumulator: SentenceAttentionAccumulator,
    chunk_tokens: int,
) -> None:
    """Run a long trace incrementally so SDPA never materializes an L x L pass."""
    if chunk_tokens <= 0:
        raise ValueError("teacher_force_chunk_tokens must be positive")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    token_count = int(input_ids.shape[-1])
    past_key_values = None
    backbone = base_text_model(model)
    for offset in range(0, token_count, chunk_tokens):
        stop = min(offset + chunk_tokens, token_count)
        accumulator.query_offset = offset
        chunk_inputs: dict[str, Any] = {
            "input_ids": input_ids[:, offset:stop],
        }
        if attention_mask is not None:
            # Cached attention keys contain the complete prefix, so the mask must
            # cover that same prefix even though only new input IDs are supplied.
            chunk_inputs["attention_mask"] = attention_mask[:, :stop]
        output = backbone(
            **chunk_inputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        del output
    del past_key_values


def collect_matrices_with_oom_retry(
    backend: Any, inputs: dict[str, Any], owners: np.ndarray,
    sentence_count: int, values: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect once, halving the teacher-forcing chunk on a CUDA OOM."""
    global ACTIVE_ACCUMULATOR
    import torch

    chunk_tokens = int(values["teacher_force_chunk_tokens"])
    minimum = int(values["minimum_teacher_force_chunk_tokens"])
    if minimum <= 0 or chunk_tokens < minimum:
        raise ValueError(
            "teacher_force_chunk_tokens must be >= "
            "minimum_teacher_force_chunk_tokens > 0"
        )
    while True:
        ACTIVE_ACCUMULATOR = SentenceAttentionAccumulator(
            owners, sentence_count,
            int(values["query_tokens_per_sentence"]),
            int(values["query_chunk_size"]),
        )
        out_of_memory = False
        try:
            with torch.inference_mode():
                cached_teacher_force(
                    backend.model, inputs, ACTIVE_ACCUMULATOR, chunk_tokens
                )
            layers, matrices = ACTIVE_ACCUMULATOR.matrices()
            return layers, matrices, chunk_tokens
        except torch.OutOfMemoryError:
            # Leave the except block before clearing CUDA: the active traceback
            # can retain the failed chunk's cache tensors until this block exits.
            out_of_memory = True
        if out_of_memory:
            ACTIVE_ACCUMULATOR = None
            gc.collect()
            torch.cuda.empty_cache()
            next_size = chunk_tokens // 2
            if next_size < minimum:
                raise RuntimeError(
                    f"CUDA OOM persisted at the minimum {minimum}-token chunk"
                )
            print(
                f"CUDA OOM with {chunk_tokens}-token teacher-forcing chunks; "
                f"retrying this trace with {next_size}", flush=True,
            )
            chunk_tokens = next_size


def worker(
    args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]
) -> None:
    global ACTIVE_ACCUMULATOR
    import torch

    register_attention_backend()
    model_config = copy.deepcopy(config["model"])
    model_config["attn_implementation"] = "thought_anchor_sdpa"
    backend = huggingface_collector_from_config(model_config, config["collection"])
    values = settings(config)
    sources = trace_files(paths)
    if args.limit is not None:
        sources = sources[:args.limit]
    sources = [
        source for index, source in enumerate(sources)
        if index % args.num_shards == args.shard_index
    ]
    destination = artifact_directory(paths)
    destination.mkdir(parents=True, exist_ok=True)
    pending = []
    for source in sources:
        trace = json.loads(source.read_text(encoding="utf-8"))
        target = destination / f"{trace['trace_id']}.npz"
        if not valid_artifact(
            target, trace, int(values["proximity_ignore"])
        ):
            pending.append((trace, target))
    complete = len(sources) - len(pending)
    print(
        f"[anchor shard {args.shard_index}] {complete}/{len(sources)} already "
        "complete", flush=True,
    )
    for trace, target in pending:
        encoded, owners, sentence_ids = tokenized_trace(backend, trace)
        inputs = {
            key: tensor.to(backend.input_device)
            for key, tensor in encoded.items()
        }
        layers, matrices, used_chunk_tokens = collect_matrices_with_oom_retry(
            backend, inputs, owners, len(sentence_ids), values
        )
        vertical, kurtosis = receiver_head_statistics(
            matrices, int(values["proximity_ignore"])
        )
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, layers=layers,
                sentence_ids=np.asarray(sentence_ids),
                vertical_scores=vertical, head_kurtosis=kurtosis,
                # Preserve the actual float32 sentence aggregates so later
                # proximity sweeps are exact and never require Qwen again.
                sentence_attention=matrices.astype(np.float32),
                proximity_ignore=np.asarray(
                    [int(values["proximity_ignore"])], dtype=np.int64
                ),
                query_tokens_per_sentence=np.asarray(
                    [int(values["query_tokens_per_sentence"])]
                ),
                teacher_force_chunk_tokens=np.asarray([used_chunk_tokens]),
            )
        temporary.replace(target)
        ACTIVE_ACCUMULATOR = None
        complete += 1
        print(
            f"[anchor shard {args.shard_index}] {complete}/{len(sources)} "
            f"saved {trace['trace_id']}: {len(sentence_ids)} sentences, "
            f"layers={layers.tolist()}, tokens={len(owners)}, "
            f"teacher_force_chunk={used_chunk_tokens}", flush=True,
        )
        torch.cuda.empty_cache()


def artifact_statistics(
    path: Path, proximity_ignore: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        layers = data["layers"].astype(np.int64)
        if "sentence_attention" in data.files:
            vertical, kurtosis = receiver_head_statistics(
                data["sentence_attention"].astype(np.float32), proximity_ignore
            )
        else:
            stored_proximity = int(
                data["proximity_ignore"][0]
            ) if "proximity_ignore" in data.files else 4
            if stored_proximity != proximity_ignore:
                raise RuntimeError(
                    f"{path} only contains scores for proximity_ignore="
                    f"{stored_proximity}; recollect it once to cache raw sentence "
                    "attention"
                )
            vertical = data["vertical_scores"].astype(np.float32)
            kurtosis = data["head_kurtosis"].astype(np.float32)
    return layers, vertical, kurtosis


def select_receiver_heads(
    artifacts: list[tuple[dict[str, Any], Path]], top_k: int,
    proximity_ignore: int,
) -> list[dict[str, Any]]:
    train = [(trace, path) for trace, path in artifacts if trace["split"] == "train"]
    if not train:
        raise RuntimeError("receiver-head selection requires training traces")
    values: dict[tuple[int, int], list[float]] = {}
    for _, path in train:
        layers, _, kurtosis = artifact_statistics(path, proximity_ignore)
        for local_layer, layer in enumerate(layers):
            for head, score in enumerate(kurtosis[local_layer]):
                if np.isfinite(score):
                    values.setdefault((int(layer), head), []).append(float(score))
    ranked = sorted(
        (
            {"layer": layer, "head": head, "median_train_kurtosis": float(np.median(scores)),
             "train_trace_count": len(scores)}
            for (layer, head), scores in values.items()
        ),
        key=lambda row: row["median_train_kurtosis"], reverse=True,
    )
    return ranked[:min(top_k, len(ranked))]


def finalize(config: dict[str, Any], paths: dict[str, Path]) -> None:
    values = settings(config)
    artifacts = []
    missing = []
    for source in trace_files(paths):
        trace = json.loads(source.read_text(encoding="utf-8"))
        path = artifact_directory(paths) / f"{trace['trace_id']}.npz"
        if valid_artifact(path, trace, int(values["proximity_ignore"])):
            artifacts.append((trace, path))
        else:
            missing.append(str(trace["trace_id"]))
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} attention artifacts; rerun collect. "
            f"First missing: {missing[0]}"
        )
    heads = select_receiver_heads(
        artifacts, int(values["top_k_heads"]),
        int(values["proximity_ignore"]),
    )
    if not heads:
        raise RuntimeError("no finite receiver heads were found")
    records = []
    fraction = float(values["anchor_fraction"])
    for trace, path in artifacts:
        layers_array, vertical_array, _ = artifact_statistics(
            path, int(values["proximity_ignore"])
        )
        layers = layers_array.astype(int).tolist()
        vertical = vertical_array.astype(np.float64)
        with np.load(path) as data:
            sentence_ids = data["sentence_ids"].astype(str).tolist()
        selected = []
        for head in heads:
            if head["layer"] in layers and head["head"] < vertical.shape[1]:
                selected.append(vertical[layers.index(head["layer"]), head["head"]])
        scores = np.nanmean(np.asarray(selected), axis=0)
        flags, percentiles = top_fraction_flags(scores, fraction)
        records.append({
            "trace_id": trace["trace_id"], "task_id": trace["task_id"],
            "domain": trace["domain"], "split": trace["split"],
            "anchor_fraction": fraction,
            "receiver_head_count": len(selected),
            "sentences": [
                {
                    "sentence_id": sentence_id,
                    "score": float(scores[index]) if np.isfinite(scores[index]) else None,
                    "within_trace_percentile": (
                        float(percentiles[index])
                        if np.isfinite(percentiles[index]) else None
                    ),
                    "thought_anchor": bool(flags[index]),
                }
                for index, sentence_id in enumerate(sentence_ids)
            ],
        })
    write_jsonl(paths["tables"] / "thought_anchor_sentences.jsonl", records)
    atomic_json(paths["tables"] / "thought_anchor_receiver_heads.json", {
        "selection_split": "train", "top_k": len(heads),
        "proximity_ignore": int(values["proximity_ignore"]),
        "query_tokens_per_sentence": int(values["query_tokens_per_sentence"]),
        "approximation_note": (
            "All key tokens are used. Query-token averaging is approximated by "
            "evenly sampled tokens per sentence to bound T4 memory and runtime."
        ),
        "heads": heads,
    })
    annotation_path = paths["tables"] / "sentence_annotations.jsonl"
    annotations = read_jsonl(annotation_path) if annotation_path.exists() else []
    anchors_by_trace = {
        str(record["trace_id"]): {
            str(row["sentence_id"]): row for row in record["sentences"]
        }
        for record in records
    }
    merged = []
    for annotation_record in annotations:
        enriched = dict(annotation_record)
        local = anchors_by_trace.get(str(annotation_record["trace_id"]), {})
        enriched_rows = []
        for row in annotation_record["annotations"]:
            enriched_row = dict(row)
            anchor = local.get(str(row["sentence_id"]))
            enriched_row.update({
                "thought_anchor": bool(anchor["thought_anchor"]) if anchor else False,
                "thought_anchor_score": anchor["score"] if anchor else None,
                "thought_anchor_percentile": (
                    anchor["within_trace_percentile"] if anchor else None
                ),
            })
            enriched_rows.append(enriched_row)
        enriched["annotations"] = enriched_rows
        merged.append(enriched)
    if merged:
        write_jsonl(
            paths["tables"] / "sentence_annotations_with_thought_anchors.jsonl",
            merged,
        )
    overlap = forward_anchor_overlap(records, annotations)
    overlap["anchor_fraction"] = fraction
    overlap["labeled_trajectory_count"] = len({
        row["trace_id"] for row in annotations
        if row.get("source") != "modernbert_sequential_event_tagger"
    })
    atomic_json(paths["tables"] / "thought_anchor_forward_overlap.json", overlap)
    print(json.dumps(overlap, indent=2), flush=True)
    print(
        f"Saved {len(records)} trace-level Thought Anchor records and "
        f"{len(heads)} frozen receiver heads.", flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.command == "finalize":
        finalize(config, paths)
        return
    if args.worker:
        worker(args, config, paths)
        return
    processes = []
    for shard in range(2):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", args.config, "collect", "--worker",
            "--shard-index", str(shard), "--num-shards", "2",
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        processes.append(subprocess.Popen(command, env=environment))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"Thought Anchor workers failed: {codes}")


if __name__ == "__main__":
    main()
