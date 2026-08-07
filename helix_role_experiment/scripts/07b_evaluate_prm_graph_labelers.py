from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from helix_role_experiment.prm_graph_labeling import (
    POSITIVE_LABELS,
    LabeledTraceFeatures,
    direct_prm_features,
    evaluate_groups,
    fit_oof,
    predict_logistic,
    prm_graph_node_vectors,
    serializable_model,
    temporal_graph_features,
    validation_gate,
)


EXCLUDED_SOURCES = {
    "attention_burst_pseudo_labeler",
    "attention_reorientation_pseudo_labeler",
    "modernbert_sequential_event_tagger",
}
METHOD_FEATURE_NAMES = {
    "prm": [
        "reward", "reward_delta", "reward_second_delta", "vs_prior_mean",
        "future_mean_delta", "above_prefix_min", "below_prefix_max",
    ],
    "embedding_graph": [
        "past_max", "past_knn_mean", "future_max", "future_knn_mean",
        "future_minus_past", "novelty", "local_change",
        "future_indegree", "future_inweight", "branch_score",
    ],
    "prm_graph": [
        "past_max", "past_knn_mean", "future_max", "future_knn_mean",
        "future_minus_past", "novelty", "local_change",
        "future_indegree", "future_inweight", "branch_score",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trajectory-aware PRM and semantic/PRM temporal graphs "
            "against the strong sentence annotations without changing labels"
        )
    )
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prm = subparsers.add_parser("collect-prm")
    prm.add_argument("--limit", type=int)
    embeddings = subparsers.add_parser("collect-embeddings")
    embeddings.add_argument("--limit", type=int)
    subparsers.add_parser("evaluate")
    return parser.parse_args()


def settings(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "prm_model_id": "Gen-Verse/ReasonFlux-PRM-7B",
        "prm_revision": "835c4809502841a4601b024f770f8d1f62efaabf",
        "embedding_model_id": "BAAI/bge-m3",
        "dtype": "float16",
        "attn_implementation": "sdpa",
        "max_length": 32768,
        "max_memory": {"0": "14GiB", "1": "14GiB", "cpu": "20GiB"},
        "embedding_device": "cuda:0",
        "embedding_batch_size": 32,
        "embedding_max_length": 512,
        "graph_k": 5,
        "ridge": 1.0,
        "minimum_threshold": 0.05,
        "target_precision": 0.5,
        "minimum_validation_precision": 0.25,
        "minimum_validation_recall": 0.25,
        "minimum_validation_lift": 2.0,
        "minimum_tolerant_f1": 0.25,
    }
    values.update(config.get("prm_graph_labeling", {}))
    return values


def strong_records(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    preferred = paths["tables"] / "sentence_annotations_strong_seed.jsonl"
    source = preferred if preferred.exists() else paths["tables"] / "sentence_annotations.jsonl"
    if not source.exists():
        raise RuntimeError("restore the 15 strong labels before this benchmark")
    records = {
        str(row["trace_id"]): row
        for row in read_jsonl(source)
        if str(row.get("source", "unknown")) not in EXCLUDED_SOURCES
    }
    if not records:
        raise RuntimeError(f"{source} contains no strong trajectory annotations")
    print(f"Using {len(records)} strong labeled trajectories from {source}.", flush=True)
    return records


def labeled_traces(
    paths: dict[str, Path], records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    traces = []
    missing = []
    root = paths["traces"] / "readiness_baseline"
    for trace_id in records:
        source = root / f"{trace_id}.json"
        if not source.exists():
            missing.append(str(source))
            continue
        traces.append(json.loads(source.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError("missing baseline traces:\n" + "\n".join(missing[:5]))
    order = {"train": 0, "val": 1, "test": 2}
    return sorted(traces, key=lambda row: (order.get(str(row["split"]), 9), str(row["trace_id"])))


def reasoning_sentences(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in trace["sentences"] if row.get("is_reasoning", False)]


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _artifact_matches(
    path: Path, sentence_ids: list[str], model_id: str, value_key: str,
    revision: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            base_matches = (
                value_key in data.files
                and data["sentence_ids"].astype(str).tolist() == sentence_ids
                and str(data["model_id"].item()) == model_id
            )
            revision_matches = (
                revision is None
                or (
                    "revision" in data.files
                    and str(data["revision"].item()) == revision
                )
            )
            return base_matches and revision_matches
    except (OSError, ValueError, KeyError):
        return False


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported PRM dtype {name!r}")
    return mapping[name]


def ensure_prm_config_compatibility(config: Any, tokenizer: Any) -> Any:
    """Fill fields omitted by ReasonFlux's legacy remote configuration."""
    if not hasattr(config, "pad_token_id"):
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("ReasonFlux tokenizer defines no pad or EOS token")
        config.pad_token_id = int(pad_token_id)
    return config


def load_prm_model(values: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the project model dependencies first") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PRM collection requires a CUDA GPU")
    model_id = str(values["prm_model_id"])
    revision = str(values["prm_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )
    config = ensure_prm_config_compatibility(config, tokenizer)
    config._attn_implementation = str(values["attn_implementation"])
    config.use_cache = False
    configured_memory = dict(values.get("max_memory", {}))
    max_memory: dict[int | str, str] = {}
    for key, value in configured_memory.items():
        if str(key).isdigit():
            gpu = int(key)
            if gpu < torch.cuda.device_count():
                max_memory[gpu] = str(value)
        elif str(key) == "cpu":
            max_memory["cpu"] = str(value)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": _torch_dtype(torch, str(values["dtype"])),
        "trust_remote_code": True,
        "revision": revision,
        "config": config,
        "low_cpu_mem_usage": True,
        "max_memory": max_memory,
    }
    if values.get("attn_implementation"):
        kwargs["attn_implementation"] = str(values["attn_implementation"])
    try:
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()
    return tokenizer, model


def score_trace_with_prm(
    tokenizer: Any, model: Any, trace: dict[str, Any], max_length: int,
) -> tuple[np.ndarray, int]:
    import torch

    sentences = reasoning_sentences(trace)
    separator = "<extra_0>"
    messages = [
        {"role": "user", "content": str(trace["prompt"])},
        {
            "role": "assistant",
            "content": separator.join(str(row["text"]) for row in sentences) + separator,
        },
    ]
    conversation = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation, return_tensors="pt")
    if input_ids.shape[1] > max_length:
        raise RuntimeError(
            f"{trace['trace_id']} is {input_ids.shape[1]} PRM tokens, over the "
            f"configured {max_length}; do not truncate because it would misalign labels"
        )
    separator_ids = tokenizer.encode(separator, add_special_tokens=False)
    if len(separator_ids) != 1:
        raise RuntimeError(f"PRM separator tokenized into {len(separator_ids)} tokens")
    positions = torch.nonzero(input_ids[0] == separator_ids[0], as_tuple=False).flatten()
    if len(positions) < len(sentences):
        raise RuntimeError(
            f"found {len(positions)} reward positions for {len(sentences)} sentences"
        )
    # A task containing the literal separator is harmless: completion markers
    # are always the final N occurrences.
    positions = positions[-len(sentences):]
    first_device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(first_device), use_cache=False)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
    selected = logits[0, positions.to(logits.device)]
    if selected.ndim != 2 or selected.shape[-1] < 2:
        raise RuntimeError(f"unexpected PRM logits shape {tuple(logits.shape)}")
    rewards = torch.softmax(selected.float(), dim=-1)[:, 1]
    return rewards.detach().cpu().numpy().astype(np.float32), int(input_ids.shape[1])


def collect_prm(
    paths: dict[str, Path], traces: list[dict[str, Any]],
    values: dict[str, Any], limit: int | None,
) -> None:
    model_id = str(values["prm_model_id"])
    revision = str(values["prm_revision"])
    root = paths["traces"] / "prm_sentence_scores"
    pending = []
    for trace in traces:
        ids = [str(row["sentence_id"]) for row in reasoning_sentences(trace)]
        target = root / f"{trace['trace_id']}.npz"
        if not _artifact_matches(
            target, ids, model_id, "prm_scores", revision=revision
        ):
            pending.append(trace)
    if limit is not None:
        pending = pending[:max(limit, 0)]
    if not pending:
        print("All requested PRM score artifacts already exist.", flush=True)
        return
    tokenizer, model = load_prm_model(values)
    for index, trace in enumerate(pending, start=1):
        sentences = reasoning_sentences(trace)
        scores, token_count = score_trace_with_prm(
            tokenizer, model, trace, int(values["max_length"])
        )
        target = root / f"{trace['trace_id']}.npz"
        _atomic_npz(
            target,
            sentence_ids=np.asarray([str(row["sentence_id"]) for row in sentences]),
            prm_scores=scores,
            model_id=np.asarray(model_id),
            revision=np.asarray(revision),
            input_token_count=np.asarray(token_count, dtype=np.int64),
        )
        print(
            f"[PRM {index}/{len(pending)}] {trace['trace_id']}: "
            f"{len(scores)} sentences, {token_count} tokens",
            flush=True,
        )
    del model
    gc.collect()


def collect_embeddings(
    paths: dict[str, Path], traces: list[dict[str, Any]],
    values: dict[str, Any], limit: int | None,
) -> None:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the project model dependencies first") from exc
    model_id = str(values["embedding_model_id"])
    root = paths["traces"] / "prm_graph_embeddings"
    pending = []
    for trace in traces:
        ids = [str(row["sentence_id"]) for row in reasoning_sentences(trace)]
        target = root / f"{trace['trace_id']}.npz"
        if not _artifact_matches(target, ids, model_id, "embeddings"):
            pending.append(trace)
    if limit is not None:
        pending = pending[:max(limit, 0)]
    if not pending:
        print("All requested sentence embedding artifacts already exist.", flush=True)
        return
    device = torch.device(str(values["embedding_device"]))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: dict[str, Any] = {}
    if device.type == "cuda":
        model_kwargs["dtype"] = torch.float16
    try:
        model = AutoModel.from_pretrained(model_id, **model_kwargs).to(device).eval()
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModel.from_pretrained(model_id, **model_kwargs).to(device).eval()
    for index, trace in enumerate(pending, start=1):
        sentences = reasoning_sentences(trace)
        texts = [str(row["text"]) for row in sentences]
        batches = []
        batch_size = int(values["embedding_batch_size"])
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=int(values["embedding_max_length"]),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                # BGE-M3 defines its dense vector as the first token in the
                # final hidden state.  Going through Transformers directly
                # avoids optional audio/video TorchCodec imports.
                dense = model(**encoded).last_hidden_state[:, 0]
                dense = functional.normalize(dense.float(), p=2, dim=-1)
            batches.append(dense.cpu().numpy())
        embeddings = np.concatenate(batches, axis=0)
        target = root / f"{trace['trace_id']}.npz"
        _atomic_npz(
            target,
            sentence_ids=np.asarray([str(row["sentence_id"]) for row in sentences]),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model_id=np.asarray(model_id),
        )
        print(
            f"[embeddings {index}/{len(pending)}] {trace['trace_id']}: "
            f"{len(sentences)} sentences",
            flush=True,
        )


def labels_for_trace(
    trace: dict[str, Any], record: dict[str, Any], sentence_ids: list[str],
) -> np.ndarray:
    annotations = {
        str(row["sentence_id"]): str(row["primary_label"])
        for row in record["annotations"]
    }
    missing = [sentence_id for sentence_id in sentence_ids if sentence_id not in annotations]
    if missing:
        raise RuntimeError(f"incomplete strong labels for {trace['trace_id']}")
    return np.asarray([
        annotations[sentence_id] in POSITIVE_LABELS for sentence_id in sentence_ids
    ], dtype=np.int64)


def load_artifacts(
    paths: dict[str, Path], trace: dict[str, Any], values: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    trace_id = str(trace["trace_id"])
    expected_ids = [str(row["sentence_id"]) for row in reasoning_sentences(trace)]
    prm_path = paths["traces"] / "prm_sentence_scores" / f"{trace_id}.npz"
    embedding_path = paths["traces"] / "prm_graph_embeddings" / f"{trace_id}.npz"
    if not prm_path.exists() or not embedding_path.exists():
        raise RuntimeError(
            f"missing features for {trace_id}; run collect-prm and collect-embeddings first"
        )
    with np.load(prm_path, allow_pickle=False) as data:
        prm_ids = data["sentence_ids"].astype(str).tolist()
        prm_scores = data["prm_scores"].astype(np.float64)
        prm_model = str(data["model_id"].item())
        prm_revision = str(data["revision"].item()) if "revision" in data.files else None
    with np.load(embedding_path, allow_pickle=False) as data:
        embedding_ids = data["sentence_ids"].astype(str).tolist()
        embeddings = data["embeddings"].astype(np.float64)
        embedding_model = str(data["model_id"].item())
    if prm_ids != expected_ids or embedding_ids != expected_ids:
        raise RuntimeError(f"feature/sentence alignment mismatch for {trace_id}")
    if prm_model != str(values["prm_model_id"]):
        raise RuntimeError(f"stale PRM artifact for {trace_id}: {prm_model}")
    if prm_revision != str(values["prm_revision"]):
        raise RuntimeError(f"stale PRM revision for {trace_id}: {prm_revision}")
    if embedding_model != str(values["embedding_model_id"]):
        raise RuntimeError(f"stale embedding artifact for {trace_id}: {embedding_model}")
    return expected_ids, prm_scores, embeddings


def annotation_digest(records: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate(
    paths: dict[str, Path], traces: list[dict[str, Any]],
    records: dict[str, dict[str, Any]], values: dict[str, Any],
) -> None:
    k = int(values["graph_k"])
    method_groups: dict[str, list[LabeledTraceFeatures]] = {
        name: [] for name in METHOD_FEATURE_NAMES
    }
    base_rows: dict[str, dict[str, Any]] = {}
    for index, trace in enumerate(traces, start=1):
        trace_id = str(trace["trace_id"])
        sentence_ids, prm_scores, embeddings = load_artifacts(paths, trace, values)
        labels = labels_for_trace(trace, records[trace_id], sentence_ids)
        features = {
            "prm": direct_prm_features(prm_scores),
            "embedding_graph": temporal_graph_features(embeddings, k=k),
            "prm_graph": temporal_graph_features(prm_graph_node_vectors(prm_scores), k=k),
        }
        for method, matrix in features.items():
            method_groups[method].append(LabeledTraceFeatures(
                trace_id=trace_id,
                domain=str(trace["domain"]),
                split=str(trace["split"]),
                sentence_ids=tuple(sentence_ids),
                labels=labels,
                features=matrix,
            ))
        base_rows[trace_id] = {
            "trace_id": trace_id,
            "task_id": str(trace["task_id"]),
            "domain": str(trace["domain"]),
            "split": str(trace["split"]),
            "sentence_ids": sentence_ids,
            "labels": labels.tolist(),
            "prm_scores": prm_scores.tolist(),
        }
        print(f"[load {index}/{len(traces)}] {trace_id}", flush=True)

    report: dict[str, Any] = {
        "purpose": "evaluation_only_no_annotation_mutation",
        "annotation_sha256": annotation_digest(records),
        "strong_trajectory_count": len(traces),
        "split_counts": {
            split: sum(str(trace["split"]) == split for trace in traces)
            for split in ("train", "val", "test")
        },
        "prm_model": str(values["prm_model_id"]),
        "prm_revision": str(values["prm_revision"]),
        "embedding_model": str(values["embedding_model_id"]),
        "benchmark_settings": {
            key: values[key] for key in (
                "graph_k", "ridge", "minimum_threshold", "target_precision",
                "embedding_max_length",
                "minimum_validation_precision", "minimum_validation_recall",
                "minimum_validation_lift", "minimum_tolerant_f1",
            )
        },
        "graph_definition": (
            "directed temporal kNN; past similarity measures redundancy/continuity, "
            "later-node retrieval measures future uptake"
        ),
        "selection_rule": (
            "fit/threshold on train trajectory-LOO predictions; validation gate and "
            "validation exact F1 select the candidate; test never selects"
        ),
        "methods": {},
    }
    score_payload: dict[str, dict[str, Any]] = {key: dict(value) for key, value in base_rows.items()}
    saved_models: dict[str, Any] = {}
    for method, groups in method_groups.items():
        train = [group for group in groups if group.split == "train"]
        validation = [group for group in groups if group.split == "val"]
        test = [group for group in groups if group.split == "test"]
        model, threshold, oof = fit_oof(
            train,
            ridge=float(values["ridge"]),
            minimum_threshold=float(values["minimum_threshold"]),
            target_precision=float(values["target_precision"]),
        )
        validation_metrics = evaluate_groups(validation, model, threshold)
        test_metrics = evaluate_groups(test, model, threshold)
        gate = validation_gate(validation_metrics, values)
        report["methods"][method] = {
            "feature_names": METHOD_FEATURE_NAMES[method],
            "threshold": threshold,
            "train_oof_exact": oof,
            "validation": validation_metrics,
            "validation_gate": gate,
            "test": test_metrics,
            "model": serializable_model(model),
        }
        saved_models[method] = model
        for group in groups:
            probabilities = predict_logistic(model, group.features)
            score_payload[group.trace_id].setdefault("methods", {})[method] = {
                "probabilities": probabilities.tolist(),
                "predictions": (probabilities >= threshold).astype(int).tolist(),
            }
        print(
            f"{method}: train-LOO F1={oof['f1']:.3f}; "
            f"val F1={validation_metrics['exact']['f1']:.3f}; "
            f"val ±1 F1={validation_metrics['tolerant_1']['f1']:.3f}; "
            f"gate={'PASS' if gate['passed'] else 'FAIL'}",
            flush=True,
        )

    passing = [
        name for name, result in report["methods"].items()
        if result["validation_gate"]["passed"]
    ]
    selected = max(
        passing,
        key=lambda name: (
            report["methods"][name]["validation"]["exact"]["f1"],
            report["methods"][name]["validation"]["tolerant_1"]["f1"],
            report["methods"][name]["validation"]["exact"]["precision"],
        ),
        default=None,
    )
    report["selected_method"] = selected
    report["safe_to_expand_labels"] = selected is not None
    report["caution"] = (
        "Only four validation and two test trajectories are available. Treat a pass "
        "as a candidate for a larger manually audited pilot, not proof of generalization."
    )
    atomic_json(paths["tables"] / "prm_graph_labeler_benchmark.json", report)
    write_jsonl(
        paths["tables"] / "prm_graph_labeler_sentence_scores.jsonl",
        [score_payload[str(trace["trace_id"])] for trace in traces],
    )
    model_arrays: dict[str, Any] = {}
    for method, model in saved_models.items():
        model_arrays[f"{method}_mean"] = np.asarray(model["mean"])
        model_arrays[f"{method}_scale"] = np.asarray(model["scale"])
        model_arrays[f"{method}_coefficients"] = np.asarray(model["coefficients"])
        model_arrays[f"{method}_intercept"] = np.asarray(model["intercept"])
        model_arrays[f"{method}_threshold"] = np.asarray(
            report["methods"][method]["threshold"]
        )
    _atomic_npz(paths["models"] / "prm_graph_labeler_probes.npz", **model_arrays)
    print(
        f"Selected method: {selected or 'none (all validation gates failed)'}. "
        f"Report: {paths['tables'] / 'prm_graph_labeler_benchmark.json'}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    values = settings(config)
    records = strong_records(paths)
    traces = labeled_traces(paths, records)
    if args.command == "collect-prm":
        collect_prm(paths, traces, values, args.limit)
    elif args.command == "collect-embeddings":
        collect_embeddings(paths, traces, values, args.limit)
    elif args.command == "evaluate":
        evaluate(paths, traces, records, values)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
