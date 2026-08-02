from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from _common import write_csv
from helix_role_experiment.config import atomic_json, ensure_output_dirs, load_config, read_jsonl
from helix_role_experiment.readiness import (
    build_survival_rows, concordance_index, fit_exponential_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and select sentence-level time-to-next-subgoal probes")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-partial-labels",
        action="store_true",
        help=(
            "Rebuild the annotation table from all valid cached results and fit "
            "only labeled trajectories. Intended for exploratory pilot runs."
        ),
    )
    return parser.parse_args()


def rebuild_partial_annotations(config: dict, paths: dict[str, Path]) -> None:
    source = Path(__file__).resolve().parent / "07b_label_subgoal_events.py"
    spec = importlib.util.spec_from_file_location("label_subgoal_events_07b_for_07c", source)
    labeler = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(labeler)
    labeler.validate_cached(config, paths, allow_missing=True)


def coverage_report(
    traces: list[dict], annotation_ids: set[str]
) -> dict:
    counts: dict[str, dict[str, int]] = {}
    labeled = []
    missing = []
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id not in annotation_ids:
            missing.append(trace_id)
            continue
        labeled.append(trace_id)
        domain = str(trace["domain"])
        split = str(trace["split"])
        counts.setdefault(domain, {}).setdefault(split, 0)
        counts[domain][split] += 1
    return {
        "total_trajectories": len(traces),
        "labeled_trajectories": len(labeled),
        "missing_trajectories": len(missing),
        "trajectory_counts_by_domain_and_split": counts,
        "labeled_trace_ids": labeled,
        "missing_trace_ids": missing,
        "exploratory_partial_fit": bool(missing),
    }


def finite_nuisance(rows: list[dict]) -> np.ndarray:
    values = np.asarray([
        [
            np.log1p(float(row["token_start"])), float(row["sentence_index"]),
            np.log1p(float(row["previous_sentence_tokens"])),
            float(row["eos_logit"]) if row["eos_logit"] is not None else np.nan,
            float(row["token_entropy"]) if row["token_entropy"] is not None else np.nan,
        ] for row in rows
    ], dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(values)
    values[missing] = np.take(medians, np.where(missing)[1])
    return values


def subset(rows: list[dict], activations: np.ndarray, names: set[str]):
    mask = np.asarray([row["split"] in names for row in rows])
    duration = np.asarray([row["duration"] for row in rows], dtype=float)[mask]
    event = np.asarray([row["event"] for row in rows], dtype=float)[mask]
    return activations[mask], duration, event, mask


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.allow_partial_labels:
        rebuild_partial_annotations(config, paths)
    annotations = {row["trace_id"]: row["annotations"] for row in read_jsonl(paths["tables"] / "sentence_annotations.jsonl")}
    trace_files = sorted((paths["traces"] / "readiness_baseline").glob("*.json"))
    if not trace_files:
        raise RuntimeError("run 07a collection first")
    traces = [json.loads(source.read_text(encoding="utf-8")) for source in trace_files]
    coverage = coverage_report(traces, set(annotations))
    atomic_json(paths["tables"] / "readiness_label_coverage.json", coverage)
    if coverage["missing_trajectories"] and not args.allow_partial_labels:
        first = coverage["missing_trace_ids"][0]
        raise RuntimeError(
            f"missing annotations for {first}; finish 07b or rerun with "
            "--allow-partial-labels for an exploratory partial fit"
        )
    labeled_traces = [trace for trace in traces if trace["trace_id"] in annotations]
    if not labeled_traces:
        raise RuntimeError("no fully labeled trajectories are available")
    split_counts = {
        split: sum(trace["split"] == split for trace in labeled_traces)
        for split in ("train", "val", "test")
    }
    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise RuntimeError(
            "partial labels must include at least one train and one validation "
            f"trajectory; observed {split_counts}"
        )
    domains = sorted({str(trace["domain"]) for trace in labeled_traces})
    if len(domains) < 2:
        print(
            f"WARNING: partial labels cover only {domains}; cross-domain "
            "generalization cannot be assessed.",
            flush=True,
        )
    print(
        f"Using {len(labeled_traces)}/{len(traces)} labeled trajectories; "
        f"split counts={split_counts}; domains={domains}. Coverage report: "
        f"{paths['tables'] / 'readiness_label_coverage.json'}",
        flush=True,
    )
    layers = labeled_traces[0]["layers"]
    layer_results = []
    layer_payload = {}
    ridge = float(config["probe"].get("ridge", 0.001))
    for layer in layers:
        rows = []
        vectors = []
        adjacent_norms = []
        for trace in labeled_traces:
            local_rows = build_survival_rows(trace, annotations[trace["trace_id"]], int(layer))
            stored = np.load(trace["activation_file"])[f"layer_{layer}"].astype(np.float32)
            sentence_index = {value["sentence_id"]: index for index, value in enumerate(trace["sentences"])}
            local_vectors = np.asarray([stored[sentence_index[row["sentence_id"]]] for row in local_rows])
            rows.extend(local_rows)
            vectors.extend(local_vectors)
            if trace["split"] == "train" and len(local_vectors) > 1:
                adjacent_norms.extend(np.linalg.norm(np.diff(local_vectors, axis=0), axis=1).tolist())
        activations = np.asarray(vectors, dtype=np.float32)
        train_x, train_t, train_e, _ = subset(rows, activations, {"train"})
        val_x, val_t, val_e, val_mask = subset(rows, activations, {"val"})
        probe = fit_exponential_probe(train_x, train_t, train_e, ridge=ridge, layer=int(layer))
        val_score = probe.score(val_x)
        cindex = concordance_index(val_t, val_e, val_score)
        nuisance = finite_nuisance(rows)
        nuisance_train, _, _, _ = subset(rows, nuisance, {"train"})
        nuisance_val, _, _, _ = subset(rows, nuisance, {"val"})
        nuisance_probe = fit_exponential_probe(nuisance_train, train_t, train_e, ridge=ridge)
        nuisance_cindex = concordance_index(val_t, val_e, nuisance_probe.score(nuisance_val))
        layer_results.append({
            "layer": int(layer), "train_rows": len(train_x), "val_rows": len(val_x),
            "val_cindex": cindex, "nuisance_val_cindex": nuisance_cindex,
            "increment_over_nuisance": cindex - nuisance_cindex,
        })
        layer_payload[int(layer)] = (rows, activations, adjacent_norms)
    eligible = [row for row in layer_results if np.isfinite(row["val_cindex"])]
    if not eligible:
        raise RuntimeError("no layer produced a finite validation concordance")
    chosen = max(eligible, key=lambda row: (row["val_cindex"], row["increment_over_nuisance"]))
    layer = int(chosen["layer"])
    rows, activations, adjacent_norms = layer_payload[layer]
    fit_x, fit_t, fit_e, _ = subset(rows, activations, {"train", "val"})
    test_x, test_t, test_e, _ = subset(rows, activations, {"test"})
    final_probe = fit_exponential_probe(fit_x, fit_t, fit_e, ridge=ridge, layer=layer)
    final_probe.threshold = float(np.quantile(final_probe.score(fit_x), float(config["probe"].get("trigger_quantile", 0.67))))
    final_probe.native_step_norm = float(np.median(adjacent_norms)) if adjacent_norms else 1.0
    test_cindex = concordance_index(test_t, test_e, final_probe.score(test_x))
    model_path = paths["models"] / "readiness_survival_probe.npz"
    final_probe.save(model_path)
    for row in layer_results:
        row["selected"] = int(row["layer"] == layer)
    write_csv(paths["tables"] / "readiness_layer_selection.csv", layer_results)
    write_csv(paths["tables"] / "readiness_probe_test.csv", [{
        "layer": layer, "test_rows": len(test_x), "test_cindex": test_cindex,
        "threshold": final_probe.threshold, "native_step_norm": final_probe.native_step_norm,
        "passes_nuisance_diagnostic": chosen["increment_over_nuisance"] > 0,
        "labeled_trajectories": len(labeled_traces),
        "total_trajectories": len(traces),
        "exploratory_partial_fit": coverage["exploratory_partial_fit"],
    }])
    print(f"Selected layer {layer}; held-out c-index={test_cindex:.3f}; probe={model_path}")


if __name__ == "__main__":
    main()
