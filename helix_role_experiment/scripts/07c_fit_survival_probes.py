from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import write_csv
from helix_role_experiment.config import ensure_output_dirs, load_config, read_jsonl
from helix_role_experiment.readiness import (
    build_survival_rows, concordance_index, fit_exponential_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and select sentence-level time-to-next-subgoal probes")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


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
    annotations = {row["trace_id"]: row["annotations"] for row in read_jsonl(paths["tables"] / "sentence_annotations.jsonl")}
    trace_files = sorted((paths["traces"] / "readiness_baseline").glob("*.json"))
    if not trace_files:
        raise RuntimeError("run 07a collection first")
    layers = json.loads(trace_files[0].read_text(encoding="utf-8"))["layers"]
    layer_results = []
    layer_payload = {}
    ridge = float(config["probe"].get("ridge", 0.001))
    for layer in layers:
        rows = []
        vectors = []
        adjacent_norms = []
        for source in trace_files:
            trace = json.loads(source.read_text(encoding="utf-8"))
            if trace["trace_id"] not in annotations:
                raise RuntimeError(f"missing annotations for {trace['trace_id']}")
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
    }])
    print(f"Selected layer {layer}; held-out c-index={test_cindex:.3f}; probe={model_path}")


if __name__ == "__main__":
    main()
