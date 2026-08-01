import os
import sys
import math
import time
import random
import queue
import argparse
import csv
import json
import hashlib
import pickle
import traceback
import multiprocessing as mp
import numpy as np
import torch

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_LAYER = 15
CALIBRATION_CACHE_VERSION = 3
_MAX_CONCURRENT_SEQS_ENV = os.getenv("HELIX_MAX_CONCURRENT_SEQS")
MAX_CONCURRENT_SEQS_OVERRIDE = (
    int(_MAX_CONCURRENT_SEQS_ENV)
    if _MAX_CONCURRENT_SEQS_ENV not in (None, "")
    else None
)
GPU_MEMORY_UTILIZATION = float(
    os.getenv("HELIX_GPU_MEMORY_UTILIZATION", "0.85")
)
MAX_MODEL_LEN = int(os.getenv("HELIX_MAX_MODEL_LEN", "8192"))
MAX_NEW_TOKENS = 1024
CALIBRATION_MAX_INPUT_TOKENS = int(
    os.getenv("HELIX_CALIBRATION_MAX_INPUT_TOKENS", "2048")
)
DEFAULT_REMAINING_PRIOR = 150
N_CALIB_PROMPTS = int(os.getenv("HELIX_N_CALIB_PROMPTS", "40"))
N_TRACE_PROMPTS = int(os.getenv("HELIX_N_TRACE_PROMPTS", "40"))
TRACE_BATCH_SIZE = int(os.getenv("HELIX_TRACE_BATCH_SIZE", "8"))
N_CONTAM = 1
ROUTING_DEADBAND_TOKENS = float(
    os.getenv("HELIX_ROUTING_DEADBAND_TOKENS", "32")
)
ROUTING_DEADBAND_SEC = float(
    os.getenv("HELIX_ROUTING_DEADBAND_SEC", "0.010")
)
if MAX_CONCURRENT_SEQS_OVERRIDE is not None and MAX_CONCURRENT_SEQS_OVERRIDE <= 0:
    raise ValueError("HELIX_MAX_CONCURRENT_SEQS must be positive when set")
if not 0.0 < GPU_MEMORY_UTILIZATION < 1.0:
    raise ValueError("HELIX_GPU_MEMORY_UTILIZATION must be between 0 and 1")
if TRACE_BATCH_SIZE <= 0:
    raise ValueError("HELIX_TRACE_BATCH_SIZE must be positive")
if ROUTING_DEADBAND_TOKENS < 0 or ROUTING_DEADBAND_SEC < 0:
    raise ValueError("routing deadbands must be non-negative")
PRIORITY_TTFT_DEADLINE_SEC = float(
    os.getenv("HELIX_PRIORITY_TTFT_DEADLINE_SEC", "20.0")
)
PRIORITY_ITL_RESCUE_SEC = float(
    os.getenv("HELIX_PRIORITY_ITL_RESCUE_SEC", "5.0")
)
HELIX_PRIORITY_COHORT_SIZE = int(
    os.getenv("HELIX_PRIORITY_COHORT_SIZE", "6")
)
ORACLE_PRIORITY_COHORT_SIZE = int(
    os.getenv("ORACLE_PRIORITY_COHORT_SIZE", "12")
)
PRIORITY_LENGTH_BUCKET_TOKENS = int(
    os.getenv("HELIX_PRIORITY_LENGTH_BUCKET_TOKENS", "32")
)
PRIORITY_LENGTH_HYSTERESIS_BUCKETS = int(
    os.getenv("HELIX_PRIORITY_LENGTH_HYSTERESIS_BUCKETS", "2")
)
PRIORITY_TTFT_BASELINE_RATIO = float(
    os.getenv("HELIX_PRIORITY_TTFT_BASELINE_RATIO", "0.80")
)
PRIORITY_ITL_BASELINE_MULTIPLIER = float(
    os.getenv("HELIX_PRIORITY_ITL_BASELINE_MULTIPLIER", "8.0")
)
PRIORITY_ITL_RESCUE_MIN_SEC = float(
    os.getenv("HELIX_PRIORITY_ITL_RESCUE_MIN_SEC", "0.50")
)
PRIORITY_UNSTARTED_BASE = int(
    os.getenv("HELIX_PRIORITY_UNSTARTED_BASE", "10000000")
)
ENABLE_EXPERIMENTAL_VLLM_PREEMPTION = (
    os.getenv("HELIX_ENABLE_VLLM_PREEMPTION", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
SERVICE_TIME_EWMA_ALPHA = 0.10
NEW_REQUEST_RISK_STD = 0.0
DEFAULT_DECODE_STEP_SEC = 0.030
DEFAULT_PREFILL_SEC_PER_TOKEN = 0.00005
PREFILL_EWMA_ALPHA = 0.10
MIN_PREFILL_SEC = 0.0005
SCHEDULER_POLL_SEC = 0.002
POLICY_SEED = 0

WORKER_READY_TIMEOUT_SEC = 240
ENGINE_WARMUP_OUTPUT_TOKENS = 8
DEFAULT_TARGET_UTILIZATION = 0.95
DEFAULT_BURSTINESS = 0.50
DEFAULT_QUEUE_WAIT_BUDGET_SEC = 5.0
DEFAULT_TTFT_SLO_SEC = 2.0
DEFAULT_ITL_SLO_SEC = 0.100
MEASUREMENT_WARMUP_FRACTION = 0.10
MEASUREMENT_COOLDOWN_FRACTION = 0.10
DEFAULT_BENCHMARK_REQUESTS = 600
DEFAULT_CAPACITY_PROBE_REQUESTS = 64
DEFAULT_LOAD_LEVELS = (0.95, 1.00, 1.05)
DEFAULT_PREFLIGHT_REQUESTS = 32
DEFAULT_PREFLIGHT_LOAD_LEVELS = (0.75, 0.90, 1.00)
DEFAULT_PREFLIGHT_BURSTINESS_LEVELS = (1.00, 0.50, 0.25)
BOOTSTRAP_SAMPLES = 400

DIAG_N_CALLS = 5
DIAG_PERIOD = 200

# Shortened lock-in: warmup 6, boot window 6
TRACKER_KWARGS = dict(
    Kp=0.175, Ki=0.003, warmup=6, dc_warmup=6, trust_sharpness=10.0,
    lockin_window_tokens=6, omega_boot_window=6, boot_shrink=0.5,
    omega_relax=0.02,
)

# The late half-cycle crossing (pi) is the most accurate online checkpoint.
# Legacy dictionary names are retained because the compiled kernel has two stages.
THRESHOLDS = {'quarter': math.pi / 2, 'half': math.pi}

# ---------------------------------------------------------------------------
# CALIBRATION & AFFINE FITTING
# ---------------------------------------------------------------------------
def get_activations(model, tokenizer, prompt, target_layer=TARGET_LAYER, max_new_tokens=1024):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activations = []

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            activations.append(h[:, -1, :].detach().cpu().float().numpy())
        return output

    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    del inputs
    acts = np.concatenate(activations, axis=0) if activations else np.empty((0, model.config.hidden_size))
    return acts

def isolate_k1_and_residual(acts):
    from scipy.fft import rfft, irfft
    L = acts.shape[0]
    mu = np.mean(acts, axis=0)
    x_ac = acts - mu
    F = rfft(x_ac, axis=0)
    F_k1 = np.zeros_like(F)
    if F.shape[0] > 1: F_k1[1] = F[1]
    x_helix = irfft(F_k1, n=L, axis=0)
    residual = x_ac - x_helix
    return x_helix, residual, mu

def build_helix_basis_cache(model, tokenizer, calibration_prompts, target_layer=TARGET_LAYER,
                             max_new_tokens=200, n_prompts=20, n_contam=1):
    from sklearn.decomposition import PCA
    v0s, v1s, mus, lens, s0s, s1s = [], [], [], [], [], []
    contam_dir_lists = [[] for _ in range(n_contam)]
    ref_v0, ref_v1 = None, None
    ref_contam = [None] * n_contam
    for prompt in calibration_prompts[:n_prompts]:
        try:
            acts = get_activations(model, tokenizer, prompt, target_layer, max_new_tokens)
            if len(acts) < 8: continue
            x_helix, residual, mu = isolate_k1_and_residual(acts)
            pca = PCA(n_components=5).fit(x_helix)
            v0_curr, v1_curr = pca.components_[0], pca.components_[1]
            pc0_chk, pc1_chk = x_helix @ v0_curr, x_helix @ v1_curr
            theta_chk = np.unwrap(np.arctan2(pc1_chk, pc0_chk))
            if theta_chk[-1] - theta_chk[0] < 0: v1_curr = -v1_curr
            if ref_v0 is None:
                ref_v0, ref_v1 = v0_curr, v1_curr
            else:
                if np.dot(v0_curr, ref_v0) < 0: v0_curr = -v0_curr
                if np.dot(v1_curr, ref_v1) < 0: v1_curr = -v1_curr
            pca_res = PCA(n_components=max(n_contam, 2)).fit(residual)
            for j in range(n_contam):
                cdir = pca_res.components_[j]
                if ref_contam[j] is None: ref_contam[j] = cdir
                elif np.dot(cdir, ref_contam[j]) < 0: cdir = -cdir
                contam_dir_lists[j].append(cdir)
            mus.append(mu); v0s.append(v0_curr); v1s.append(v1_curr)
            s0s.append(np.sqrt(pca.explained_variance_[0]))
            s1s.append(np.sqrt(pca.explained_variance_[1]))
            lens.append(len(acts))
        except Exception:
            continue
    if not v0s: raise RuntimeError("No valid calibration prompts.")
    v0_mean = np.mean(v0s, axis=0); v0_mean /= np.linalg.norm(v0_mean) + 1e-8
    v1_mean = np.mean(v1s, axis=0)
    v1_mean -= np.dot(v1_mean, v0_mean) * v0_mean
    v1_mean /= np.linalg.norm(v1_mean) + 1e-8
    contam_dirs = []
    for j in range(n_contam):
        cd = np.mean(contam_dir_lists[j], axis=0)
        for prev in contam_dirs: cd = cd - np.dot(cd, prev) * prev
        norm = np.linalg.norm(cd)
        if norm > 1e-6: contam_dirs.append(cd / norm)
    for cd in contam_dirs:
        v0_mean = v0_mean - np.dot(v0_mean, cd) * cd
        v1_mean = v1_mean - np.dot(v1_mean, cd) * cd
    v0_mean /= np.linalg.norm(v0_mean) + 1e-8
    v1_mean -= np.dot(v1_mean, v0_mean) * v0_mean
    v1_mean /= np.linalg.norm(v1_mean) + 1e-8
    mean_len = float(np.mean(lens))
    print(f"Basis cache ready. Typical baseline length ~{mean_len:.0f} tokens.")
    return {
        'mu': np.mean(mus, axis=0),
        'v0': v0_mean, 'v1': v1_mean,
        's0': float(np.mean(s0s)), 's1': float(np.mean(s1s)),
        'omega_prior': 2 * math.pi / mean_len,
        'mean_len': mean_len
    }

def collect_offline_traces_both(
    model,
    tokenizer,
    prompts,
    cache,
    target_layer,
    max_new_tokens=400,
    batch_size=TRACE_BATCH_SIZE,
):
    """Collect independent natural-EOS traces in compacting decode batches.

    Hugging Face ``generate`` was previously called once per prompt, leaving
    almost all of the GPU idle. A first batched implementation kept completed
    rows in every subsequent decode iteration, so one long answer forced the
    model to recompute all of the short rows. This version removes completed
    rows (including their KV cache entries) after every step.
    """
    if batch_size <= 0:
        raise ValueError("trace batch_size must be positive")

    traces = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)
    else:
        eos_token_ids = tuple(eos_token_ids or ())

    def eos_mask(token_ids):
        result = torch.zeros_like(token_ids, dtype=torch.bool)
        for eos_id in eos_token_ids:
            result |= token_ids.eq(int(eos_id))
        return result

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch_prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        current_batch_size = len(batch_prompts)
        tracker = BatchedCausalPLLTracker(
            cache,
            batch_size=current_batch_size,
            device=model.device,
            dtype=torch.float32,
            **TRACKER_KWARGS,
        )
        decode_slots = None

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.ndim == 3 and h.shape[1] == 1 and decode_slots is not None:
                tracker.update_slots(
                    h[:, -1, :].detach().to(torch.float32),
                    decode_slots,
                )
            return output

        def select_cache_rows(past_key_values, keep_indices):
            """Select live batch rows for both modern and legacy HF caches."""
            if hasattr(past_key_values, "batch_select_indices"):
                selected = past_key_values.batch_select_indices(keep_indices)
                return past_key_values if selected is None else selected
            return tuple(
                tuple(
                    value.index_select(0, keep_indices)
                    if torch.is_tensor(value) else value
                    for value in layer
                )
                for layer in past_key_values
            )

        handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                prefill_position_ids = (
                    inputs.attention_mask.long().cumsum(dim=-1) - 1
                )
                prefill_position_ids.masked_fill_(
                    inputs.attention_mask.eq(0), 0
                )
                outputs = model.model(
                    **inputs,
                    position_ids=prefill_position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                next_tokens = model.lm_head(
                    outputs.last_hidden_state[:, -1, :]
                ).argmax(dim=-1)
                generated_lengths = torch.ones(
                    current_batch_size,
                    dtype=torch.long,
                    device=model.device,
                )
                finished = eos_mask(next_tokens)
                past_key_values = outputs.past_key_values
                attention_mask = inputs.attention_mask
                active_indices = torch.nonzero(
                    (~finished)
                    & generated_lengths.lt(int(max_new_tokens)),
                    as_tuple=False,
                ).flatten()
                if active_indices.numel() > 0:
                    next_tokens = next_tokens.index_select(0, active_indices)
                    attention_mask = attention_mask.index_select(
                        0, active_indices
                    )
                    past_key_values = select_cache_rows(
                        past_key_values, active_indices
                    )

                while active_indices.numel() > 0:
                    decode_slots = active_indices.tolist()

                    position_ids = attention_mask.long().sum(
                        dim=1, keepdim=True
                    )
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (active_indices.numel(), 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
                    outputs = model.model(
                        input_ids=next_tokens.unsqueeze(1),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    candidate_tokens = model.lm_head(
                        outputs.last_hidden_state[:, -1, :]
                    ).argmax(dim=-1)
                    generated_lengths[active_indices] += 1
                    row_finished = (
                        eos_mask(candidate_tokens)
                        | generated_lengths[active_indices].ge(
                            int(max_new_tokens)
                        )
                    )
                    finished[active_indices] |= row_finished
                    keep_rows = torch.nonzero(
                        ~row_finished, as_tuple=False
                    ).flatten()
                    if keep_rows.numel() == 0:
                        active_indices = active_indices[:0]
                        break
                    active_indices = active_indices.index_select(0, keep_rows)
                    next_tokens = candidate_tokens.index_select(0, keep_rows)
                    attention_mask = attention_mask.index_select(0, keep_rows)
                    past_key_values = select_cache_rows(
                        past_key_values, keep_rows
                    )
                decode_slots = None
        finally:
            handle.remove()

        for row_idx in range(current_batch_size):
            rec = {"true": int(generated_lengths[row_idx].item())}
            for label in THRESHOLDS:
                crossed = bool(tracker.crossed[label][row_idx].item())
                rec[f"{label}_crossed"] = crossed
                rec[f"{label}_t"] = (
                    int(tracker.cross_step[label][row_idx].item())
                    if crossed else None
                )
                rec[f"{label}_pred"] = (
                    float(tracker.pred_total[label][row_idx].item())
                    if crossed else None
                )
            traces.append(rec)

        completed = min(batch_start + current_batch_size, len(prompts))
        print(
            f"  historical traces: {completed}/{len(prompts)} "
            f"(batch size {current_batch_size})"
        )
        del inputs, outputs, tracker, past_key_values
        torch.cuda.empty_cache()

    tokenizer.padding_side = old_padding_side
    return traces

def calibrate_thresholds_and_gap(traces):
    calib = {}
    for label in THRESHOLDS:
        yp = np.array([tr[f'{label}_pred'] for tr in traces if tr[f'{label}_crossed']])
        yt = np.array([tr['true'] for tr in traces if tr[f'{label}_crossed']])
        if len(yp) < 3:
            calib[f'a_{label}'], calib[f'b_{label}'] = 1.0, 0.0
            continue
        A = np.vstack([yp, np.ones_like(yp)]).T
        (a, b), *_ = np.linalg.lstsq(A, yt, rcond=None)
        calib[f'a_{label}'], calib[f'b_{label}'] = float(a), float(b)
        resid = (a * yp + b) - yt
        calib[f'{label}_ci_width'] = float(1.96 * np.std(resid))
        print(f"[{label}] n={len(yp)} a={a:.3f} b={b:.1f} residual_std={np.std(resid):.1f} "
              f"ci_width(95%)={calib[f'{label}_ci_width']:.1f}")
    gaps = [tr['half_t'] - tr['quarter_t'] for tr in traces
            if tr['quarter_crossed'] and tr['half_crossed']]
    q_times = [tr['quarter_t'] for tr in traces
               if tr['quarter_crossed'] and tr['half_crossed']]
    if gaps:
        gaps = np.array(gaps); q_times = np.array(q_times)
        ratio = (q_times + gaps) / q_times
        calib['gap_mean'] = float(np.mean(gaps))
        calib['gap_std'] = float(np.std(gaps))
        calib['half_over_quarter_ratio_mean'] = float(np.mean(ratio))
        calib['half_over_quarter_ratio_std'] = float(np.std(ratio))
        print(f"\nquarter->half gap: mean={calib['gap_mean']:.1f} tok, std={calib['gap_std']:.1f} tok")
        print(f"t_half / t_quarter ratio: mean={calib['half_over_quarter_ratio_mean']:.2f}")
    else:
        calib['half_over_quarter_ratio_mean'] = 2.0
    return calib

def _wrap_delta(a, b):
    return torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi

# ---------------------------------------------------------------------------
# CAUSAL PLL TRACKER
# ---------------------------------------------------------------------------
def _pll_update_kernel(
    h_last, active_mask,
    mu, v0, v1, s0, s1,
    count, rms_est, sum0, sum1,
    I_est, Q_est, theta_hat, omega_hat,
    theta_ref, ref_captured,
    raw_theta_running, prev_raw_phase, has_prev_raw_phase,
    raw_theta_snapshot, snapshot_taken,
    crossed_q, cross_step_q, pred_total_q,
    crossed_h, cross_step_h, pred_total_h,
    mag_floor: float, trust_sharpness: float, Kp: float, Ki: float,
    agc_target: float, agc_alpha: float,
    warmup: int, dc_warmup: int,
    omega_lo: float, omega_hi: float,
    lockin_alpha: float, omega_boot_window: int, snapshot_step: int,
    boot_shrink: float, omega_relax: float, omega_prior: float,
    thresh_q: float, thresh_h: float,
):
    dtype = h_last.dtype
    x_ac = h_last - mu
    pc0, pc1 = (x_ac @ v0) / s0, (x_ac @ v1) / s1
    am = active_mask.to(dtype)
 
    count = count + am
    sum0 = sum0 + pc0 * am
    sum1 = sum1 + pc1 * am
    sc = count.clamp(min=1)
    pc0c, pc1c = pc0 - (sum0 / sc), pc1 - (sum1 / sc)
    raw_mag = torch.hypot(pc0c, pc1c)
 
    first = (count == 1)
    rms_est = torch.where(first, raw_mag.clamp(min=1e-3),
                           agc_alpha * raw_mag + (1 - agc_alpha) * rms_est)
    gain = (agc_target / rms_est.clamp(min=1e-3)).clamp(max=50.0)
    pc0n, pc1n = pc0c * gain, pc1c * gain
 
    raw_phase = torch.atan2(pc1c, pc0c)
    delta = _wrap_delta(raw_phase, prev_raw_phase)
    delta = torch.where(has_prev_raw_phase, delta, torch.zeros_like(delta))
    raw_theta_running = torch.where(active_mask, raw_theta_running + delta, raw_theta_running)
    prev_raw_phase = raw_phase
    has_prev_raw_phase = has_prev_raw_phase | active_mask
 
    should_snapshot = active_mask & (~snapshot_taken) & (count == snapshot_step)
    raw_theta_snapshot = torch.where(should_snapshot, raw_theta_running, raw_theta_snapshot)
    snapshot_taken = snapshot_taken | should_snapshot
 
    cos_h, sin_h = torch.cos(theta_hat), torch.sin(theta_hat)
    I_raw, Q_raw = pc0n * cos_h + pc1n * sin_h, pc1n * cos_h - pc0n * sin_h
    I_est = lockin_alpha * I_raw + (1 - lockin_alpha) * I_est
    Q_est = lockin_alpha * Q_raw + (1 - lockin_alpha) * Q_est
    mag_n = torch.hypot(I_est, Q_est)
    err = torch.atan2(Q_est, I_est)
    trust = am * (count > dc_warmup).to(dtype) * torch.sigmoid((mag_n - mag_floor) * trust_sharpness)
 
    omega_hat = omega_hat + Ki * err * trust + omega_relax * (omega_prior - omega_hat) * (1 - trust)
    omega_hat = torch.clamp(omega_hat, min=omega_lo, max=omega_hi)
    theta_new = theta_hat + omega_hat + Kp * err * trust
 
    just_warmed_up = (count == dc_warmup + 1)
    omega_boot = boot_shrink * ((raw_theta_running - raw_theta_snapshot) / omega_boot_window).clamp(min=omega_lo, max=omega_hi) \
                 + (1 - boot_shrink) * omega_prior
    omega_hat = torch.where(just_warmed_up, omega_boot, omega_hat)
    theta_hat = torch.where(active_mask, torch.where(just_warmed_up, raw_phase, theta_new), theta_hat)
 
    effective_warmup = max(warmup, dc_warmup + 1)
    should_capture = active_mask & (~ref_captured) & (count >= effective_warmup)
    theta_ref = torch.where(should_capture, theta_hat, theta_ref)
    ref_captured = ref_captured | should_capture
    rel_theta = theta_hat - theta_ref
 
    newly_q = active_mask & ref_captured & ~crossed_q & (torch.abs(rel_theta) >= thresh_q)
    cross_step_q = torch.where(newly_q, count.to(torch.long), cross_step_q)
    pred_q = effective_warmup + (count - effective_warmup) / (thresh_q / (2 * math.pi))
    pred_total_q = torch.where(newly_q, pred_q, pred_total_q)
    crossed_q = crossed_q | newly_q
 
    newly_h = active_mask & ref_captured & ~crossed_h & (torch.abs(rel_theta) >= thresh_h)
    cross_step_h = torch.where(newly_h, count.to(torch.long), cross_step_h)
    pred_h = effective_warmup + (count - effective_warmup) / (thresh_h / (2 * math.pi))
    pred_total_h = torch.where(newly_h, pred_h, pred_total_h)
    crossed_h = crossed_h | newly_h
 
    return (count, rms_est, sum0, sum1, I_est, Q_est, theta_hat, omega_hat,
            theta_ref, ref_captured, raw_theta_running, prev_raw_phase,
            has_prev_raw_phase, raw_theta_snapshot, snapshot_taken,
            crossed_q, cross_step_q, pred_total_q,
            crossed_h, cross_step_h, pred_total_h, mag_n)
 

_compiled_pll_update_kernel = None

def _get_compiled_kernel():
    global _compiled_pll_update_kernel
    if _compiled_pll_update_kernel is None:
        _compiled_pll_update_kernel = torch.compile(
            _pll_update_kernel, dynamic=True
        )
    return _compiled_pll_update_kernel
    
class BatchedCausalPLLTracker:
    THRESHOLDS = THRESHOLDS
    def __init__(self, cache, batch_size, device, dtype=torch.float32,
                 mag_floor=0.15, trust_sharpness=10.0, Kp=0.15, Ki=0.02,
                 agc_target=1.0, agc_alpha=0.05, warmup=8, dc_warmup=15,
                 omega_clamp=None, lockin_window_tokens=6,
                 omega_boot_window=20, boot_shrink=0.5, omega_relax=0.02,
                 use_compile=True):
        self.device, self.dtype = device, dtype
        self.mu = torch.as_tensor(cache['mu'], dtype=dtype, device=device)
        self.v0 = torch.as_tensor(cache['v0'], dtype=dtype, device=device)
        self.v1 = torch.as_tensor(cache['v1'], dtype=dtype, device=device)
        self.s0, self.s1, self.omega_prior = cache['s0'], cache['s1'], cache['omega_prior']
        self.B = batch_size
        self.mag_floor, self.trust_sharpness = mag_floor, trust_sharpness
        self.Kp, self.Ki = Kp, Ki
        self.agc_target, self.agc_alpha = agc_target, agc_alpha
        self.warmup, self.dc_warmup = warmup, dc_warmup
        self.boot_shrink, self.omega_relax = boot_shrink, omega_relax
        if omega_clamp is None:
            omega_clamp = (max(1e-4, self.omega_prior / 2.0), self.omega_prior * 2.0)
        self.omega_clamp = omega_clamp
        self.lockin_alpha = 2.0 / (lockin_window_tokens + 1.0)
        self.omega_boot_window = max(1, min(omega_boot_window, self.dc_warmup))
        self.snapshot_step = max(1, self.dc_warmup + 1 - self.omega_boot_window)
        self.count = torch.zeros(self.B, device=device, dtype=dtype)
        self.rms_est = torch.full((self.B,), agc_target, device=device, dtype=dtype)
        self.sum0 = torch.zeros(self.B, device=device, dtype=dtype)
        self.sum1 = torch.zeros(self.B, device=device, dtype=dtype)
        self.I_est = torch.zeros(self.B, device=device, dtype=dtype)
        self.Q_est = torch.zeros(self.B, device=device, dtype=dtype)
        self.theta_hat = torch.zeros(self.B, device=device, dtype=dtype)
        self.omega_hat = torch.full((self.B,), self.omega_prior, device=device, dtype=dtype)
        self.theta_ref = torch.zeros(self.B, device=device, dtype=dtype)
        self.ref_captured = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.t = 0
        self.raw_theta_running = torch.zeros(self.B, device=device, dtype=dtype)
        self.prev_raw_phase = torch.zeros(self.B, device=device, dtype=dtype)
        self.has_prev_raw_phase = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.raw_theta_snapshot = torch.zeros(self.B, device=device, dtype=dtype)
        self.snapshot_taken = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.crossed = {k: torch.zeros(self.B, dtype=torch.bool, device=device) for k in self.THRESHOLDS}
        self.cross_step = {k: torch.full((self.B,), -1, dtype=torch.long, device=device) for k in self.THRESHOLDS}
        self.pred_total = {k: torch.zeros(self.B, device=device, dtype=dtype) for k in self.THRESHOLDS}
        self._kernel = _get_compiled_kernel() if use_compile else _pll_update_kernel

    @torch.inference_mode()
    def reset_slots(self, slot_indices):
        """Reset PLL state, including state produced by vLLM inference mode."""
        if slot_indices is None:
            return
        idx = torch.as_tensor(slot_indices, dtype=torch.long, device=self.device)
        if idx.numel() == 0:
            return
        self.count[idx] = 0
        self.rms_est[idx] = self.agc_target
        self.sum0[idx] = 0
        self.sum1[idx] = 0
        self.I_est[idx] = 0
        self.Q_est[idx] = 0
        self.theta_hat[idx] = 0
        self.omega_hat[idx] = self.omega_prior
        self.theta_ref[idx] = 0
        self.ref_captured[idx] = False
        self.raw_theta_running[idx] = 0
        self.prev_raw_phase[idx] = 0
        self.has_prev_raw_phase[idx] = False
        self.raw_theta_snapshot[idx] = 0
        self.snapshot_taken[idx] = False
        for label in self.THRESHOLDS:
            self.crossed[label][idx] = False
            self.cross_step[label][idx] = -1
            self.pred_total[label][idx] = 0

    @torch.inference_mode()
    def update(self, h_last, active_mask):
        h_last = h_last.to(device=self.device, dtype=self.dtype)
        active_mask = active_mask.to(self.device)
        (self.count, self.rms_est, self.sum0, self.sum1,
         self.I_est, self.Q_est, self.theta_hat, self.omega_hat,
         self.theta_ref, self.ref_captured,
         self.raw_theta_running, self.prev_raw_phase, self.has_prev_raw_phase,
         self.raw_theta_snapshot, self.snapshot_taken,
         self.crossed['quarter'], self.cross_step['quarter'], self.pred_total['quarter'],
         self.crossed['half'], self.cross_step['half'], self.pred_total['half'],
         mag_n) = self._kernel(
            h_last, active_mask,
            self.mu, self.v0, self.v1, self.s0, self.s1,
            self.count, self.rms_est, self.sum0, self.sum1,
            self.I_est, self.Q_est, self.theta_hat, self.omega_hat,
            self.theta_ref, self.ref_captured,
            self.raw_theta_running, self.prev_raw_phase, self.has_prev_raw_phase,
            self.raw_theta_snapshot, self.snapshot_taken,
            self.crossed['quarter'], self.cross_step['quarter'], self.pred_total['quarter'],
            self.crossed['half'], self.cross_step['half'], self.pred_total['half'],
            self.mag_floor, self.trust_sharpness, self.Kp, self.Ki,
            self.agc_target, self.agc_alpha, self.warmup, self.dc_warmup,
            self.omega_clamp[0], self.omega_clamp[1],
            self.lockin_alpha, self.omega_boot_window, self.snapshot_step,
            self.boot_shrink, self.omega_relax, self.omega_prior,
            self.THRESHOLDS['quarter'], self.THRESHOLDS['half'],
        )
        self.t += 1
        return self.theta_hat, mag_n

    @torch.inference_mode()
    def update_slots(self, h_rows, slot_indices):
        """Update only scheduled rows instead of projecting every free slot."""
        if not slot_indices:
            return None, None
        idx = torch.as_tensor(
            slot_indices, dtype=torch.long, device=self.device
        )
        h_rows = h_rows.to(device=self.device, dtype=self.dtype)
        active = torch.ones(
            idx.numel(), dtype=torch.bool, device=self.device
        )
        crossed_q = self.crossed["quarter"][idx]
        crossed_h = self.crossed["half"][idx]
        (
            count, rms_est, sum0, sum1,
            I_est, Q_est, theta_hat, omega_hat,
            theta_ref, ref_captured,
            raw_theta_running, prev_raw_phase, has_prev_raw_phase,
            raw_theta_snapshot, snapshot_taken,
            crossed_q, cross_step_q, pred_total_q,
            crossed_h, cross_step_h, pred_total_h,
            mag_n,
        ) = self._kernel(
            h_rows, active,
            self.mu, self.v0, self.v1, self.s0, self.s1,
            self.count[idx], self.rms_est[idx],
            self.sum0[idx], self.sum1[idx],
            self.I_est[idx], self.Q_est[idx],
            self.theta_hat[idx], self.omega_hat[idx],
            self.theta_ref[idx], self.ref_captured[idx],
            self.raw_theta_running[idx], self.prev_raw_phase[idx],
            self.has_prev_raw_phase[idx],
            self.raw_theta_snapshot[idx], self.snapshot_taken[idx],
            crossed_q, self.cross_step["quarter"][idx],
            self.pred_total["quarter"][idx],
            crossed_h, self.cross_step["half"][idx],
            self.pred_total["half"][idx],
            self.mag_floor, self.trust_sharpness, self.Kp, self.Ki,
            self.agc_target, self.agc_alpha, self.warmup, self.dc_warmup,
            self.omega_clamp[0], self.omega_clamp[1],
            self.lockin_alpha, self.omega_boot_window, self.snapshot_step,
            self.boot_shrink, self.omega_relax, self.omega_prior,
            self.THRESHOLDS["quarter"], self.THRESHOLDS["half"],
        )
        self.count[idx] = count
        self.rms_est[idx] = rms_est
        self.sum0[idx] = sum0
        self.sum1[idx] = sum1
        self.I_est[idx] = I_est
        self.Q_est[idx] = Q_est
        self.theta_hat[idx] = theta_hat
        self.omega_hat[idx] = omega_hat
        self.theta_ref[idx] = theta_ref
        self.ref_captured[idx] = ref_captured
        self.raw_theta_running[idx] = raw_theta_running
        self.prev_raw_phase[idx] = prev_raw_phase
        self.has_prev_raw_phase[idx] = has_prev_raw_phase
        self.raw_theta_snapshot[idx] = raw_theta_snapshot
        self.snapshot_taken[idx] = snapshot_taken
        self.crossed["quarter"][idx] = crossed_q
        self.cross_step["quarter"][idx] = cross_step_q
        self.pred_total["quarter"][idx] = pred_total_q
        self.crossed["half"][idx] = crossed_h
        self.cross_step["half"][idx] = cross_step_h
        self.pred_total["half"][idx] = pred_total_h
        self.t += 1
        return theta_hat, mag_n

def predicted_remaining(tracker, slot_idx, calib, default_prior, deadline_multiplier=None):
    """Decays gracefully if uncrossed, returns predicted remaining if crossed."""
    mult = deadline_multiplier or calib.get('half_over_quarter_ratio_mean', 2.0)
    current_t = float(tracker.count[slot_idx].item())
    if bool(tracker.crossed['half'][slot_idx].item()):
        raw = float(tracker.pred_total['half'][slot_idx].item())
        pred = calib['a_half'] * raw + calib['b_half']
        return max(0.0, pred - current_t)
    if bool(tracker.crossed['quarter'][slot_idx].item()):
        t_q = int(tracker.cross_step['quarter'][slot_idx].item())
        deadline = mult * t_q
        if current_t >= deadline:
            raw_q = float(tracker.pred_total['quarter'][slot_idx].item())
            pred_q = calib['a_quarter'] * raw_q + calib['b_quarter']
            pred_q_upper = pred_q + calib.get('quarter_ci_width', 0.0)
            return max(0.0, pred_q_upper - current_t)
        return max(0.0, default_prior - current_t)
    return max(0.0, default_prior - current_t)

# ---------------------------------------------------------------------------
# vLLM internal introspection
# ---------------------------------------------------------------------------
def locate_decoder_layers_engine(engine, layer_idx, gpu_id=0):
    candidates = [
        ("engine.model_executor.driver_worker.model_runner",
         lambda: engine.model_executor.driver_worker.model_runner),
        ("engine.model_executor.driver_worker.worker.model_runner",
         lambda: engine.model_executor.driver_worker.worker.model_runner),
        ("engine.engine_core.model_executor.driver_worker.model_runner",
         lambda: engine.engine_core.model_executor.driver_worker.model_runner),
    ]
    for idx, (desc, fn) in enumerate(candidates):
        try:
            model_runner = fn()
            layer = model_runner.model.model.layers[layer_idx]
            print(f"[GPU {gpu_id}] locate_decoder_layers_engine: candidate #{idx} SUCCEEDED -> {desc}")
            return layer, model_runner, idx, desc
        except AttributeError as e:
            print(f"[GPU {gpu_id}] locate_decoder_layers_engine: candidate #{idx} failed ({desc}): {e}")
            continue
    raise RuntimeError("Could not locate decoder layers via LLMEngine.")


_first_error_printed = {}


def resolve_external_request_id(owner, known_request_ids):
    """Map a vLLM hook owner ID back to the ID passed to add_request.

    vLLM 0.26 stores scheduler state under an internal ID of the form
    ``<external-id>-<8 hex chars>`` even though RequestOutput.request_id uses
    the original external ID.  Only remove that exact suffix shape, and only
    when the resulting prefix is a currently known request.
    """
    if isinstance(owner, bytes):
        owner_id = owner.decode("utf-8", errors="replace")
    else:
        owner_id = str(getattr(owner, "request_id", owner))

    if owner_id in known_request_ids:
        return owner_id

    prefix, separator, suffix = owner_id.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(char in "0123456789abcdefABCDEF" for char in suffix)
        and prefix in known_request_ids
    ):
        return prefix
    return None


def get_row_owners(model_runner, gpu_id=0, verbose=False):
    try:
        # vLLM V1 (including 0.26) keeps the current forward-row ordering in
        # InputBatch. Prefer this over older private req_states layouts.
        input_batch = getattr(model_runner, "input_batch", None)
        if input_batch is not None:
            req_ids = getattr(input_batch, "req_ids", None)
            if req_ids is not None:
                num_reqs = int(
                    getattr(input_batch, "num_reqs", len(req_ids))
                )
                owners = [
                    owner for owner in list(req_ids[:num_reqs])
                    if owner is not None
                ]
                if owners:
                    return owners

            req_id_to_index = getattr(
                input_batch, "req_id_to_index", None
            )
            if isinstance(req_id_to_index, dict) and req_id_to_index:
                return [
                    rid for rid, _ in sorted(
                        req_id_to_index.items(), key=lambda pair: pair[1]
                    )
                ]

        # Compatibility fallback for older or downstream runner layouts.
        req_states = getattr(model_runner, "req_states", None)
        if req_states is None:
            return None
        req_id_to_index = getattr(req_states, "req_id_to_index", None)
        if isinstance(req_id_to_index, dict) and req_id_to_index:
            slot_to_rid = {v: k for k, v in req_id_to_index.items()}
            max_slot = max(slot_to_rid.keys())
            owners = [slot_to_rid.get(i) for i in range(max_slot + 1)]
            owners = [o for o in owners if o is not None]
            return owners
        return None
    except Exception as e:
        if not _first_error_printed.get(gpu_id):
            print(f"[GPU {gpu_id}] get_row_owners: EXCEPTION on first failure:")
            traceback.print_exc()
            _first_error_printed[gpu_id] = True
        return None


# ---------------------------------------------------------------------------
# PROVIDER-STYLE TWO-REPLICA SERVING EXPERIMENT
# ---------------------------------------------------------------------------
# Architecture:
#   incoming request -> immediate router decision -> one of two independent
#   vLLM engines. Each engine owns its own continuous-batching scheduler.
#   Queue discipline stays FCFS unless experimental priority scheduling is
#   explicitly enabled; Helix is always used for replica placement.
#
# The policies differ only in the information used at the routing decision:
#   round_robin   : no load information
#   queue_size    : unfinished requests assigned to each replica
#   helix_work    : measured backlog routing using Helix estimates for active
#                   requests and class priors for queued/unseen requests
#   oracle_work   : the same routing score using fixed trace lengths

HELIX_FORECAST_GRACE_SEC = float(
    os.getenv("HELIX_FORECAST_GRACE_SEC", "0.50")
)
HELIX_FORECAST_HALF_LIFE_SEC = float(
    os.getenv("HELIX_FORECAST_HALF_LIFE_SEC", "2.0")
)
if HELIX_FORECAST_GRACE_SEC < 0 or HELIX_FORECAST_HALF_LIFE_SEC <= 0:
    raise ValueError("Helix forecast grace must be non-negative and half-life positive")
MIN_HELIX_ACTIVE_SNAPSHOT_FRACTION = 0.05
MIN_HELIX_INFORMATIVE_ESTIMATE_FRACTION = 0.01
HELIX_LATE_CONFIDENCE = 1.00
HELIX_EARLY_CONFIDENCE = 0.55
HELIX_PRIOR_CONFIDENCE = 0.00
RESULTS_JSON = "router_local_queue_results.json"
RESULTS_CSV = "router_local_queue_results.csv"
REQUEST_RESULTS_CSV = "router_local_queue_requests.csv"
EXPERIMENT_CONFIG_JSON = "router_experiment_config.json"
AGGREGATE_RESULTS_JSON = "router_seed_aggregates.json"
AGGREGATE_RESULTS_CSV = "router_seed_aggregates.csv"

POLICY_LABELS = {
    "round_robin": "Round robin",
    "queue_size": "Queue size",
    "helix_work": "Helix",
    "oracle_work": "Oracle",
}


def _diag_write(hook_diagnostics, gpu_id, calls, skips, true_errors, slot_exhaustions):
    base = gpu_id * 4
    hook_diagnostics[base] = int(calls)
    hook_diagnostics[base + 1] = int(skips)
    hook_diagnostics[base + 2] = int(true_errors)
    hook_diagnostics[base + 3] = int(slot_exhaustions)


def locate_vllm_priority_schedulers(engine):
    """Locate in-process vLLM scheduler objects across V0/V1 layouts."""
    schedulers = []
    seen = set()
    frontier = [(engine, "engine")]
    child_names = (
        "scheduler", "schedulers", "engine_core", "engine_core_client",
        "core", "engine", "_engine", "_engine_core",
    )
    for _ in range(6):
        next_frontier = []
        for obj, path in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "waiting") and hasattr(obj, "running"):
                schedulers.append((obj, path))
            for name in child_names:
                try:
                    child = getattr(obj, name)
                except Exception:
                    continue
                if isinstance(child, (list, tuple)):
                    next_frontier.extend(
                        (value, f"{path}.{name}[{index}]")
                        for index, value in enumerate(child)
                    )
                else:
                    next_frontier.append((child, f"{path}.{name}"))
        frontier = next_frontier
    return schedulers


def read_vllm_max_num_seqs(engine):
    """Read the effective scheduler capacity chosen by this vLLM engine."""
    candidates = (
        lambda: engine.scheduler_config.max_num_seqs,
        lambda: engine.vllm_config.scheduler_config.max_num_seqs,
        lambda: engine.engine_core.scheduler_config.max_num_seqs,
    )
    for candidate in candidates:
        try:
            value = int(candidate())
        except (AttributeError, TypeError, ValueError):
            continue
        if value > 0:
            return value
    raise RuntimeError(
        "Could not read vLLM's effective max_num_seqs from the engine."
    )


def update_vllm_request_priorities(schedulers, priorities):
    """Mutate live priorities and restore priority-queue heap order."""
    import heapq

    updated = set()
    discovered = set()
    for scheduler, _ in schedulers:
        containers = [
            getattr(scheduler, "running", ()),
            getattr(scheduler, "waiting", ()),
        ]
        for container in containers:
            try:
                requests = list(container)
            except TypeError:
                requests = list(getattr(container, "_heap", ()))
            for request in requests:
                request_id = str(getattr(request, "request_id", ""))
                if not request_id:
                    continue
                external_id = (
                    resolve_external_request_id(request_id, priorities)
                    or request_id
                )
                discovered.add(external_id)
                if external_id not in priorities:
                    continue
                new_priority = int(priorities[external_id])
                if int(getattr(request, "priority", new_priority)) != new_priority:
                    setattr(request, "priority", new_priority)
                    updated.add(external_id)
        waiting = getattr(scheduler, "waiting", None)
        heap = getattr(waiting, "_heap", None)
        if isinstance(heap, list):
            heapq.heapify(heap)
    return updated, discovered


def read_vllm_request_preemptions(schedulers, known_request_ids):
    """Read live V1/V0 request preemption counters when exposed."""
    counters = {}
    observed = set()
    for scheduler, _ in schedulers:
        containers = [
            getattr(scheduler, "running", ()),
            getattr(scheduler, "waiting", ()),
        ]
        for container in containers:
            try:
                requests = list(container)
            except TypeError:
                requests = list(getattr(container, "_heap", ()))
            for request in requests:
                request_id = str(getattr(request, "request_id", ""))
                if not request_id:
                    continue
                external_id = (
                    resolve_external_request_id(
                        request_id, known_request_ids
                    )
                    or request_id
                )
                value = getattr(request, "num_preemptions", None)
                if value is None:
                    continue
                observed.add(external_id)
                counters[external_id] = max(
                    counters.get(external_id, 0), int(value)
                )
    return counters, observed


def normal_scheduler_priority(
    arrival_rank,
    cohort_size,
    predicted_total_tokens,
    bucket_tokens=PRIORITY_LENGTH_BUCKET_TOKENS,
):
    """Stable cohort/SJF priority; it must not change every decode token."""
    cohort = int(arrival_rank) // max(1, int(cohort_size))
    bucket = max(
        0,
        int(round(float(predicted_total_tokens) / max(1, bucket_tokens))),
    )
    return 10000 + cohort * 2048 + min(2047, bucket), bucket


def request_priority_state(
    normal_priority,
    generated_tokens,
    arrival_time,
    last_token_time,
    now,
    ttft_deadline_sec,
    itl_rescue_sec,
    normal_state="normal",
):
    if (
        int(generated_tokens) > 0
        and last_token_time is not None
        and now - float(last_token_time) >= float(itl_rescue_sec)
    ):
        return 0, "itl_rescue"
    if (
        int(generated_tokens) == 0
        and now - float(arrival_time) >= float(ttft_deadline_sec)
    ):
        return 1, "ttft_urgent"
    return int(normal_priority), str(normal_state)


def _write_fixed_slots(shared_array, gpu_id, values, width, fill_value):
    start = gpu_id * width
    for i in range(width):
        shared_array[start + i] = values[i] if i < len(values) else fill_value


def predicted_remaining_with_confidence(tracker, slot_idx, calib, default_total):
    """Return remaining tokens, confidence and checkpoint stage.

    The late pi checkpoint is trusted most. Before a reliable crossing, the
    request-specific historical prior is used rather than inventing precision.
    """
    current_t = float(tracker.count[slot_idx].item())
    if bool(tracker.crossed['half'][slot_idx].item()):
        raw = float(tracker.pred_total['half'][slot_idx].item())
        pred_total = calib['a_half'] * raw + calib['b_half']
        return max(0.0, pred_total - current_t), HELIX_LATE_CONFIDENCE, "late"

    if bool(tracker.crossed['quarter'][slot_idx].item()):
        raw = float(tracker.pred_total['quarter'][slot_idx].item())
        pred_total = calib['a_quarter'] * raw + calib['b_quarter']
        # Return the raw calibrated estimate plus its confidence. The consumer
        # performs shrinkage exactly once; the old code blended here and again
        # in the parent router, reducing early Helix weight from 0.55 to 0.30.
        return (
            max(0.0, pred_total - current_t),
            HELIX_EARLY_CONFIDENCE,
            "early",
        )

    return max(0.0, float(default_total) - current_t), HELIX_PRIOR_CONFIDENCE, "prior"


def _create_shared_state(n_requests):
    # Storage follows the workload, not an assumed vLLM sequence limit. Each
    # worker publishes its engine-discovered max_num_seqs separately.
    slot_width = max(1, int(n_requests))
    table_width = slot_width + 1
    return {
        "slot_width": slot_width,
        "table_width": table_width,
        "locks": [mp.Lock(), mp.Lock()],
        "engine_capacity": mp.Array('i', [0, 0], lock=False),
        "scheduler_running_count": mp.Array('i', [0, 0], lock=False),
        "scheduler_waiting_count": mp.Array('i', [0, 0], lock=False),
        "local_unfinished": mp.Array('i', [0, 0], lock=False),
        "active_decode_count": mp.Array('i', [0, 0], lock=False),
        "active_ids": mp.Array(
            'i', [-1] * (2 * slot_width), lock=False
        ),
        "active_helix_remaining": mp.Array(
            'd', [float('inf')] * (2 * slot_width), lock=False
        ),
        "active_helix_confidence": mp.Array(
            'd', [0.0] * (2 * slot_width), lock=False
        ),
        "generated_all": mp.Array('i', [0] * n_requests, lock=False),
        "forecast_timestamp": mp.Array('d', [0.0, 0.0], lock=False),
        "decode_step_table": mp.Array(
            'd', [0.0] * (2 * table_width), lock=False
        ),
        "decode_step_samples": mp.Array(
            'i', [0] * (2 * table_width), lock=False
        ),
        "prefill_sec_per_token": mp.Array(
            'd', [DEFAULT_PREFILL_SEC_PER_TOKEN] * 2, lock=False
        ),
        "engine_busy_seconds": mp.Array('d', [0.0, 0.0], lock=False),
        "decode_slot_seconds": mp.Array('d', [0.0, 0.0], lock=False),
        "hook_diag": mp.Array('i', [0] * 8, lock=False),
    }


def gpu_worker_local_queue(
    gpu_id,
    model_id,
    cache,
    calib_params,
    historical_mean_output,
    enable_helix_tracker,
    enable_priority_scheduler,
    priority_ttft_deadline_sec,
    priority_itl_rescue_sec,
    priority_cohort_size,
    priority_knowledge,
    input_queue,
    output_queue,
    ready_queue,
    state_lock,
    shared_engine_capacity,
    shared_scheduler_running_count,
    shared_scheduler_waiting_count,
    shared_slot_width,
    shared_table_width,
    shared_local_unfinished,
    shared_active_decode_count,
    shared_active_ids,
    shared_active_helix_remaining,
    shared_active_helix_confidence,
    shared_generated_all,
    shared_forecast_timestamp,
    shared_decode_step_table,
    shared_decode_step_samples,
    shared_prefill_sec_per_token,
    shared_engine_busy_seconds,
    shared_decode_slot_seconds,
    hook_diagnostics,
):
    """One vLLM replica with FCFS or dynamic priority scheduling."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu{gpu_id}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/inductor_cache_gpu{gpu_id}"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch as _torch
    from multiprocessing.util import Finalize
    global torch
    torch = _torch
    from vllm import LLMEngine, EngineArgs, SamplingParams
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GPU {gpu_id}] Loading independent vLLM replica ({model_id})...")
    engine_kwargs = dict(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    if MAX_CONCURRENT_SEQS_OVERRIDE is not None:
        engine_kwargs["max_num_seqs"] = int(MAX_CONCURRENT_SEQS_OVERRIDE)
    if enable_priority_scheduler:
        engine_kwargs["scheduling_policy"] = "priority"
    try:
        engine_args = EngineArgs(**engine_kwargs)
    except TypeError as exc:
        if enable_priority_scheduler:
            ready_queue.put({
                "gpu_id": gpu_id,
                "status": "error",
                "error": (
                    "Installed vLLM does not accept scheduling_policy='priority': "
                    f"{exc}"
                ),
            })
        raise
    engine = LLMEngine.from_engine_args(engine_args)
    engine_capacity = read_vllm_max_num_seqs(engine)
    tracking_capacity = max(1, min(engine_capacity, int(shared_slot_width)))
    shared_engine_capacity[gpu_id] = int(engine_capacity)
    print(
        f"[GPU {gpu_id}] vLLM effective max_num_seqs={engine_capacity}; "
        f"tracking up to {tracking_capacity} requests for this workload."
    )
    engine_schedulers = locate_vllm_priority_schedulers(engine)
    priority_schedulers = engine_schedulers if enable_priority_scheduler else []
    if enable_priority_scheduler and not priority_schedulers:
        ready_queue.put({
            "gpu_id": gpu_id,
            "status": "error",
            "error": "In-process vLLM priority scheduler could not be located.",
        })
        raise RuntimeError(
            "Priority policy requested, but the in-process vLLM scheduler "
            "could not be located. Use a vLLM release exposing V1/V0 priority "
            "scheduling through LLMEngine, or run queue_size/round_robin."
        )
    if priority_schedulers:
        print(
            f"[GPU {gpu_id}] Priority schedulers: "
            + ", ".join(path for _, path in priority_schedulers)
        )

    def live_scheduler_counts():
        running = 0
        waiting = 0
        for scheduler, _ in engine_schedulers:
            for field in ("running", "waiting"):
                container = getattr(scheduler, field, ())
                try:
                    count = len(container)
                except TypeError:
                    try:
                        count = len(list(container))
                    except TypeError:
                        count = len(getattr(container, "_heap", ()))
                if field == "running":
                    running += int(count)
                else:
                    waiting += int(count)
        return running, waiting

    def cleanup_process_group():
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.destroy_process_group()
        except Exception as exc:
            print(f"[GPU {gpu_id}] NCCL cleanup warning: {exc}")

    # multiprocessing uses os._exit, so its own finalizer is more reliable
    # than atexit when an uncaught worker exception bypasses normal shutdown.
    process_group_finalizer = Finalize(
        None, cleanup_process_group, exitpriority=10
    )

    layer_module = None
    model_runner = None
    tracker = None
    if enable_helix_tracker:
        layer_module, model_runner, _, root_desc = locate_decoder_layers_engine(
            engine, TARGET_LAYER, gpu_id=gpu_id
        )
        print(f"[GPU {gpu_id}] Model loaded. Decoder path: {root_desc}")
        _get_compiled_kernel()
        tracker = BatchedCausalPLLTracker(
            cache,
            batch_size=tracking_capacity,
            device=device,
            dtype=torch.float32,
            **TRACKER_KWARGS,
        )
    else:
        print(f"[GPU {gpu_id}] Model loaded. Helix detector disabled.")

    req_meta = {}                 # accepted and unfinished; includes local queue
    output_tokens = {}            # cumulative output length per request
    token_times = {}
    priority_update_counts = {}
    tracker_slot_by_rid = {}
    free_tracker_slots = list(range(tracking_capacity))
    current_active_rids = []       # current target-layer decode rows only
    published_active_count = 0

    total_hook_calls = 0
    skipped_hook_calls = 0
    true_error_calls = 0
    tracker_slot_exhaustions = 0
    step_had_prefill = False
    step_decode_rows = 0

    def output_prior(rid):
        meta = req_meta.get(rid, {})
        return float(meta.get(
            "scheduler_output_tokens",
            meta.get("estimated_output_tokens", historical_mean_output),
        ))

    def scheduler_total_forecast(rid):
        generated = int(output_tokens.get(rid, 0))
        prior_total = output_prior(rid)
        if priority_knowledge != "helix" or not enable_helix_tracker:
            return prior_total, (
                "oracle" if priority_knowledge == "oracle" else "prior"
            )
        slot = tracker_slot_by_rid.get(rid)
        if slot is None:
            return prior_total, "prior"
        remaining, confidence, stage = predicted_remaining_with_confidence(
            tracker, slot, calib_params, output_prior(rid)
        )
        prior_remaining = max(0.0, prior_total - generated)
        blended_remaining = (
            float(confidence) * float(remaining)
            + (1.0 - float(confidence)) * prior_remaining
        )
        return max(1.0, generated + blended_remaining), stage

    def refresh_normal_priority(rid, force=False):
        meta = req_meta[rid]
        predicted_total, stage = scheduler_total_forecast(rid)
        stage_rank = {
            "prior": 0, "early": 1, "late": 2, "oracle": 3
        }.get(stage, 0)
        old_stage_rank = int(meta.get("priority_length_stage_rank", -1))
        candidate, bucket = normal_scheduler_priority(
            arrival_rank=meta["arrival_rank"],
            cohort_size=priority_cohort_size,
            predicted_total_tokens=predicted_total,
        )
        old_bucket = meta.get("priority_length_bucket")
        material_change = (
            old_bucket is None
            or abs(int(bucket) - int(old_bucket))
            >= PRIORITY_LENGTH_HYSTERESIS_BUCKETS
        )
        if force or (stage_rank > old_stage_rank and material_change):
            meta["normal_scheduler_priority"] = int(candidate)
            meta["priority_length_bucket"] = int(bucket)
            meta["priority_predicted_total_tokens"] = float(predicted_total)
        if stage_rank > old_stage_rank:
            meta["priority_length_stage"] = stage
            meta["priority_length_stage_rank"] = stage_rank
        return int(meta["normal_scheduler_priority"])

    def scheduler_priority(rid, now):
        meta = req_meta[rid]
        generated = int(output_tokens.get(rid, 0))
        times = token_times.get(rid, ())
        normal_priority = refresh_normal_priority(
            rid, force="normal_scheduler_priority" not in meta
        )
        # Match the simulator's guarded discipline: normal decode work remains
        # resident, while requests without a first token wait FCFS until their
        # calibrated TTFT deadline. This avoids repeatedly interrupting a batch
        # just because a newly arrived request has a shorter predicted length.
        normal_state = "normal"
        if generated == 0:
            normal_priority = (
                PRIORITY_UNSTARTED_BASE + int(meta["arrival_rank"])
            )
            normal_state = "unstarted"
        priority, state = request_priority_state(
            normal_priority=normal_priority,
            generated_tokens=generated,
            arrival_time=meta["arrival_time"],
            last_token_time=times[-1] if times else None,
            now=now,
            ttft_deadline_sec=priority_ttft_deadline_sec,
            itl_rescue_sec=priority_itl_rescue_sec,
            normal_state=normal_state,
        )
        previous_state = meta.get("scheduler_priority_state")
        if previous_state is not None and previous_state != state:
            meta["scheduler_priority_transitions"] = (
                int(meta.get("scheduler_priority_transitions", 0)) + 1
            )
        meta["scheduler_priority_state"] = state
        return int(priority)

    def refresh_scheduler_priorities(now):
        if not enable_priority_scheduler or not req_meta:
            return
        priorities = {
            rid: scheduler_priority(rid, now) for rid in req_meta
        }
        updated, discovered = update_vllm_request_priorities(
            priority_schedulers, priorities
        )
        preemptions, preemption_observed = read_vllm_request_preemptions(
            priority_schedulers, req_meta
        )
        for rid, count in preemptions.items():
            if rid in req_meta:
                req_meta[rid]["scheduler_preemptions_live"] = max(
                    int(req_meta[rid].get("scheduler_preemptions_live", 0)),
                    int(count),
                )
                req_meta[rid]["scheduler_preemptions_observed"] = True
        for rid in updated:
            priority_update_counts[rid] = (
                priority_update_counts.get(rid, 0) + 1
            )
            if rid in req_meta:
                req_meta[rid]["current_scheduler_priority"] = priorities[rid]
        for rid in req_meta:
            if rid not in preemption_observed:
                req_meta[rid].setdefault(
                    "scheduler_preemptions_observed", False
                )
        missing = set(req_meta) - discovered
        if missing and engine.has_unfinished_requests():
            # Requests can briefly be between scheduler containers while an
            # engine step is being finalized; persistent misses are caught by
            # the per-result priority-update coverage metrics.
            for rid in missing:
                req_meta[rid].setdefault("priority_lookup_misses", 0)
                req_meta[rid]["priority_lookup_misses"] += 1

    def publish_state(now=None):
        nonlocal published_active_count
        now = time.time() if now is None else now
        scheduler_running, scheduler_waiting = live_scheduler_counts()
        active_ids = [rid for rid in current_active_rids if rid in req_meta]
        remaining = []
        confidence = []
        numeric_ids = []

        for rid in active_ids[:shared_slot_width]:
            slot = tracker_slot_by_rid.get(rid)
            if slot is None:
                rem = max(0.0, output_prior(rid) - output_tokens.get(rid, 0))
                conf = 0.0
            else:
                rem, conf, _ = predicted_remaining_with_confidence(
                    tracker, slot, calib_params, output_prior(rid)
                )
            numeric_ids.append(int(rid))
            remaining.append(float(rem))
            confidence.append(float(conf))

        with state_lock:
            shared_local_unfinished[gpu_id] = len(req_meta)
            shared_scheduler_running_count[gpu_id] = scheduler_running
            shared_scheduler_waiting_count[gpu_id] = scheduler_waiting
            shared_active_decode_count[gpu_id] = len(numeric_ids)
            write_count = max(published_active_count, len(numeric_ids))
            start = gpu_id * shared_slot_width
            for index in range(write_count):
                shared_active_ids[start + index] = (
                    numeric_ids[index] if index < len(numeric_ids) else -1
                )
                shared_active_helix_remaining[start + index] = (
                    remaining[index]
                    if index < len(remaining) else float("inf")
                )
                shared_active_helix_confidence[start + index] = (
                    confidence[index] if index < len(confidence) else 0.0
                )
            published_active_count = len(numeric_ids)
            shared_forecast_timestamp[gpu_id] = now

    def hook_fn(module, inp, output):
        nonlocal total_hook_calls, skipped_hook_calls, true_error_calls
        nonlocal tracker_slot_exhaustions, current_active_rids
        nonlocal step_had_prefill, step_decode_rows

        h = output[0] if isinstance(output, tuple) else output
        h = h.detach()
        total_hook_calls += 1
        owners = get_row_owners(model_runner, gpu_id=gpu_id, verbose=False)

        if owners is None:
            skipped_hook_calls += 1
            step_had_prefill = True
            current_active_rids = []
            _diag_write(
                hook_diagnostics, gpu_id, total_hook_calls,
                skipped_hook_calls, true_error_calls, tracker_slot_exhaustions,
            )
            return output

        if len(owners) != h.shape[0]:
            skipped_hook_calls += 1
            current_active_rids = []
            # More hidden rows than decode owners generally means prefill or a
            # packed mixed step. Do not feed mismatched rows into the tracker.
            if h.shape[0] < len(owners):
                true_error_calls += 1
            else:
                step_had_prefill = True
            _diag_write(
                hook_diagnostics, gpu_id, total_hook_calls,
                skipped_hook_calls, true_error_calls, tracker_slot_exhaustions,
            )
            return output

        last_row_for_req = {}
        for row_idx, owner in enumerate(owners):
            rid = resolve_external_request_id(owner, req_meta)
            if rid is not None:
                last_row_for_req[rid] = row_idx

        if not last_row_for_req and req_meta:
            skipped_hook_calls += 1
            true_error_calls += 1
            current_active_rids = []
            _diag_write(
                hook_diagnostics, gpu_id, total_hook_calls,
                skipped_hook_calls, true_error_calls, tracker_slot_exhaustions,
            )
            return output

        current_active_rids = list(last_row_for_req.keys())
        step_decode_rows = len(current_active_rids)

        selected_rows = []
        selected_slots = []
        for rid, row_idx in last_row_for_req.items():
            if rid not in tracker_slot_by_rid:
                if not free_tracker_slots:
                    tracker_slot_exhaustions += 1
                    continue
                slot = free_tracker_slots.pop()
                tracker.reset_slots([slot])
                tracker_slot_by_rid[rid] = slot
            slot = tracker_slot_by_rid[rid]
            selected_rows.append(row_idx)
            selected_slots.append(slot)

        if selected_slots:
            tracker.update_slots(
                h[selected_rows].detach().to(torch.float32),
                selected_slots,
            )
        publish_state()
        _diag_write(
            hook_diagnostics, gpu_id, total_hook_calls,
            skipped_hook_calls, true_error_calls, tracker_slot_exhaustions,
        )
        return output

    # Warm the actual online engine before the parent starts the arrival clock.
    # Fireworks returns 503 while a scaled-to-zero replica starts; it does not
    # silently include model initialization in request TTFT. The benchmark
    # therefore measures only already-ready replicas.
    warmup_params = SamplingParams(
        max_tokens=ENGINE_WARMUP_OUTPUT_TOKENS,
        temperature=0.0,
    )
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the word ready."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    warmup_requests = min(
        engine_capacity,
        max(8, int(math.ceil(2.0 * math.sqrt(engine_capacity)))),
    )
    for i in range(warmup_requests):
        warmup_kwargs = {"priority": 10000} if enable_priority_scheduler else {}
        engine.add_request(
            f"warmup-{gpu_id}-{i}",
            warmup_text,
            warmup_params,
            **warmup_kwargs,
        )
    while engine.has_unfinished_requests():
        engine.step()

    if enable_helix_tracker:
        layer_module.register_forward_hook(hook_fn)
    publish_state()
    ready_queue.put({
        "gpu_id": gpu_id,
        "status": "ready",
        "ready_time": time.time(),
        "engine_capacity": int(engine_capacity),
    })
    print(f"[GPU {gpu_id}] Warm and accepting traffic.")

    def sampling_params_for(req):
        gp = dict(req.get("generation_params") or {})
        kwargs = {
            "max_tokens": int(gp.get("max_tokens", MAX_NEW_TOKENS)),
            "temperature": float(gp.get("temperature", 0.0)),
        }
        for name in ("top_p", "top_k", "min_p", "min_tokens", "stop"):
            if name in gp and gp[name] is not None:
                kwargs[name] = gp[name]
        return SamplingParams(**kwargs)

    stop_requested = False
    while True:
        # Immediate dispatch from the global router. vLLM owns local admission;
        # work-aware runs dynamically refresh its priority queue.
        while True:
            try:
                req = input_queue.get_nowait()
            except queue.Empty:
                break
            if req is None:
                stop_requested = True
                break

            rid = str(req["req_id"])
            req_meta[rid] = dict(req)
            output_tokens[rid] = 0
            shared_generated_all[int(rid)] = 0
            messages = []
            if req.get("system_prompt"):
                messages.append({"role": "system", "content": req["system_prompt"]})
            messages.append({"role": "user", "content": req["prompt"]})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            add_kwargs = {}
            if enable_priority_scheduler:
                initial_priority = scheduler_priority(rid, time.time())
                req_meta[rid]["initial_scheduler_priority"] = initial_priority
                req_meta[rid]["current_scheduler_priority"] = initial_priority
                add_kwargs["priority"] = initial_priority
            engine.add_request(
                rid,
                formatted,
                sampling_params_for(req),
                **add_kwargs,
            )
        publish_state()
        refresh_scheduler_priorities(time.time())

        if engine.has_unfinished_requests():
            step_had_prefill = False
            step_decode_rows = 0
            step_started = time.time()
            step_outputs = engine.step()
            step_time = time.time()
            step_elapsed = max(1e-6, step_time - step_started)

            shared_engine_busy_seconds[gpu_id] += step_elapsed

            newly_started_prompt_tokens = 0
            finished_records = []
            observed_decode_rows = 0
            decoded_rids = []
            for out in step_outputs:
                rid = str(out.request_id)
                cumulative = len(out.outputs[0].token_ids) if out.outputs else 0
                previous = int(output_tokens.get(rid, 0))
                delta = max(0, cumulative - previous)
                output_tokens[rid] = cumulative
                shared_generated_all[int(rid)] = cumulative

                if previous == 0 and cumulative > 0 and rid in req_meta:
                    newly_started_prompt_tokens += max(
                        1, int(req_meta[rid].get("prompt_len", 1))
                    )
                if delta > 0:
                    observed_decode_rows += 1
                    decoded_rids.append(rid)
                    # With speculative decoding disabled this is normally one
                    # output token per engine step. One timestamp per step gives
                    # the true user-visible inter-iteration interval.
                    token_times.setdefault(rid, []).append(step_time)
                if out.finished:
                    finished_records.append((rid, out, req_meta.get(rid)))

            # RequestOutput deltas provide an engine-version-independent active
            # set for policies that do not install the Helix hidden-state hook.
            # They also repair hook ownership gaps during mixed prefill/decode
            # iterations.
            current_active_rids = list(dict.fromkeys(decoded_rids))

            # Some vLLM builds execute the target-layer hook asynchronously,
            # so the hook-visible row count can be zero even though outputs
            # advanced. Output deltas are a stable fallback for occupancy.
            effective_decode_rows = max(step_decode_rows, observed_decode_rows)
            # First output tokens prove that this iteration contained prefill,
            # even for policies that deliberately do not install the Helix
            # hook. Previously Oracle mislabeled those expensive mixed steps as
            # pure decode while Helix did not, corrupting their timing tables.
            effective_step_had_prefill = (
                step_had_prefill or newly_started_prompt_tokens > 0
            )
            shared_decode_slot_seconds[gpu_id] += (
                step_elapsed * effective_decode_rows
            )
            table_idx = (
                gpu_id * shared_table_width
                + min(shared_table_width - 1, max(0, effective_decode_rows))
            )
            if effective_decode_rows > 0 and not effective_step_had_prefill:
                prev = float(shared_decode_step_table[table_idx])
                shared_decode_step_table[table_idx] = (
                    step_elapsed if prev <= 0 else
                    SERVICE_TIME_EWMA_ALPHA * step_elapsed
                    + (1.0 - SERVICE_TIME_EWMA_ALPHA) * prev
                )
                shared_decode_step_samples[table_idx] += 1

            if effective_step_had_prefill and newly_started_prompt_tokens > 0:
                decode_baseline = 0.0
                if effective_decode_rows > 0:
                    decode_baseline = float(shared_decode_step_table[table_idx])
                observed = max(0.0, step_elapsed - decode_baseline)
                observed_spt = observed / newly_started_prompt_tokens
                if observed_spt > 0:
                    prev = float(shared_prefill_sec_per_token[gpu_id])
                    shared_prefill_sec_per_token[gpu_id] = (
                        observed_spt if prev <= 0 else
                        PREFILL_EWMA_ALPHA * observed_spt
                        + (1.0 - PREFILL_EWMA_ALPHA) * prev
                    )

            for rid, out, meta in finished_records:
                req_meta.pop(rid, None)
                output_tokens.pop(rid, None)
                if rid in current_active_rids:
                    current_active_rids = [x for x in current_active_rids if x != rid]
                slot = tracker_slot_by_rid.pop(rid, None)
                if slot is not None:
                    tracker.reset_slots([slot])
                    free_tracker_slots.append(slot)

                times = token_times.pop(rid, [])
                if meta is not None:
                    ttft = times[0] - meta["arrival_time"] if times else None
                    engine_metrics = getattr(out, "metrics", None)
                    engine_queue_time = (
                        getattr(engine_metrics, "time_in_queue", None)
                        if engine_metrics is not None else None
                    )
                    if engine_queue_time is not None:
                        engine_queue_time = max(
                            0.0, float(engine_queue_time)
                        )
                    first_scheduled_time = (
                        getattr(engine_metrics, "first_scheduled_time", None)
                        if engine_metrics is not None else None
                    )
                    scheduler_queue_time = None
                    if first_scheduled_time is not None:
                        scheduler_queue_time = max(
                            0.0,
                            float(first_scheduled_time)
                            - float(meta["arrival_time"]),
                        )
                    scheduler_preemptions = None
                    if engine_metrics is not None:
                        for attr in (
                            "num_preemptions",
                            "cumulative_preemption_count",
                            "preemption_count",
                        ):
                            value = getattr(engine_metrics, attr, None)
                            if value is not None:
                                scheduler_preemptions = int(value)
                                break
                    if meta.get("scheduler_preemptions_observed", False):
                        scheduler_preemptions = int(
                            meta.get("scheduler_preemptions_live", 0)
                        )
                    itl_gaps = [
                        t2 - t1 for t1, t2 in zip(times[:-1], times[1:])
                    ] if len(times) > 1 else []
                    output_queue.put({
                        "req_id": int(meta["req_id"]),
                        "gpu_id": gpu_id,
                        "arrival_time": float(meta["arrival_time"]),
                        "scheduled_arrival_time": float(
                            meta.get("scheduled_arrival_time", meta["arrival_time"])
                        ),
                        "finish_time": time.time(),
                        "n_tokens": len(out.outputs[0].token_ids),
                        "ttft": ttft,
                        "engine_queue_time_sec": engine_queue_time,
                        "scheduler_queue_time_sec": scheduler_queue_time,
                        "scheduler_preemptions": scheduler_preemptions,
                        "scheduler_priority_initial": meta.get(
                            "initial_scheduler_priority"
                        ),
                        "scheduler_priority_final": meta.get(
                            "current_scheduler_priority"
                        ),
                        "scheduler_priority_updates": int(
                            priority_update_counts.pop(rid, 0)
                        ),
                        "scheduler_priority_transitions": int(
                            meta.get("scheduler_priority_transitions", 0)
                        ),
                        "scheduler_priority_state_final": meta.get(
                            "scheduler_priority_state"
                        ),
                        "priority_length_stage": meta.get(
                            "priority_length_stage"
                        ),
                        "priority_length_bucket": meta.get(
                            "priority_length_bucket"
                        ),
                        "priority_predicted_total_tokens": meta.get(
                            "priority_predicted_total_tokens"
                        ),
                        "priority_lookup_misses": int(
                            meta.get("priority_lookup_misses", 0)
                        ),
                        "itl_gaps": itl_gaps,
                        "prompt_len": int(meta.get("prompt_len", 0)),
                        "estimated_output_tokens": float(
                            meta.get("estimated_output_tokens", 0.0)
                        ),
                        "output_prior_source": meta.get(
                            "output_prior_source", "__global__"
                        ),
                        "traffic_class": meta.get("traffic_class", "unknown"),
                        "measure": bool(meta.get("measure", True)),
                        "predicted_queue_wait_sec": float(
                            meta.get("predicted_queue_wait_sec", 0.0)
                        ),
                        "routing_reason": meta.get("routing_reason"),
                        "routing_queue_choice": meta.get(
                            "routing_queue_choice"
                        ),
                        "routing_work_choice": meta.get(
                            "routing_work_choice"
                        ),
                        "routing_overrode_queue": meta.get(
                            "routing_overrode_queue"
                        ),
                        "routing_predicted_gain_sec": meta.get(
                            "routing_predicted_gain_sec"
                        ),
                        "routing_predicted_gain_tokens": meta.get(
                            "routing_predicted_gain_tokens"
                        ),
                        "routing_score_gpu0_sec": meta.get(
                            "routing_score_gpu0_sec"
                        ),
                        "routing_score_gpu1_sec": meta.get(
                            "routing_score_gpu1_sec"
                        ),
                        "routing_gpu0_active_work_tokens": meta.get(
                            "routing_gpu0_active_work_tokens"
                        ),
                        "routing_gpu1_active_work_tokens": meta.get(
                            "routing_gpu1_active_work_tokens"
                        ),
                        "routing_gpu0_total_work_tokens": meta.get(
                            "routing_gpu0_total_work_tokens"
                        ),
                        "routing_gpu1_total_work_tokens": meta.get(
                            "routing_gpu1_total_work_tokens"
                        ),
                    })

            publish_state(step_time)

        elif stop_requested:
            break
        else:
            current_active_rids = []
            publish_state()
            time.sleep(0.001)

    print(
        f"[GPU {gpu_id}] shutdown: hook skip={skipped_hook_calls}/{total_hook_calls}, "
        f"true errors={true_error_calls}, tracker slot exhaustions={tracker_slot_exhaustions}"
    )
    # vLLM initializes a one-rank NCCL process group in each replica process.
    # Destroy it explicitly because this benchmark repeatedly starts fresh
    # replicas for policy isolation.
    process_group_finalizer()


# ---------------------------------------------------------------------------
# SNAPSHOTS AND LOCAL-QUEUE COMPLETION MODEL
# ---------------------------------------------------------------------------

def _read_fixed(shared_array, gpu_id, count, width):
    start = gpu_id * width
    return [shared_array[start + i] for i in range(min(count, width))]


def snapshot_worker_local(state, gpu_id):
    slot_width = int(state["slot_width"])
    table_width = int(state["table_width"])
    with state["locks"][gpu_id]:
        active_count = int(state["active_decode_count"][gpu_id])
        active_ids = [
            int(x) for x in _read_fixed(
                state["active_ids"], gpu_id, active_count, slot_width
            )
            if int(x) >= 0
        ]
        helix_remaining = [
            float(x) for x in _read_fixed(
                state["active_helix_remaining"],
                gpu_id,
                active_count,
                slot_width,
            )
        ]
        helix_confidence = [
            float(x) for x in _read_fixed(
                state["active_helix_confidence"],
                gpu_id,
                active_count,
                slot_width,
            )
        ]
        table_start = gpu_id * table_width
        step_table = [
            float(state["decode_step_table"][table_start + b])
            for b in range(table_width)
        ]
        step_samples = [
            int(state["decode_step_samples"][table_start + b])
            for b in range(table_width)
        ]
        return {
            "engine_capacity": max(
                1, int(state["engine_capacity"][gpu_id])
            ),
            "scheduler_running_count": int(
                state["scheduler_running_count"][gpu_id]
            ),
            "scheduler_waiting_count": int(
                state["scheduler_waiting_count"][gpu_id]
            ),
            "local_unfinished": int(state["local_unfinished"][gpu_id]),
            "active_ids": active_ids,
            "helix_by_id": {
                rid: (rem, conf)
                for rid, rem, conf in zip(
                    active_ids, helix_remaining, helix_confidence
                )
            },
            "forecast_age": max(
                0.0, time.time() - float(state["forecast_timestamp"][gpu_id])
            ),
            "step_table": step_table,
            "step_samples": step_samples,
            "prefill_spt": max(
                0.0, float(state["prefill_sec_per_token"][gpu_id])
            ) or DEFAULT_PREFILL_SEC_PER_TOKEN,
        }


def decode_step_seconds(batch_size, step_table, engine_capacity=None):
    """Decode iteration duration from measured per-batch buckets."""
    table_capacity = max(1, len(step_table) - 1)
    effective_capacity = min(
        table_capacity,
        int(engine_capacity) if engine_capacity is not None else table_capacity,
    )
    batch_size = max(1, min(effective_capacity, int(batch_size)))
    direct = float(step_table[batch_size]) if batch_size < len(step_table) else 0.0
    if direct > 0:
        return direct

    measured = [(b, float(v)) for b, v in enumerate(step_table) if b > 0 and float(v) > 0]
    if not measured:
        return DEFAULT_DECODE_STEP_SEC
    nearest_b, nearest_v = min(measured, key=lambda pair: abs(pair[0] - batch_size))
    ratio = batch_size / max(1, nearest_b)
    return max(1e-5, nearest_v * (0.75 + 0.25 * ratio))


def estimate_new_output_tokens(mean_length, std_length, generation_params=None):
    generation_params = generation_params or {}
    estimate = float(mean_length) + NEW_REQUEST_RISK_STD * float(std_length or 0.0)
    max_tokens = generation_params.get("max_tokens")
    if max_tokens is not None:
        estimate = min(estimate, float(max_tokens))
    min_tokens = generation_params.get("min_tokens", 1)
    return max(float(min_tokens), estimate)


def build_class_output_priors(trace_records, historical_lengths, strength=5.0):
    """Empirical-Bayes output priors for requests not yet visible to Helix."""
    values = np.asarray(historical_lengths, dtype=float)
    global_mean = float(np.mean(values))
    global_std = float(np.std(values))
    grouped = {}
    for record, length in zip(trace_records, historical_lengths):
        grouped.setdefault(record["traffic_class"], []).append(float(length))
    priors = {}
    for label, lengths in grouped.items():
        count = len(lengths)
        local_mean = float(np.mean(lengths))
        weight = count / max(1e-9, count + float(strength))
        priors[label] = {
            "mean": weight * local_mean + (1.0 - weight) * global_mean,
            "std": (
                float(np.std(lengths)) if count > 1 else global_std
            ),
            "n": int(count),
        }
    priors["__global__"] = {
        "mean": global_mean,
        "std": global_std,
        "n": int(len(values)),
    }
    return priors


def gamma_arrival_offsets(n_requests, request_rate, burstiness, seed):
    """Generate open-loop arrival offsets with Gamma inter-arrivals.

    burstiness=1 is a Poisson process. Values below one create increasingly
    bursty traffic while preserving the requested mean arrival rate.
    """
    if n_requests <= 0:
        return []
    if not np.isfinite(request_rate) or request_rate <= 0:
        return [0.0] * n_requests
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")

    rng = np.random.default_rng(seed)
    scale = 1.0 / (float(request_rate) * float(burstiness))
    gaps = rng.gamma(
        shape=float(burstiness),
        scale=scale,
        size=max(0, n_requests - 1),
    )
    return [0.0] + np.cumsum(gaps).astype(float).tolist()


def trace_arrival_offsets(
    csv_path,
    n_requests,
    request_rate,
    start_row=0,
    log_type=None,
):
    """Replay consecutive BurstGPT timestamps, scaled to this testbed's RPS.

    Only the timing pattern is replayed. Natural prompts and natural EOS are
    retained because forcing trace output lengths would invalidate evaluation
    of the online length detector.
    """
    timestamps = []
    skipped_matching_rows = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if log_type and row.get("Log Type") != log_type:
                continue
            response_tokens = row.get("Response tokens")
            if response_tokens not in (None, ""):
                try:
                    if float(response_tokens) <= 0:
                        continue
                except ValueError:
                    pass
            if skipped_matching_rows < int(start_row):
                skipped_matching_rows += 1
                continue
            try:
                timestamps.append(float(row["Timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if len(timestamps) >= n_requests:
                break

    if len(timestamps) < n_requests:
        raise ValueError(
            f"Trace supplied only {len(timestamps)} usable arrivals; "
            f"{n_requests} are required"
        )

    raw = np.asarray(timestamps, dtype=float)
    raw -= raw[0]
    raw_duration = float(raw[-1])
    if raw_duration <= 0 or not np.isfinite(request_rate) or request_rate <= 0:
        return raw.tolist()

    target_duration = (n_requests - 1) / float(request_rate)
    return (raw * (target_duration / raw_duration)).tolist()


def measurement_flags(n_requests):
    """Mark a steady-state middle window, leaving warmup and cooldown load."""
    if n_requests <= 0:
        return []
    warmup = int(math.floor(n_requests * MEASUREMENT_WARMUP_FRACTION))
    cooldown = int(math.floor(n_requests * MEASUREMENT_COOLDOWN_FRACTION))
    if n_requests - warmup - cooldown < max(10, n_requests // 2):
        warmup = cooldown = 0
    return [
        warmup <= i < (n_requests - cooldown)
        for i in range(n_requests)
    ]


def block_bootstrap_percentile_ci(
    values,
    percentile,
    seed,
    n_bootstrap=BOOTSTRAP_SAMPLES,
):
    """95% moving-block bootstrap CI for correlated request latencies."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 20:
        return float("nan"), float("nan")

    block_size = max(10, int(math.ceil(math.sqrt(n))))
    block_size = min(block_size, n)
    n_blocks = int(math.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([
            values[s:s + block_size] for s in starts
        ])[:n]
        estimates[i] = np.percentile(sample, percentile)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def block_bootstrap_paired_percentile_difference_ci(
    baseline_values,
    candidate_values,
    percentile,
    seed,
    n_bootstrap=BOOTSTRAP_SAMPLES,
):
    """CI for candidate-minus-baseline percentile using paired request IDs."""
    baseline_values = np.asarray(baseline_values, dtype=float)
    candidate_values = np.asarray(candidate_values, dtype=float)
    valid = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    baseline_values = baseline_values[valid]
    candidate_values = candidate_values[valid]
    n = len(baseline_values)
    if n < 20:
        return float("nan"), float("nan")

    block_size = min(n, max(10, int(math.ceil(math.sqrt(n)))))
    n_blocks = int(math.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        starts = rng.integers(0, max_start, size=n_blocks)
        indices = np.concatenate([
            np.arange(start, start + block_size) for start in starts
        ])[:n]
        estimates[i] = (
            np.percentile(candidate_values[indices], percentile)
            - np.percentile(baseline_values[indices], percentile)
        )
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def attach_paired_queue_comparisons(results, baseline_policy="queue_size"):
    """Attach paired p95 latency and TTFT differences to one condition group."""
    baseline = next(r for r in results if r["policy"] == baseline_policy)
    baseline_by_id = {
        int(rec["req_id"]): rec for rec in baseline["details"]
        if rec.get("measure", True)
    }
    for result in results:
        candidate_by_id = {
            int(rec["req_id"]): rec for rec in result["details"]
            if rec.get("measure", True)
        }
        common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
        baseline_latency = np.asarray([
            baseline_by_id[rid]["finish_time"]
            - baseline_by_id[rid]["arrival_time"]
            for rid in common_ids
        ], dtype=float)
        candidate_latency = np.asarray([
            candidate_by_id[rid]["finish_time"]
            - candidate_by_id[rid]["arrival_time"]
            for rid in common_ids
        ], dtype=float)
        ttft_ids = [
            rid for rid in common_ids
            if baseline_by_id[rid].get("ttft") is not None
            and candidate_by_id[rid].get("ttft") is not None
        ]
        baseline_ttft = np.asarray([
            baseline_by_id[rid]["ttft"] for rid in ttft_ids
        ], dtype=float)
        candidate_ttft = np.asarray([
            candidate_by_id[rid]["ttft"] for rid in ttft_ids
        ], dtype=float)

        result["paired_request_count"] = len(common_ids)
        result["paired_p95_delta_sec"] = (
            float(np.percentile(candidate_latency, 95))
            - float(np.percentile(baseline_latency, 95))
            if common_ids else float("nan")
        )
        latency_ci = block_bootstrap_paired_percentile_difference_ci(
            baseline_latency,
            candidate_latency,
            percentile=95,
            seed=int(result.get("arrival_seed", 0)) + 29500,
        )
        result["paired_p95_delta_ci_low"] = latency_ci[0]
        result["paired_p95_delta_ci_high"] = latency_ci[1]
        result["paired_ttft_p95_delta_sec"] = (
            float(np.percentile(candidate_ttft, 95))
            - float(np.percentile(baseline_ttft, 95))
            if ttft_ids else float("nan")
        )
        ttft_ci = block_bootstrap_paired_percentile_difference_ci(
            baseline_ttft,
            candidate_ttft,
            percentile=95,
            seed=int(result.get("arrival_seed", 0)) + 39500,
        )
        result["paired_ttft_p95_delta_ci_low"] = ttft_ci[0]
        result["paired_ttft_p95_delta_ci_high"] = ttft_ci[1]


def simulate_local_fcfs_completion(
    active_remaining_tokens,
    queued_jobs,
    target_job,
    step_table,
    prefill_sec_per_token,
    max_concurrent=None,
    return_timing=False,
):
    """Approximate completion time for a request appended to a local FCFS queue.

    Under sustained load, vLLM behaves like up to `max_concurrent` decode lanes:
    each live sequence receives one token per decode iteration, and the oldest
    queued request fills the next released slot. A min-heap of slot release
    times therefore gives a stable, work-conserving approximation without
    pretending that the router can reserve or delay future capacity.
    """
    import heapq

    total_jobs = len(active_remaining_tokens) + len(queued_jobs) + 1
    if max_concurrent is None:
        max_concurrent = max(1, total_jobs)
    max_concurrent = max(1, int(max_concurrent))
    expected_batch = max(1, min(max_concurrent, total_jobs))
    step_sec = decode_step_seconds(
        expected_batch, step_table, engine_capacity=max_concurrent
    )

    lanes = [
        max(0.0, float(rem)) * step_sec
        for rem in active_remaining_tokens
        if np.isfinite(rem) and float(rem) > 0
    ]
    if len(lanes) > max_concurrent:
        lanes = sorted(lanes)[:max_concurrent]
    lanes.extend([0.0] * (max_concurrent - len(lanes)))
    heapq.heapify(lanes)

    def add_job(job):
        available = heapq.heappop(lanes)
        generated = max(0.0, float(job.get("generated", 0.0)))
        prompt_cost = 0.0 if generated > 0 else (
            max(1.0, float(job.get("prompt_tokens", 1.0)))
            * max(0.0, float(prefill_sec_per_token))
        )
        decode_cost = max(0.0, float(job["remaining_tokens"])) * step_sec
        finish = available + prompt_cost + decode_cost
        first_token = available + prompt_cost
        if float(job["remaining_tokens"]) > 0:
            first_token += step_sec
        heapq.heappush(lanes, finish)
        return {
            "queue_wait_sec": float(available),
            "prefill_sec": float(prompt_cost),
            "first_token_sec": float(first_token),
            "service_sec": float(prompt_cost + decode_cost),
            "completion_sec": float(finish),
        }

    for job in queued_jobs:
        add_job(job)
    timing = add_job(target_job)
    return timing if return_timing else timing["completion_sec"]


def effective_helix_confidence(confidence, forecast_age):
    """Decay an estimate smoothly instead of dropping Helix at one cutoff."""
    excess_age = max(
        0.0, float(forecast_age) - HELIX_FORECAST_GRACE_SEC
    )
    decay = math.exp(
        -math.log(2.0) * excess_age / HELIX_FORECAST_HALF_LIFE_SEC
    )
    return min(1.0, max(0.0, float(confidence))) * decay


def _request_remaining(policy, rid, snapshot, request_meta, generated_all, true_lengths):
    generated = max(0, int(generated_all[rid]))
    prior_total = float(request_meta[rid]["estimated_output_tokens"])
    prior_remaining = max(0.0, prior_total - generated)

    if policy == "oracle_work":
        return max(0.0, float(true_lengths[rid]) - generated)

    if policy == "helix_work":
        helix = snapshot["helix_by_id"].get(rid)
        if helix is not None:
            helix_remaining, confidence = helix
            confidence = effective_helix_confidence(
                confidence, snapshot["forecast_age"]
            )
            # Confidence-weighted shrinkage protects the router before pi.
            return (
                confidence * max(0.0, float(helix_remaining))
                + (1.0 - confidence) * prior_remaining
            )

    return prior_remaining


def score_worker_local_queue(
    policy,
    gpu_id,
    snapshot,
    pending_ids,
    new_request,
    request_meta,
    generated_all,
    true_lengths=None,
    return_timing=False,
):
    active_set = set(snapshot["active_ids"]) & set(pending_ids)
    active_ids = [rid for rid in snapshot["active_ids"] if rid in active_set]
    queued_ids = [rid for rid in pending_ids if rid not in active_set]

    active_remaining = [
        _request_remaining(
            policy, rid, snapshot, request_meta, generated_all, true_lengths
        )
        for rid in active_ids
    ]
    queued_jobs = [
        {
            "remaining_tokens": _request_remaining(
                policy, rid, snapshot, request_meta, generated_all, true_lengths
            ),
            "generated": int(generated_all[rid]),
            "prompt_tokens": request_meta[rid]["prompt_len"],
        }
        for rid in queued_ids
    ]

    target_remaining = (
        float(true_lengths[new_request["req_id"]])
        if policy == "oracle_work"
        else float(new_request["estimated_output_tokens"])
    )
    target = {
        "remaining_tokens": target_remaining,
        "generated": 0,
        "prompt_tokens": new_request["prompt_len"],
    }
    return simulate_local_fcfs_completion(
        active_remaining_tokens=active_remaining,
        queued_jobs=queued_jobs,
        target_job=target,
        step_table=snapshot["step_table"],
        prefill_sec_per_token=snapshot["prefill_spt"],
        max_concurrent=snapshot["engine_capacity"],
        return_timing=return_timing,
    )


def estimate_admission_waits(
    snapshots,
    pending_by_gpu,
    new_request,
    request_meta,
    generated_all,
):
    """Policy-neutral queue-wait estimates used only for load shedding."""
    waits = {}
    for gpu in (0, 1):
        timing = score_worker_local_queue(
            policy="prior_work",
            gpu_id=gpu,
            snapshot=snapshots[gpu],
            pending_ids=pending_by_gpu[gpu],
            new_request=new_request,
            request_meta=request_meta,
            generated_all=generated_all,
            true_lengths=None,
            return_timing=True,
        )
        waits[gpu] = float(timing["queue_wait_sec"])
    return waits


def score_worker_backlog(
    policy,
    snapshot,
    pending_ids,
    new_request,
    request_meta,
    generated_all,
    true_lengths=None,
):
    """Estimate replica drain time from all unfinished work.

    Continuous batching does not provide independent request lanes below
    max_num_seqs. The useful routing signal is therefore total unfinished
    decode work, adjusted by the replica's measured tokens-per-step and queued
    prefill work. Helix directly changes the active portion of this score.
    """
    active_set = set(snapshot["active_ids"]) & set(pending_ids)
    active_work_tokens = 0.0
    queued_work_tokens = 0.0
    prefill_seconds = 0.0
    informative_active = 0

    for rid in pending_ids:
        remaining = _request_remaining(
            policy,
            rid,
            snapshot,
            request_meta,
            generated_all,
            true_lengths,
        )
        if rid in active_set:
            active_work_tokens += remaining
            helix = snapshot["helix_by_id"].get(rid)
            if (
                policy == "helix_work"
                and helix is not None
                and float(helix[1]) > 0.0
                and effective_helix_confidence(
                    helix[1], snapshot["forecast_age"]
                ) > 0.0
            ):
                informative_active += 1
        else:
            queued_work_tokens += remaining
            if int(generated_all[rid]) <= 0:
                prefill_seconds += (
                    max(1.0, float(request_meta[rid]["prompt_len"]))
                    * float(snapshot["prefill_spt"])
                )

    target_tokens = (
        float(true_lengths[new_request["req_id"]])
        if policy == "oracle_work"
        else float(new_request["estimated_output_tokens"])
    )
    target_prefill_seconds = (
        max(1.0, float(new_request["prompt_len"]))
        * float(snapshot["prefill_spt"])
    )
    total_work_tokens = (
        active_work_tokens + queued_work_tokens + target_tokens
    )
    capacity = max(1, int(snapshot["engine_capacity"]))
    expected_batch = min(
        capacity, max(1, len(pending_ids) + 1)
    )
    step_seconds = decode_step_seconds(
        expected_batch,
        snapshot["step_table"],
        engine_capacity=capacity,
    )
    output_tokens_per_second = expected_batch / max(1e-6, step_seconds)
    decode_drain_seconds = (
        total_work_tokens / max(1e-6, output_tokens_per_second)
    )
    drain_seconds = (
        decode_drain_seconds + prefill_seconds + target_prefill_seconds
    )
    return {
        "drain_sec": float(drain_seconds),
        "decode_drain_sec": float(decode_drain_seconds),
        "prefill_sec": float(prefill_seconds + target_prefill_seconds),
        "remaining_work_tokens": float(total_work_tokens),
        "active_work_tokens": float(active_work_tokens),
        "queued_work_tokens": float(queued_work_tokens),
        "target_work_tokens": float(target_tokens),
        "expected_batch": int(expected_batch),
        "engine_capacity": int(capacity),
        "decode_step_sec": float(step_seconds),
        "output_tokens_per_sec": float(output_tokens_per_second),
        "seconds_per_work_token": float(
            1.0 / max(1e-6, output_tokens_per_second)
        ),
        "informative_active": int(informative_active),
        "active_count": int(len(active_set)),
    }


def pick_gpu_immediate(
    policy,
    rr_next,
    snapshots,
    pending_by_gpu,
    new_request,
    request_meta,
    generated_all,
    historical_mean,
    true_lengths=None,
):
    """Return an immediate and permanent replica assignment.

    Helix and the trace oracle minimize measured predicted backlog drain time.
    Queue size remains unchanged and is used only as the comparison baseline.
    """
    pending_counts = [len(pending_by_gpu[0]), len(pending_by_gpu[1])]

    if policy == "round_robin":
        return rr_next, None

    if policy == "queue_size":
        if pending_counts[0] != pending_counts[1]:
            return int(pending_counts[1] < pending_counts[0]), None
        return rr_next, None

    if pending_counts[0] != pending_counts[1]:
        queue_choice = int(pending_counts[1] < pending_counts[0])
    else:
        queue_choice = rr_next

    timings = {
        gpu: score_worker_backlog(
            policy=policy,
            snapshot=snapshots[gpu],
            pending_ids=pending_by_gpu[gpu],
            new_request=new_request,
            request_meta=request_meta,
            generated_all=generated_all,
            true_lengths=true_lengths,
        )
        for gpu in (0, 1)
    }
    objective = "drain_sec"
    scores = {gpu: float(timings[gpu][objective]) for gpu in (0, 1)}
    candidate = 0 if scores[0] < scores[1] else 1
    predicted_gain = max(0.0, scores[queue_choice] - scores[candidate])
    count_disadvantage = pending_counts[candidate] - pending_counts[queue_choice]
    mean_seconds_per_token = 0.5 * sum(
        float(timings[gpu]["seconds_per_work_token"]) for gpu in (0, 1)
    )
    required_gain = max(
        float(ROUTING_DEADBAND_SEC),
        float(ROUTING_DEADBAND_TOKENS) * mean_seconds_per_token,
    )

    if candidate == queue_choice:
        gpu = queue_choice
        reason = "work_score_agrees_with_queue"
    elif predicted_gain < required_gain:
        gpu = queue_choice
        reason = "predicted_gain_guard"
    else:
        gpu = candidate
        reason = f"minimum_predicted_{objective}"

    overrode = gpu != queue_choice

    return gpu, {
        "scores": scores,
        "timings": timings,
        "objective": objective,
        "queue_choice": queue_choice,
        "work_choice": candidate,
        "overrode_queue": overrode,
        "predicted_gain_sec": float(max(0.0, predicted_gain)),
        "predicted_gain_tokens": float(
            predicted_gain / max(1e-9, mean_seconds_per_token)
        ),
        "count_disadvantage": int(count_disadvantage),
        "min_gain_sec": float(required_gain),
        "deadband_tokens": float(ROUTING_DEADBAND_TOKENS),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# EXPERIMENT DRIVER - SAME LOCAL-QUEUE ARCHITECTURE FOR EVERY POLICY
# ---------------------------------------------------------------------------

def run_local_queue_experiment(
    policy_name,
    prompts,
    cache,
    calib_params,
    historical_mean,
    historical_std,
    prompt_lengths,
    historical_priors=None,
    true_lengths=None,
    generation_params=None,
    traffic_classes=None,
    system_prompts=None,
    arrival_offsets=None,
    queue_wait_budget_sec=DEFAULT_QUEUE_WAIT_BUDGET_SEC,
    offered_rps=float("inf"),
    load_factor=None,
    arrival_mode="gamma",
    burstiness=DEFAULT_BURSTINESS,
    arrival_seed=POLICY_SEED,
    ttft_slo_sec=DEFAULT_TTFT_SLO_SEC,
    itl_slo_sec=DEFAULT_ITL_SLO_SEC,
    experiment_id=None,
    use_measurement_window=True,
    enable_helix_tracker=True,
    priority_ttft_deadline_sec=None,
    priority_itl_rescue_sec=None,
):
    if policy_name not in POLICY_LABELS:
        raise ValueError(f"Unknown policy: {policy_name}")
    if policy_name == "oracle_work" and true_lengths is None:
        raise ValueError("oracle_work requires true_lengths")

    n_requests = len(prompts)
    historical_priors = historical_priors or {
        "__global__": {
            "mean": float(historical_mean),
            "std": float(historical_std),
        }
    }
    generation_params = generation_params or [{} for _ in prompts]
    if isinstance(generation_params, dict):
        generation_params = [generation_params] * n_requests
    traffic_classes = traffic_classes or ["unknown"] * n_requests
    if isinstance(traffic_classes, str):
        traffic_classes = [traffic_classes] * n_requests
    system_prompts = system_prompts or [None] * n_requests
    if isinstance(system_prompts, str):
        system_prompts = [system_prompts] * n_requests
    arrival_offsets = (
        [0.0] * n_requests if arrival_offsets is None
        else [float(v) for v in arrival_offsets]
    )
    if not (
        len(generation_params)
        == len(traffic_classes)
        == len(system_prompts)
        == len(arrival_offsets)
        == n_requests
    ):
        raise ValueError("All per-request inputs must match prompts in length")
    if any(b < a for a, b in zip(arrival_offsets, arrival_offsets[1:])):
        raise ValueError("arrival_offsets must be nondecreasing")
    measured_request = (
        measurement_flags(n_requests)
        if use_measurement_window else [True] * n_requests
    )

    shared = _create_shared_state(n_requests)
    input_queues = [mp.Queue(), mp.Queue()]
    output_queue = mp.Queue()
    ready_queue = mp.Queue()
    enable_priority_scheduler = (
        ENABLE_EXPERIMENTAL_VLLM_PREEMPTION
        and policy_name in ("helix_work", "oracle_work")
    )
    effective_priority_ttft_deadline_sec = (
        float(priority_ttft_deadline_sec)
        if priority_ttft_deadline_sec is not None
        else PRIORITY_TTFT_DEADLINE_SEC
    )
    if queue_wait_budget_sec is not None:
        effective_priority_ttft_deadline_sec = min(
            effective_priority_ttft_deadline_sec,
            0.80 * float(queue_wait_budget_sec),
        )
    effective_priority_itl_rescue_sec = (
        float(priority_itl_rescue_sec)
        if priority_itl_rescue_sec is not None
        else PRIORITY_ITL_RESCUE_SEC
    )
    priority_cohort_size = (
        HELIX_PRIORITY_COHORT_SIZE
        if policy_name == "helix_work"
        else ORACLE_PRIORITY_COHORT_SIZE
    )
    priority_knowledge = (
        "helix" if policy_name == "helix_work" else "oracle"
    )

    workers = []
    for gpu_id in (0, 1):
        p = mp.Process(
            target=gpu_worker_local_queue,
            args=(
                gpu_id, MODEL_ID, cache, calib_params, historical_mean,
                enable_helix_tracker,
                enable_priority_scheduler,
                effective_priority_ttft_deadline_sec,
                effective_priority_itl_rescue_sec,
                priority_cohort_size,
                priority_knowledge,
                input_queues[gpu_id], output_queue, ready_queue,
                shared["locks"][gpu_id],
                shared["engine_capacity"],
                shared["scheduler_running_count"],
                shared["scheduler_waiting_count"],
                shared["slot_width"],
                shared["table_width"],
                shared["local_unfinished"], shared["active_decode_count"],
                shared["active_ids"], shared["active_helix_remaining"],
                shared["active_helix_confidence"], shared["generated_all"],
                shared["forecast_timestamp"], shared["decode_step_table"],
                shared["decode_step_samples"], shared["prefill_sec_per_token"],
                shared["engine_busy_seconds"], shared["decode_slot_seconds"],
                shared["hook_diag"],
            ),
        )
        # A failed benchmark must not leave a vLLM replica keeping the Kaggle
        # GPU alive while the parent unwinds an exception.
        p.daemon = True
        p.start()
        workers.append(p)

    last_worker_health_check = [0.0]

    def assert_workers_healthy(context, force=False):
        now = time.time()
        if not force and now - last_worker_health_check[0] < 0.5:
            return
        last_worker_health_check[0] = now
        dead = [
            (gpu, process.exitcode)
            for gpu, process in enumerate(workers)
            if not process.is_alive()
        ]
        if not dead:
            return
        for process in workers:
            if process.is_alive():
                process.terminate()
        detail = ", ".join(
            f"GPU {gpu} exitcode={exitcode}" for gpu, exitcode in dead
        )
        raise RuntimeError(
            f"vLLM worker failed during {context}: {detail}. "
            "The run was stopped immediately instead of waiting forever."
        )

    ready_gpus = set()
    ready_deadline = time.time() + WORKER_READY_TIMEOUT_SEC
    try:
        while len(ready_gpus) < 2:
            remaining = ready_deadline - time.time()
            if remaining <= 0:
                raise queue.Empty
            try:
                msg = ready_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                assert_workers_healthy("startup", force=True)
                continue
            if msg.get("status") != "ready":
                raise RuntimeError(f"Worker readiness failed: {msg}")
            ready_gpus.add(int(msg["gpu_id"]))
    except queue.Empty as exc:
        for p in workers:
            if p.is_alive():
                p.terminate()
        raise TimeoutError(
            f"Only workers {sorted(ready_gpus)} became ready within "
            f"{WORKER_READY_TIMEOUT_SEC}s"
        ) from exc
    except Exception:
        for p in workers:
            if p.is_alive():
                p.terminate()
        raise
    engine_capacities = [
        max(1, int(shared["engine_capacity"][gpu])) for gpu in (0, 1)
    ]

    pending_by_gpu = [[], []]
    request_meta = {}
    rr_next = 0
    results = []
    rejections = []
    accepted_count = 0
    assignment_count = [0, 0]
    queued_on_chosen_worker = 0
    system_saturated_arrivals = 0
    max_pending = [0, 0]
    max_waiting = [0, 0]
    pending_depth_at_arrival = []
    waiting_depth_at_arrival = []
    engine_waiting_depth_at_arrival = []
    engine_running_depth_at_arrival = []
    active_depth_at_arrival = []
    score_margins_sec = []
    helix_conf_samples = []
    helix_active_snapshot_arrivals = 0
    helix_fresh_active_snapshot_arrivals = 0
    helix_active_ids_seen = 0
    helix_estimates_seen = 0
    helix_informative_estimates_seen = 0
    work_override_count = 0
    work_candidate_disagreement_count = 0
    routing_reason_counts = {}
    predicted_gain_samples = []
    predicted_gain_token_samples = []
    routing_score_gpu0_samples = []
    routing_score_gpu1_samples = []

    t0 = time.time()
    last_arrival_time = t0

    def collect_results():
        while True:
            try:
                rec = output_queue.get_nowait()
            except queue.Empty:
                break
            rid = int(rec["req_id"])
            gpu = int(rec["gpu_id"])
            if rid in pending_by_gpu[gpu]:
                pending_by_gpu[gpu].remove(rid)
            results.append(rec)

    for rid, prompt in enumerate(prompts):
        scheduled_time = t0 + arrival_offsets[rid]
        while time.time() < scheduled_time:
            collect_results()
            assert_workers_healthy("arrival scheduling")
            time.sleep(min(0.003, max(0.0002, scheduled_time - time.time())))
        collect_results()
        assert_workers_healthy("request routing")
        arrival_time = time.time()
        last_arrival_time = arrival_time
        snapshots = {
            gpu: snapshot_worker_local(shared, gpu) for gpu in (0, 1)
        }
        pending_depth_at_arrival.append(
            sum(len(pending_by_gpu[gpu]) for gpu in (0, 1))
        )
        waiting_depth_at_arrival.append(
            sum(
                max(
                    0,
                    len(pending_by_gpu[gpu])
                    - len(snapshots[gpu]["active_ids"]),
                )
                for gpu in (0, 1)
            )
        )
        engine_waiting_depth_at_arrival.append(
            sum(
                snapshots[gpu]["scheduler_waiting_count"]
                for gpu in (0, 1)
            )
        )
        engine_running_depth_at_arrival.append(
            sum(
                snapshots[gpu]["scheduler_running_count"]
                for gpu in (0, 1)
            )
        )
        active_depth_at_arrival.append(
            sum(len(snapshots[gpu]["active_ids"]) for gpu in (0, 1))
        )
        if policy_name == "helix_work":
            active_this_arrival = sum(
                len(snap["active_ids"]) for snap in snapshots.values()
            )
            estimates = [
                estimate
                for snap in snapshots.values()
                for estimate in snap["helix_by_id"].values()
            ]
            if active_this_arrival > 0:
                helix_active_snapshot_arrivals += 1
                if any(
                    snap["active_ids"]
                    and snap["forecast_age"] <= HELIX_FORECAST_GRACE_SEC
                    for snap in snapshots.values()
                ):
                    helix_fresh_active_snapshot_arrivals += 1
            helix_active_ids_seen += active_this_arrival
            helix_estimates_seen += len(estimates)
            helix_informative_estimates_seen += sum(
                float(confidence) > 0.0
                for _, confidence in estimates
            )
            helix_conf_samples.extend(
                float(confidence) for _, confidence in estimates
            )

        if all(
            len(pending_by_gpu[g]) > len(snapshots[g]["active_ids"])
            for g in (0, 1)
        ):
            system_saturated_arrivals += 1

        gp = generation_params[rid]
        class_prior = historical_priors.get(
            traffic_classes[rid],
            historical_priors["__global__"],
        )
        estimated_output_tokens = estimate_new_output_tokens(
            class_prior["mean"], class_prior["std"], gp
        )
        req = {
            "req_id": rid,
            "prompt": prompt,
            "prompt_len": int(prompt_lengths[rid]),
            "arrival_time": arrival_time,
            "scheduled_arrival_time": scheduled_time,
            "estimated_output_tokens": estimated_output_tokens,
            "output_prior_source": (
                traffic_classes[rid]
                if traffic_classes[rid] in historical_priors
                else "__global__"
            ),
            "generation_params": gp,
            "traffic_class": traffic_classes[rid],
            "system_prompt": system_prompts[rid],
            "measure": measured_request[rid],
            "arrival_rank": rid,
            "scheduler_output_tokens": (
                float(true_lengths[rid])
                if policy_name == "oracle_work"
                else estimated_output_tokens
            ),
        }
        # Insert metadata before scoring so helper functions can address the new
        # request uniformly. It is not yet present in either pending queue.
        request_meta[rid] = req

        gpu, decision = pick_gpu_immediate(
            policy=policy_name,
            rr_next=rr_next,
            snapshots=snapshots,
            pending_by_gpu=pending_by_gpu,
            new_request=req,
            request_meta=request_meta,
            generated_all=shared["generated_all"],
            historical_mean=historical_mean,
            true_lengths=true_lengths,
        )
        rr_next = 1 - rr_next

        admission_waits = (
            estimate_admission_waits(
                snapshots=snapshots,
                pending_by_gpu=pending_by_gpu,
                new_request=req,
                request_meta=request_meta,
                generated_all=shared["generated_all"],
            )
            if queue_wait_budget_sec is not None
            else {0: 0.0, 1: 0.0}
        )
        req["predicted_queue_wait_sec"] = admission_waits[gpu]

        should_shed = (
            queue_wait_budget_sec is not None
            and min(admission_waits.values()) > float(queue_wait_budget_sec)
        )
        if should_shed:
            request_meta.pop(rid, None)
            rejections.append({
                "req_id": rid,
                "arrival_time": arrival_time,
                "scheduled_arrival_time": scheduled_time,
                "traffic_class": traffic_classes[rid],
                "measure": measured_request[rid],
                "reason": "queue_wait_budget",
                "queue_wait_budget_sec": float(queue_wait_budget_sec),
                "predicted_wait_gpu0_sec": admission_waits[0],
                "predicted_wait_gpu1_sec": admission_waits[1],
                "policy_choice": gpu,
            })
            continue

        if (
            len(pending_by_gpu[gpu])
            > len(snapshots[gpu]["active_ids"])
        ):
            queued_on_chosen_worker += 1
        if decision is not None:
            scores = decision.get("scores")
            if scores is not None and np.isfinite(scores[0]) and np.isfinite(scores[1]):
                score_margins_sec.append(abs(scores[0] - scores[1]))
                routing_score_gpu0_samples.append(float(scores[0]))
                routing_score_gpu1_samples.append(float(scores[1]))
            predicted_gain_samples.append(
                float(decision.get("predicted_gain_sec", 0.0))
            )
            predicted_gain_token_samples.append(
                float(decision.get("predicted_gain_tokens", 0.0))
            )
            reason = str(decision.get("reason", "unknown"))
            routing_reason_counts[reason] = (
                routing_reason_counts.get(reason, 0) + 1
            )
            if decision.get("work_choice") != decision.get("queue_choice"):
                work_candidate_disagreement_count += 1
            if decision.get("overrode_queue"):
                work_override_count += 1
            req.update({
                "routing_reason": reason,
                "routing_queue_choice": decision.get("queue_choice"),
                "routing_work_choice": decision.get("work_choice"),
                "routing_overrode_queue": bool(
                    decision.get("overrode_queue")
                ),
                "routing_predicted_gain_sec": float(
                    decision.get("predicted_gain_sec", 0.0)
                ),
                "routing_predicted_gain_tokens": float(
                    decision.get("predicted_gain_tokens", 0.0)
                ),
                "routing_score_gpu0_sec": float(scores[0]),
                "routing_score_gpu1_sec": float(scores[1]),
                "routing_gpu0_active_work_tokens": float(
                    decision["timings"][0]["active_work_tokens"]
                ),
                "routing_gpu1_active_work_tokens": float(
                    decision["timings"][1]["active_work_tokens"]
                ),
                "routing_gpu0_total_work_tokens": float(
                    decision["timings"][0]["remaining_work_tokens"]
                ),
                "routing_gpu1_total_work_tokens": float(
                    decision["timings"][1]["remaining_work_tokens"]
                ),
            })
        pending_by_gpu[gpu].append(rid)
        accepted_count += 1
        assignment_count[gpu] += 1
        max_pending[gpu] = max(max_pending[gpu], len(pending_by_gpu[gpu]))
        max_waiting[gpu] = max(
            max_waiting[gpu],
            max(
                0,
                len(pending_by_gpu[gpu])
                - len(snapshots[gpu]["active_ids"]),
            ),
        )
        input_queues[gpu].put(req)

    while len(results) < accepted_count:
        collect_results()
        assert_workers_healthy("request completion")
        time.sleep(0.01)

    for q in input_queues:
        q.put(None)
    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    collect_results()
    results.sort(key=lambda x: x["req_id"])
    if len(results) != accepted_count:
        raise RuntimeError(
            f"Completed {len(results)}/{accepted_count} accepted requests"
        )

    final_time = (
        max(r["finish_time"] for r in results)
        if results else last_arrival_time
    )
    wall = final_time - t0
    wall = max(1e-9, wall)
    measured_results = [r for r in results if r.get("measure", True)]
    measured_rejections = [r for r in rejections if r.get("measure", True)]
    if not measured_results and results:
        measured_results = list(results)
    measurement_start = (
        min(r["arrival_time"] for r in measured_results)
        if measured_results else t0
    )
    measurement_end = (
        max(r["finish_time"] for r in measured_results)
        if measured_results else max(last_arrival_time, measurement_start)
    )
    measurement_wall = max(1e-9, measurement_end - measurement_start)
    latencies = np.asarray([
        r["finish_time"] - r["arrival_time"] for r in measured_results
    ], dtype=float)
    ttfts = np.asarray([
        r["ttft"] for r in measured_results if r.get("ttft") is not None
    ], dtype=float)
    itls = np.concatenate([
        np.asarray(r["itl_gaps"], dtype=float)
        for r in measured_results if r.get("itl_gaps")
    ]) if any(r.get("itl_gaps") for r in measured_results) else np.asarray([], dtype=float)
    max_itls = np.asarray([
        max(r.get("itl_gaps") or [0.0]) for r in measured_results
    ], dtype=float)
    scheduler_preemption_values = np.asarray([
        r["scheduler_preemptions"] for r in measured_results
        if r.get("scheduler_preemptions") is not None
    ], dtype=float)
    priority_update_values = np.asarray([
        r.get("scheduler_priority_updates", 0) for r in measured_results
    ], dtype=float)
    priority_transition_values = np.asarray([
        r.get("scheduler_priority_transitions", 0)
        for r in measured_results
    ], dtype=float)
    priority_lookup_misses = sum(
        int(r.get("priority_lookup_misses", 0)) for r in measured_results
    )

    def pct(values, q):
        return float(np.percentile(values, q)) if len(values) else float('nan')

    latency_p95_ci = block_bootstrap_percentile_ci(
        latencies, 95, seed=arrival_seed + 9500
    )
    latency_p99_ci = block_bootstrap_percentile_ci(
        latencies, 99, seed=arrival_seed + 9900
    )
    ttft_p95_ci = block_bootstrap_percentile_ci(
        ttfts, 95, seed=arrival_seed + 19500
    )

    total_tokens = sum(int(r["n_tokens"]) for r in measured_results)
    total_prompt_tokens = sum(int(r.get("prompt_len", 0)) for r in measured_results)
    measured_submitted = len(measured_results) + len(measured_rejections)
    good_requests = 0
    for rec in measured_results:
        rec_itl_p95 = pct(np.asarray(rec.get("itl_gaps", []), dtype=float), 95)
        ttft_ok = rec.get("ttft") is not None and rec["ttft"] <= ttft_slo_sec
        itl_ok = not np.isfinite(rec_itl_p95) or rec_itl_p95 <= itl_slo_sec
        if ttft_ok and itl_ok:
            good_requests += 1

    predicted_queue_wait_values = np.asarray([
        r.get("predicted_queue_wait_sec", 0.0) for r in measured_results
    ], dtype=float)
    engine_queue_wait_values = np.asarray([
        r["engine_queue_time_sec"] for r in measured_results
        if r.get("engine_queue_time_sec") is not None
    ], dtype=float)
    scheduler_queue_wait_values = np.asarray([
        r["scheduler_queue_time_sec"] for r in measured_results
        if r.get("scheduler_queue_time_sec") is not None
    ], dtype=float)
    gpu_busy = [
        float(shared["engine_busy_seconds"][g]) / wall for g in (0, 1)
    ]
    slot_occupancy = [
        float(shared["decode_slot_seconds"][g])
        / (engine_capacities[g] * wall)
        for g in (0, 1)
    ]
    hook_calls = sum(int(shared["hook_diag"][g * 4]) for g in (0, 1))
    hook_skips = sum(int(shared["hook_diag"][g * 4 + 1]) for g in (0, 1))
    hook_errors = sum(int(shared["hook_diag"][g * 4 + 2]) for g in (0, 1))
    slot_exhaustions = sum(int(shared["hook_diag"][g * 4 + 3]) for g in (0, 1))
    helix_active_snapshot_fraction = (
        helix_active_snapshot_arrivals / max(1, n_requests)
    )
    helix_fresh_active_snapshot_fraction = (
        helix_fresh_active_snapshot_arrivals / max(1, n_requests)
    )
    helix_estimate_coverage = (
        helix_estimates_seen / max(1, helix_active_ids_seen)
    )
    helix_informative_estimate_fraction = (
        helix_informative_estimates_seen / max(1, helix_estimates_seen)
    )
    helix_snapshot_valid = (
        policy_name != "helix_work"
        or (
            helix_active_snapshot_fraction
            >= MIN_HELIX_ACTIVE_SNAPSHOT_FRACTION
            and helix_fresh_active_snapshot_fraction
            >= MIN_HELIX_ACTIVE_SNAPSHOT_FRACTION
            and helix_estimate_coverage >= 0.80
            and helix_informative_estimate_fraction
            >= MIN_HELIX_INFORMATIVE_ESTIMATE_FRACTION
        )
    )

    result = {
        "policy": policy_name,
        "label": POLICY_LABELS[policy_name],
        "helix_tracker_enabled": bool(enable_helix_tracker),
        "experiment_id": experiment_id,
        "arrival_mode": arrival_mode,
        "arrival_seed": int(arrival_seed),
        "burstiness": float(burstiness),
        "offered_rps": float(offered_rps),
        "load_factor": (
            float(load_factor) if load_factor is not None else None
        ),
        "queue_wait_budget_sec": (
            float(queue_wait_budget_sec)
            if queue_wait_budget_sec is not None else None
        ),
        "p50": pct(latencies, 50),
        "p95": pct(latencies, 95),
        "p99": pct(latencies, 99),
        "p95_ci_low": latency_p95_ci[0],
        "p95_ci_high": latency_p95_ci[1],
        "p99_ci_low": latency_p99_ci[0],
        "p99_ci_high": latency_p99_ci[1],
        "ttft_p50": pct(ttfts, 50),
        "ttft_p95": pct(ttfts, 95),
        "ttft_p99": pct(ttfts, 99),
        "ttft_p95_ci_low": ttft_p95_ci[0],
        "ttft_p95_ci_high": ttft_p95_ci[1],
        "itl_p50": pct(itls, 50),
        "itl_p95": pct(itls, 95),
        "itl_p99": pct(itls, 99),
        "max_itl_p95": pct(max_itls, 95),
        "max_itl_max": float(np.max(max_itls)) if len(max_itls) else float("nan"),
        "priority_scheduler_enabled": bool(enable_priority_scheduler),
        "priority_ttft_deadline_sec": (
            float(effective_priority_ttft_deadline_sec)
            if enable_priority_scheduler else None
        ),
        "priority_itl_rescue_sec": (
            float(effective_priority_itl_rescue_sec)
            if enable_priority_scheduler else None
        ),
        "priority_cohort_size": (
            int(priority_cohort_size) if enable_priority_scheduler else None
        ),
        "scheduler_preemptions_mean": (
            float(np.mean(scheduler_preemption_values))
            if len(scheduler_preemption_values) else float("nan")
        ),
        "scheduler_preemptions_coverage": (
            len(scheduler_preemption_values) / max(1, len(measured_results))
        ),
        "priority_updates_mean": (
            float(np.mean(priority_update_values))
            if len(priority_update_values) else 0.0
        ),
        "priority_transitions_mean": (
            float(np.mean(priority_transition_values))
            if len(priority_transition_values) else 0.0
        ),
        "priority_lookup_misses": int(priority_lookup_misses),
        "tokens_per_sec": total_tokens / measurement_wall,
        "prompt_tokens_per_sec": total_prompt_tokens / measurement_wall,
        "accepted_requests_per_sec": len(measured_results) / measurement_wall,
        "goodput_requests_per_sec": good_requests / measurement_wall,
        "goodput_fraction": good_requests / max(1, measured_submitted),
        "shed_fraction": len(measured_rejections) / max(1, measured_submitted),
        "predicted_queue_wait_p50": pct(predicted_queue_wait_values, 50),
        "predicted_queue_wait_p95": pct(predicted_queue_wait_values, 95),
        "predicted_queue_wait_p99": pct(predicted_queue_wait_values, 99),
        "engine_queue_wait_p50": pct(engine_queue_wait_values, 50),
        "engine_queue_wait_p95": pct(engine_queue_wait_values, 95),
        "engine_queue_wait_p99": pct(engine_queue_wait_values, 99),
        "engine_queue_metrics_coverage": (
            len(engine_queue_wait_values) / max(1, len(measured_results))
        ),
        "scheduler_queue_wait_p50": pct(scheduler_queue_wait_values, 50),
        "scheduler_queue_wait_p95": pct(scheduler_queue_wait_values, 95),
        "scheduler_queue_wait_p99": pct(scheduler_queue_wait_values, 99),
        "scheduler_queue_metrics_coverage": (
            len(scheduler_queue_wait_values) / max(1, len(measured_results))
        ),
        "gpu_busy": gpu_busy,
        "mean_gpu_busy": float(np.mean(gpu_busy)),
        "slot_occupancy": slot_occupancy,
        "mean_slot_occupancy": float(np.mean(slot_occupancy)),
        "engine_capacity_gpu0": int(engine_capacities[0]),
        "engine_capacity_gpu1": int(engine_capacities[1]),
        "mean_engine_capacity": float(np.mean(engine_capacities)),
        "assignment_count": assignment_count,
        "max_pending": max_pending,
        "max_waiting": max_waiting,
        "pending_depth_at_arrival_mean": float(
            np.mean(pending_depth_at_arrival)
        ) if pending_depth_at_arrival else 0.0,
        "pending_depth_at_arrival_p95": pct(
            np.asarray(pending_depth_at_arrival, dtype=float), 95
        ),
        "waiting_depth_at_arrival_p95": pct(
            np.asarray(waiting_depth_at_arrival, dtype=float), 95
        ),
        "engine_waiting_depth_at_arrival_p95": pct(
            np.asarray(engine_waiting_depth_at_arrival, dtype=float), 95
        ),
        "engine_running_depth_at_arrival_p95": pct(
            np.asarray(engine_running_depth_at_arrival, dtype=float), 95
        ),
        "active_depth_at_arrival_mean": float(
            np.mean(active_depth_at_arrival)
        ) if active_depth_at_arrival else 0.0,
        "queue_nonempty_at_arrival_fraction": (
            sum(depth > 0 for depth in pending_depth_at_arrival)
            / max(1, len(pending_depth_at_arrival))
        ),
        "waiting_at_arrival_fraction": (
            sum(depth > 0 for depth in waiting_depth_at_arrival)
            / max(1, len(waiting_depth_at_arrival))
        ),
        "queued_on_chosen_fraction": queued_on_chosen_worker / max(1, accepted_count),
        "system_saturated_fraction": system_saturated_arrivals / max(1, n_requests),
        "mean_score_margin_sec": (
            float(np.mean(score_margins_sec))
            if score_margins_sec else float('nan')
        ),
        "mean_helix_confidence": (
            float(np.mean(helix_conf_samples)) if helix_conf_samples else float('nan')
        ),
        "helix_active_snapshot_fraction": helix_active_snapshot_fraction,
        "helix_fresh_active_snapshot_fraction": (
            helix_fresh_active_snapshot_fraction
        ),
        "helix_estimate_coverage": helix_estimate_coverage,
        "helix_informative_estimate_fraction": (
            helix_informative_estimate_fraction
        ),
        "helix_snapshot_valid": bool(helix_snapshot_valid),
        "assignment_gpu0": int(assignment_count[0]),
        "assignment_gpu1": int(assignment_count[1]),
        "max_pending_gpu0": int(max_pending[0]),
        "max_pending_gpu1": int(max_pending[1]),
        "work_override_count": work_override_count,
        "work_override_fraction": (
            work_override_count / max(1, accepted_count)
        ),
        "work_candidate_disagreement_count": (
            work_candidate_disagreement_count
        ),
        "work_override_acceptance_fraction": (
            work_override_count
            / max(1, work_candidate_disagreement_count)
        ),
        "routing_reason_counts": routing_reason_counts,
        "routing_predicted_gain_mean_sec": (
            float(np.mean(predicted_gain_samples))
            if predicted_gain_samples else float("nan")
        ),
        "routing_predicted_gain_p95_sec": pct(
            np.asarray(predicted_gain_samples, dtype=float), 95
        ),
        "routing_predicted_gain_mean_tokens": (
            float(np.mean(predicted_gain_token_samples))
            if predicted_gain_token_samples else float("nan")
        ),
        "routing_score_gpu0_mean_sec": (
            float(np.mean(routing_score_gpu0_samples))
            if routing_score_gpu0_samples else float("nan")
        ),
        "routing_score_gpu1_mean_sec": (
            float(np.mean(routing_score_gpu1_samples))
            if routing_score_gpu1_samples else float("nan")
        ),
        "hook_skip_ratio": hook_skips / hook_calls if hook_calls else 0.0,
        "hook_true_error_ratio": hook_errors / hook_calls if hook_calls else 0.0,
        "tracker_slot_exhaustions": slot_exhaustions,
        "n_submitted": n_requests,
        "n_accepted": accepted_count,
        "n_rejected": len(rejections),
        "n_completed": len(results),
        "n_measured_completed": len(measured_results),
        "measurement_wall_sec": measurement_wall,
        "wall_sec": wall,
        "mean_output_tokens": (
            total_tokens / len(measured_results) if measured_results else float("nan")
        ),
        "mean_prompt_tokens": (
            total_prompt_tokens / len(measured_results)
            if measured_results else float("nan")
        ),
        "traffic_class_metrics": summarize_traffic_classes(measured_results),
        "details": results,
        "rejections": rejections,
    }
    print_policy_card(result)
    return result


def summarize_traffic_classes(records):
    grouped = {}
    for rec in records:
        grouped.setdefault(rec.get("traffic_class", "unknown"), []).append(rec)

    summary = {}
    for label, rows in sorted(grouped.items()):
        latencies = np.asarray([
            r["finish_time"] - r["arrival_time"] for r in rows
        ], dtype=float)
        ttfts = np.asarray([
            r["ttft"] for r in rows if r.get("ttft") is not None
        ], dtype=float)
        outputs = np.asarray([r["n_tokens"] for r in rows], dtype=float)
        prompts = np.asarray([r.get("prompt_len", 0) for r in rows], dtype=float)

        def pct(values, q):
            return float(np.percentile(values, q)) if len(values) else float("nan")

        summary[label] = {
            "n": len(rows),
            "latency_p50": pct(latencies, 50),
            "latency_p95": pct(latencies, 95),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "prompt_tokens_mean": float(np.mean(prompts)) if len(prompts) else 0.0,
            "output_tokens_mean": float(np.mean(outputs)) if len(outputs) else 0.0,
            "output_tokens_p95": pct(outputs, 95),
        }
    return summary


# ---------------------------------------------------------------------------
# SANITY CHECKS AND READABLE REPORTING
# ---------------------------------------------------------------------------

def run_routing_sanity_checks():
    test_capacity = 24
    tiny_cache = {
        "mu": np.zeros(4, dtype=np.float32),
        "v0": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "v1": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "s0": 1.0,
        "s1": 1.0,
        "omega_prior": 0.1,
    }
    with torch.inference_mode():
        inference_tracker = BatchedCausalPLLTracker(
            tiny_cache,
            batch_size=2,
            device=torch.device("cpu"),
            use_compile=False,
        )
        inference_tracker.count += 1
    # Regression check: inference tensors must remain resettable after leaving
    # the context used by vLLM's model forward.
    inference_tracker.reset_slots([0])
    assert float(inference_tracker.count[0].item()) == 0.0

    # Sparse online updates must match the original dense kernel for scheduled
    # rows; otherwise the performance optimization would change Helix itself.
    dense_tracker = BatchedCausalPLLTracker(
        tiny_cache,
        batch_size=4,
        device=torch.device("cpu"),
        use_compile=False,
    )
    sparse_tracker = BatchedCausalPLLTracker(
        tiny_cache,
        batch_size=4,
        device=torch.device("cpu"),
        use_compile=False,
    )
    active_slots = [1, 3]
    for step in range(4):
        active_rows = torch.tensor(
            [
                [1.0 + step, 0.5, -0.25, 0.1],
                [0.2, 1.5 + step, 0.3, -0.2],
            ],
            dtype=torch.float32,
        )
        dense_rows = torch.zeros((4, 4), dtype=torch.float32)
        dense_rows[active_slots] = active_rows
        dense_mask = torch.zeros(4, dtype=torch.bool)
        dense_mask[active_slots] = True
        dense_tracker.update(dense_rows, dense_mask)
        sparse_tracker.update_slots(active_rows, active_slots)
    for name in ("count", "theta_hat", "omega_hat", "sum0", "sum1"):
        dense_values = getattr(dense_tracker, name)[active_slots]
        sparse_values = getattr(sparse_tracker, name)[active_slots]
        assert torch.allclose(
            dense_values, sparse_values, atol=1e-6, rtol=1e-5
        ), f"sparse tracker mismatch in {name}"

    known_ids = {"17": {}, "request-with-dashes": {}}
    assert resolve_external_request_id("17", known_ids) == "17"
    assert (
        resolve_external_request_id("17-a1b2c3d4", known_ids) == "17"
    )
    assert (
        resolve_external_request_id(
            "request-with-dashes-DEADBEEF", known_ids
        )
        == "request-with-dashes"
    )
    assert resolve_external_request_id("17-not-a-uuid", known_ids) is None

    fake_request = type(
        "FakePriorityRequest",
        (),
        {"request_id": "17", "priority": 100, "num_preemptions": 3},
    )()
    fake_scheduler = type(
        "FakeScheduler", (), {"running": [fake_request], "waiting": []}
    )()
    fake_engine = type(
        "FakeEngine",
        (),
        {
            "scheduler": fake_scheduler,
            "scheduler_config": type(
                "FakeSchedulerConfig", (), {"max_num_seqs": 96}
            )(),
        },
    )()
    assert read_vllm_max_num_seqs(fake_engine) == 96
    located = locate_vllm_priority_schedulers(fake_engine)
    assert located and located[0][0] is fake_scheduler
    updated, discovered = update_vllm_request_priorities(
        located, {"17": 1}
    )
    assert updated == {"17"} and discovered == {"17"}
    assert fake_request.priority == 1
    preemptions, observed = read_vllm_request_preemptions(
        located, known_ids
    )
    assert preemptions == {"17": 3} and observed == {"17"}

    stable_priority_a, stable_bucket_a = normal_scheduler_priority(
        arrival_rank=17,
        cohort_size=6,
        predicted_total_tokens=257,
    )
    stable_priority_b, stable_bucket_b = normal_scheduler_priority(
        arrival_rank=17,
        cohort_size=6,
        predicted_total_tokens=257,
    )
    assert stable_priority_a == stable_priority_b
    assert stable_bucket_a == stable_bucket_b
    normal_state = request_priority_state(
        normal_priority=stable_priority_a,
        generated_tokens=0,
        arrival_time=100.0,
        last_token_time=None,
        now=101.0,
        ttft_deadline_sec=2.0,
        itl_rescue_sec=0.5,
    )
    urgent_state = request_priority_state(
        normal_priority=stable_priority_a,
        generated_tokens=0,
        arrival_time=100.0,
        last_token_time=None,
        now=102.1,
        ttft_deadline_sec=2.0,
        itl_rescue_sec=0.5,
    )
    rescue_state = request_priority_state(
        normal_priority=stable_priority_a,
        generated_tokens=10,
        arrival_time=100.0,
        last_token_time=102.0,
        now=102.6,
        ttft_deadline_sec=2.0,
        itl_rescue_sec=0.5,
    )
    unstarted_state = request_priority_state(
        normal_priority=PRIORITY_UNSTARTED_BASE + 17,
        generated_tokens=0,
        arrival_time=100.0,
        last_token_time=None,
        now=101.0,
        ttft_deadline_sec=2.0,
        itl_rescue_sec=0.5,
        normal_state="unstarted",
    )
    assert normal_state == (stable_priority_a, "normal")
    assert urgent_state == (1, "ttft_urgent")
    assert rescue_state == (0, "itl_rescue")
    assert unstarted_state == (
        PRIORITY_UNSTARTED_BASE + 17, "unstarted"
    )

    synthetic_preflights = [
        {
            "load_factor": 0.70,
            "burstiness": 1.0,
            "offered_rps": 1.0,
            "result": {
                "mean_gpu_busy": 0.45,
                "pending_depth_at_arrival_p95": 0.0,
                "queue_nonempty_at_arrival_fraction": 0.10,
                "waiting_at_arrival_fraction": 0.0,
                "shed_fraction": 0.0,
                "system_saturated_fraction": 0.0,
                "ttft_p95": 0.2,
            },
        },
        {
            "load_factor": 0.90,
            "burstiness": 0.5,
            "offered_rps": 1.3,
            "result": {
                "mean_gpu_busy": 0.88,
                "pending_depth_at_arrival_p95": (
                    0.75 * test_capacity
                ),
                "engine_capacity_gpu0": test_capacity,
                "engine_capacity_gpu1": test_capacity,
                "queue_nonempty_at_arrival_fraction": 0.75,
                "waiting_at_arrival_fraction": 0.10,
                "shed_fraction": 0.0,
                "system_saturated_fraction": 0.05,
                "ttft_p95": 3.0,
            },
        },
        {
            "load_factor": 1.10,
            "burstiness": 0.2,
            "offered_rps": 1.6,
            "result": {
                "mean_gpu_busy": 0.99,
                "pending_depth_at_arrival_p95": 100.0,
                "queue_nonempty_at_arrival_fraction": 1.0,
                "waiting_at_arrival_fraction": 0.80,
                "shed_fraction": 0.20,
                "system_saturated_fraction": 0.80,
                "ttft_p95": 12.0,
            },
        },
    ]
    selected, scored = choose_preflight_workload(
        synthetic_preflights,
        queue_wait_budget_sec=5.0,
        ttft_slo_sec=2.0,
    )
    assert selected["load_factor"] == 0.90
    assert len(scored) == 3

    fake_input_batch = type("FakeInputBatch", (), {
        "req_ids": ["req-a", "req-b"],
        "num_reqs": 2,
        "req_id_to_index": {"req-a": 0, "req-b": 1},
    })()
    fake_runner = type("FakeRunner", (), {
        "input_batch": fake_input_batch,
    })()
    assert get_row_owners(fake_runner) == ["req-a", "req-b"], (
        "vLLM V1 InputBatch owner ordering is broken"
    )

    table = [0.0] * (test_capacity + 1)
    table[test_capacity] = 0.03

    short = simulate_local_fcfs_completion(
        active_remaining_tokens=[10.0] * test_capacity,
        queued_jobs=[],
        target_job={"remaining_tokens": 50.0, "prompt_tokens": 100, "generated": 0},
        step_table=table,
        prefill_sec_per_token=0.00005,
        max_concurrent=test_capacity,
    )
    long = simulate_local_fcfs_completion(
        active_remaining_tokens=[200.0] * test_capacity,
        queued_jobs=[],
        target_job={"remaining_tokens": 50.0, "prompt_tokens": 100, "generated": 0},
        step_table=table,
        prefill_sec_per_token=0.00005,
        max_concurrent=test_capacity,
    )
    assert short < long, "Remaining-length simulator failed basic ordering"

    one_ahead = simulate_local_fcfs_completion(
        active_remaining_tokens=[100.0] * test_capacity,
        queued_jobs=[{"remaining_tokens": 10.0, "prompt_tokens": 10, "generated": 0}],
        target_job={"remaining_tokens": 50.0, "prompt_tokens": 100, "generated": 0},
        step_table=table,
        prefill_sec_per_token=0.00005,
        max_concurrent=test_capacity,
    )
    no_queue = simulate_local_fcfs_completion(
        active_remaining_tokens=[100.0] * test_capacity,
        queued_jobs=[],
        target_job={"remaining_tokens": 50.0, "prompt_tokens": 100, "generated": 0},
        step_table=table,
        prefill_sec_per_token=0.00005,
        max_concurrent=test_capacity,
    )
    assert one_ahead >= no_queue, "Queued work must not improve target completion"
    timing = simulate_local_fcfs_completion(
        active_remaining_tokens=[100.0] * test_capacity,
        queued_jobs=[],
        target_job={"remaining_tokens": 50.0, "prompt_tokens": 100, "generated": 0},
        step_table=table,
        prefill_sec_per_token=0.00005,
        max_concurrent=test_capacity,
        return_timing=True,
    )
    assert timing["queue_wait_sec"] > 0
    assert timing["completion_sec"] > timing["queue_wait_sec"]

    gpu0_ids = list(range(test_capacity))
    gpu1_ids = list(range(
        test_capacity, 2 * test_capacity
    ))
    target_id = 2 * test_capacity
    snapshots = [
        {
            "active_ids": gpu0_ids,
            "helix_by_id": {
                rid: (500.0, 1.0) for rid in gpu0_ids
            },
            "forecast_age": 0.0,
            "step_table": table,
            "prefill_spt": DEFAULT_PREFILL_SEC_PER_TOKEN,
            "engine_capacity": test_capacity,
        },
        {
            "active_ids": gpu1_ids,
            "helix_by_id": {
                rid: (5.0, 1.0) for rid in gpu1_ids
            },
            "forecast_age": 0.0,
            "step_table": table,
            "prefill_spt": DEFAULT_PREFILL_SEC_PER_TOKEN,
            "engine_capacity": test_capacity,
        },
    ]
    meta = {
        rid: {"estimated_output_tokens": 10.0, "prompt_len": 10}
        for rid in range(target_id + 1)
    }
    true_lengths = (
        [500] * test_capacity
        + [5] * test_capacity
        + [10]
    )
    chosen, decision = pick_gpu_immediate(
        policy="oracle_work",
        rr_next=0,
        snapshots={0: snapshots[0], 1: snapshots[1]},
        pending_by_gpu=[gpu0_ids, gpu1_ids],
        new_request={
            "req_id": target_id,
            "estimated_output_tokens": 10,
            "prompt_len": 10,
        },
        request_meta=meta,
        generated_all=[0] * (target_id + 1),
        historical_mean=10.0,
        true_lengths=true_lengths,
    )
    assert chosen == 1, (
        "Oracle must choose the replica whose full batch releases a slot first"
    )
    assert decision["scores"][1] < decision["scores"][0]
    assert decision["objective"] == "drain_sec"
    assert decision["timings"][1]["remaining_work_tokens"] < (
        decision["timings"][0]["remaining_work_tokens"]
    )

    chosen, decision = pick_gpu_immediate(
        policy="helix_work",
        rr_next=0,
        snapshots={0: snapshots[0], 1: snapshots[1]},
        pending_by_gpu=[gpu0_ids, gpu1_ids],
        new_request={
            "req_id": target_id,
            "estimated_output_tokens": 10,
            "prompt_len": 10,
        },
        request_meta=meta,
        generated_all=[0] * (target_id + 1),
        historical_mean=100.0,
    )
    assert chosen == 1, (
        "Helix backlog scoring must use predicted active remaining work"
    )
    assert decision["objective"] == "drain_sec"

    # Remaining-work routing may intentionally accept one extra request when
    # the detector sees dramatically less unfinished work on that replica.
    extra_id = target_id + 1
    guarded_target_id = target_id + 2
    guarded_meta = dict(meta)
    guarded_meta[extra_id] = {
        "estimated_output_tokens": 5.0,
        "prompt_len": 10,
    }
    guarded_meta[guarded_target_id] = {
        "estimated_output_tokens": 10.0,
        "prompt_len": 10,
    }
    guarded_generated = [0] * (guarded_target_id + 1)
    chosen, decision = pick_gpu_immediate(
        policy="helix_work",
        rr_next=0,
        snapshots={0: snapshots[0], 1: snapshots[1]},
        pending_by_gpu=[gpu0_ids, gpu1_ids + [extra_id]],
        new_request={
            "req_id": guarded_target_id,
            "estimated_output_tokens": 10,
            "prompt_len": 10,
        },
        request_meta=guarded_meta,
        generated_all=guarded_generated,
        historical_mean=100.0,
    )
    assert decision["work_choice"] == 1, (
        "Guard test must present Helix with a tempting lower-work replica"
    )
    assert chosen == 1 and decision["overrode_queue"], (
        "Helix must be allowed to use remaining work instead of collapsing "
        "back to request count"
    )

    arrivals = gamma_arrival_offsets(
        n_requests=1000,
        request_rate=4.0,
        burstiness=0.5,
        seed=123,
    )
    assert len(arrivals) == 1000 and arrivals[0] == 0.0
    observed_rate = (len(arrivals) - 1) / arrivals[-1]
    assert 3.4 < observed_rate < 4.6, "Gamma arrival rate is mis-scaled"
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:]))
    print("Routing simulator sanity checks passed.")


def _pct_improvement(value, baseline, higher_is_better=False):
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return float('nan')
    if higher_is_better:
        return 100.0 * (value - baseline) / baseline
    return 100.0 * (baseline - value) / baseline


def _fmt(value, decimals=2, suffix=""):
    if value is None:
        return "n/a"
    try:
        finite = bool(np.isfinite(value))
    except (TypeError, ValueError):
        return "n/a"
    if not finite:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _delta(value, baseline, higher=False):
    d = _pct_improvement(value, baseline, higher)
    if not np.isfinite(d):
        return "n/a"
    return f"{d:+.1f}%"


def _legacy_print_policy_card(r):
    print("\n" + "-" * 72)
    print(f"{r['label']} - {r['n_completed']} completed")
    print("-" * 72)
    print(
        f"Latency   p50 {_fmt(r['p50'])}s   p95 {_fmt(r['p95'])}s   "
        f"p99 {_fmt(r['p99'])}s"
    )
    print(
        f"95% CI    p95 [{_fmt(r['p95_ci_low'])}, "
        f"{_fmt(r['p95_ci_high'])}]s   p99 "
        f"[{_fmt(r['p99_ci_low'])}, {_fmt(r['p99_ci_high'])}]s"
    )
    print(
        f"TTFT      p50 {_fmt(r['ttft_p50'])}s   p95 {_fmt(r['ttft_p95'])}s   "
        f"p99 {_fmt(r['ttft_p99'])}s"
    )
    print(
        f"ITL       p50 {_fmt(r['itl_p50'] * 1000, 1)}ms   "
        f"p95 {_fmt(r['itl_p95'] * 1000, 1)}ms   "
        f"p99 {_fmt(r['itl_p99'] * 1000, 1)}ms"
    )
    print(
        f"Capacity  {_fmt(r['tokens_per_sec'], 1)} tok/s   "
        f"GPU busy {_fmt(r['mean_gpu_busy'] * 100, 1)}%   "
        f"decode occupancy {_fmt(r['mean_slot_occupancy'] * 100, 1)}%"
    )
    print(
        f"Routing   assignments {r['assignment_count']}   "
        f"max local pending {r['max_pending']}   "
        f"queued on chosen worker {_fmt(r['queued_on_chosen_fraction'] * 100, 1)}%   "
        f"work overrides {r['work_override_count']}"
    )


def _legacy_print_comparison_report(results, baseline_policy="queue_size"):
    baseline = next(r for r in results if r["policy"] == baseline_policy)
    ordered = sorted(results, key=lambda r: r["p95"])

    print("\n" + "=" * 114)
    print("FINAL COMPARISON - immediate dispatch; each GPU owns an independent vLLM FCFS queue")
    print("Improvements are relative to Queue size. Lower latency/TTFT/ITL is better; higher tok/s is better.")
    print("=" * 114)

    print("\nRANKED USER EXPERIENCE")
    print(
        f"{'#':>2}  {'Policy':<18} {'p50':>8} {'p95':>8} {'p99':>8} "
        f"{'TTFT95':>9} {'ITL95':>9} {'tok/s':>9} {'p95 Î”':>9}"
    )
    print("-" * 94)
    for rank, r in enumerate(ordered, 1):
        print(
            f"{rank:>2}  {r['label']:<18} "
            f"{r['p50']:>7.2f}s {r['p95']:>7.2f}s {r['p99']:>7.2f}s "
            f"{r['ttft_p95']:>8.2f}s {r['itl_p95']*1000:>8.1f}ms "
            f"{r['tokens_per_sec']:>9.1f} {_delta(r['p95'], baseline['p95']):>9}"
        )

    print("\nHEAD-TO-HEAD VS QUEUE SIZE")
    print(
        f"{'Policy':<18} {'p50 Î”':>10} {'p95 Î”':>10} {'p99 Î”':>10} "
        f"{'TTFT95 Î”':>11} {'ITL95 Î”':>10} {'tok/s Î”':>10}"
    )
    print("-" * 84)
    for r in results:
        if r["policy"] == baseline_policy:
            continue
        print(
            f"{r['label']:<18} "
            f"{_delta(r['p50'], baseline['p50']):>10} "
            f"{_delta(r['p95'], baseline['p95']):>10} "
            f"{_delta(r['p99'], baseline['p99']):>10} "
            f"{_delta(r['ttft_p95'], baseline['ttft_p95']):>11} "
            f"{_delta(r['itl_p95'], baseline['itl_p95']):>10} "
            f"{_delta(r['tokens_per_sec'], baseline['tokens_per_sec'], True):>10}"
        )

    print("\nLOAD BALANCE AND DIAGNOSTICS")
    print(
        f"{'Policy':<18} {'Assignments':>14} {'Max pending':>15} "
        f"{'GPU busy':>10} {'Decode occ':>11} {'Hook skip':>10} {'Hook err':>10}"
    )
    print("-" * 96)
    for r in results:
        print(
            f"{r['label']:<18} {str(r['assignment_count']):>14} "
            f"{str(r['max_pending']):>15} "
            f"{r['mean_gpu_busy']*100:>9.1f}% "
            f"{r['mean_slot_occupancy']*100:>10.1f}% "
            f"{r['hook_skip_ratio']*100:>9.2f}% "
            f"{r['hook_true_error_ratio']*100:>9.3f}%"
        )

    helix = next(r for r in results if r["policy"] == "helix_work")
    oracle = next(r for r in results if r["policy"] == "oracle_work")

    print("\nINTERPRETATION")
    print(
        f"- Remaining prediction headroom, Oracle vs Helix: "
        f"p50 {_delta(oracle['p50'], helix['p50'])}, "
        f"p95 {_delta(oracle['p95'], helix['p95'])}, "
        f"p99 {_delta(oracle['p99'], helix['p99'])}."
    )
    if oracle["p95"] >= baseline["p95"] and oracle["p50"] >= baseline["p50"]:
        print(
            "- WARNING: Oracle did not beat Queue size on p50 or p95. Do not conclude "
            "that length is useless; this means the local service-time model is "
            "still miscalibrated for this vLLM build/workload. Inspect the decode "
            "step buckets and prefill estimate before interpreting Helix results."
        )
    else:
        print(
            "- Oracle beats Queue size on at least one central latency metric, so "
            "the benchmark exposes measurable value from remaining-work knowledge."
        )
    if helix["tracker_slot_exhaustions"]:
        print(
            f"- WARNING: Tracker slots were exhausted {helix['tracker_slot_exhaustions']} "
            "times. This indicates vLLM preemption/resumption or stale tracker "
            "ownership and invalidates some Helix forecasts."
        )


def print_policy_card(r):
    print("\n" + "-" * 88)
    print(
        f"{r['label']} - {r['n_completed']} completed, "
        f"{r['n_rejected']} shed"
    )
    print("-" * 88)
    print(
        f"Traffic   {r['arrival_mode']}   offered "
        f"{_fmt(r['offered_rps'], 2)} req/s   load "
        f"{_fmt(r['load_factor'], 2)}   burstiness "
        f"{_fmt(r['burstiness'], 2)}"
    )
    print(
        f"Latency   p50 {_fmt(r['p50'])}s   p95 {_fmt(r['p95'])}s   "
        f"p99 {_fmt(r['p99'])}s"
    )
    print(
        f"95% CI    p95 [{_fmt(r['p95_ci_low'])}, "
        f"{_fmt(r['p95_ci_high'])}]s   p99 "
        f"[{_fmt(r['p99_ci_low'])}, {_fmt(r['p99_ci_high'])}]s"
    )
    print(
        f"TTFT      p50 {_fmt(r['ttft_p50'])}s   "
        f"p95 {_fmt(r['ttft_p95'])}s   p99 {_fmt(r['ttft_p99'])}s"
    )
    print(
        f"ITL       p50 {_fmt(r['itl_p50'] * 1000, 1)}ms   "
        f"p95 {_fmt(r['itl_p95'] * 1000, 1)}ms   "
        f"p99 {_fmt(r['itl_p99'] * 1000, 1)}ms"
    )
    print(
        f"Capacity  {_fmt(r['tokens_per_sec'], 1)} output tok/s   "
        f"{_fmt(r['accepted_requests_per_sec'], 2)} accepted req/s   "
        f"goodput {_fmt(r['goodput_requests_per_sec'], 2)} req/s   "
        f"shed {_fmt(r['shed_fraction'] * 100, 2)}%"
    )
    print(
        f"Queue     measured p50 {_fmt(r['scheduler_queue_wait_p50'], 3)}s   "
        f"p95 {_fmt(r['scheduler_queue_wait_p95'], 3)}s   "
        f"predicted p95 {_fmt(r['predicted_queue_wait_p95'], 3)}s   "
        f"max waiting {r['max_waiting']}"
    )
    print(
        f"Routing   assignments {r['assignment_count']}   "
        f"max pending {r['max_pending']}   "
        f"engine capacity "
        f"[{r['engine_capacity_gpu0']}, {r['engine_capacity_gpu1']}]"
    )
    print(
        f"Work      candidates disagreed "
        f"{r['work_candidate_disagreement_count']} times   "
        f"overrides {r['work_override_count']} "
        f"({_fmt(r['work_override_fraction'] * 100, 1)}%)   "
        f"mean predicted gain "
        f"{_fmt(r['routing_predicted_gain_mean_sec'], 3)}s"
    )
    if r.get("priority_scheduler_enabled"):
        print(
            f"Priority  deadline "
            f"{_fmt(r['priority_ttft_deadline_sec'], 3)}s   "
            f"ITL rescue {_fmt(r['priority_itl_rescue_sec'], 3)}s   "
            f"updates/request {_fmt(r['priority_updates_mean'], 2)}   "
            f"state changes/request "
            f"{_fmt(r['priority_transitions_mean'], 2)}   "
            f"preemption coverage "
            f"{_fmt(r['scheduler_preemptions_coverage'] * 100, 1)}%"
        )
    if r["policy"] == "helix_work":
        print(
            f"Helix     active snapshots "
            f"{_fmt(r['helix_active_snapshot_fraction'] * 100, 1)}%   "
            f"fresh {_fmt(r['helix_fresh_active_snapshot_fraction'] * 100, 1)}%   "
            f"informative "
            f"{_fmt(r['helix_informative_estimate_fraction'] * 100, 1)}%   "
            f"valid {r['helix_snapshot_valid']}"
        )


def print_comparison_report(results, baseline_policy="queue_size"):
    baseline = next(r for r in results if r["policy"] == baseline_policy)
    ordered = sorted(results, key=lambda r: r["p95"])
    print("\n" + "=" * 122)
    print("FINAL COMPARISON - warmed replicas, open-loop arrivals, bounded admission")
    print(
        f"Load={baseline.get('load_factor')}  "
        f"offered={baseline.get('offered_rps'):.2f} req/s  "
        f"arrival={baseline.get('arrival_mode')}  "
        f"burstiness={baseline.get('burstiness')}"
    )
    print("All deltas are relative to Queue size.")
    print("=" * 122)
    print(
        f"{'#':>2}  {'Policy':<18} {'p50':>8} {'p95':>8} {'p99':>8} "
        f"{'TTFT95':>9} {'ITL95':>9} {'tok/s':>9} {'shed':>8} {'p95 d':>9}"
    )
    print("-" * 108)
    for rank, r in enumerate(ordered, 1):
        print(
            f"{rank:>2}  {r['label']:<18} "
            f"{r['p50']:>7.2f}s {r['p95']:>7.2f}s {r['p99']:>7.2f}s "
            f"{r['ttft_p95']:>8.2f}s {r['itl_p95']*1000:>8.1f}ms "
            f"{r['tokens_per_sec']:>9.1f} {r['shed_fraction']*100:>7.2f}% "
            f"{_delta(r['p95'], baseline['p95']):>9}"
        )

    print("\nHEAD-TO-HEAD VS QUEUE SIZE")
    for r in results:
        if r["policy"] == baseline_policy:
            continue
        print(
            f"{r['label']:<18} "
            f"p50 {_delta(r['p50'], baseline['p50'])}, "
            f"p95 {_delta(r['p95'], baseline['p95'])}, "
            f"p99 {_delta(r['p99'], baseline['p99'])}, "
            f"TTFT95 {_delta(r['ttft_p95'], baseline['ttft_p95'])}, "
            f"tok/s {_delta(r['tokens_per_sec'], baseline['tokens_per_sec'], True)}, "
            f"shed {r['shed_fraction']*100:.2f}%"
        )
        print(
            f"{'':18} paired p95 delta "
            f"{_fmt(r.get('paired_p95_delta_sec'), 3)}s "
            f"95% CI [{_fmt(r.get('paired_p95_delta_ci_low'), 3)}, "
            f"{_fmt(r.get('paired_p95_delta_ci_high'), 3)}]s; "
            f"paired TTFT95 delta "
            f"{_fmt(r.get('paired_ttft_p95_delta_sec'), 3)}s"
        )


def aggregate_seed_results(results):
    metrics = (
        "p50", "p95", "p99", "ttft_p95", "itl_p95", "max_itl_p95",
        "tokens_per_sec", "accepted_requests_per_sec",
        "goodput_requests_per_sec", "shed_fraction",
        "scheduler_preemptions_mean", "priority_updates_mean",
        "priority_transitions_mean",
        "work_override_fraction",
        "work_override_acceptance_fraction",
        "routing_predicted_gain_mean_sec",
        "helix_active_snapshot_fraction",
        "helix_informative_estimate_fraction",
        "paired_p95_delta_sec", "paired_ttft_p95_delta_sec",
    )
    grouped = {}
    for result in results:
        key = (
            result.get("arrival_mode"),
            result.get("load_factor"),
            result.get("burstiness"),
            result.get("policy"),
        )
        grouped.setdefault(key, []).append(result)

    aggregates = []
    for (mode, load, burstiness, policy), rows in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        rec = {
            "arrival_mode": mode,
            "load_factor": load,
            "burstiness": burstiness,
            "policy": policy,
            "n_seeds": len(rows),
        }
        for metric in metrics:
            values = np.asarray([
                r.get(metric, float("nan")) for r in rows
            ], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                rec[f"{metric}_mean"] = float("nan")
                rec[f"{metric}_ci95"] = float("nan")
                continue
            rec[f"{metric}_mean"] = float(np.mean(values))
            rec[f"{metric}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1 else float("nan")
            )
        aggregates.append(rec)
    return aggregates


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(results, directory=".", experiment_config=None):
    import csv
    import json
    from pathlib import Path

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / RESULTS_JSON
    csv_path = out_dir / RESULTS_CSV
    request_csv_path = out_dir / REQUEST_RESULTS_CSV
    config_path = out_dir / EXPERIMENT_CONFIG_JSON
    aggregate_json_path = out_dir / AGGREGATE_RESULTS_JSON
    aggregate_csv_path = out_dir / AGGREGATE_RESULTS_CSV

    serializable = []
    for r in results:
        rec = {
            k: v for k, v in r.items()
            if k not in ("details", "rejections")
        }
        serializable.append(rec)
    json_path.write_text(
        json.dumps(json_safe(serializable), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    fields = [
        "experiment_id", "policy", "label", "helix_tracker_enabled",
        "arrival_mode", "arrival_seed",
        "burstiness", "offered_rps", "load_factor", "queue_wait_budget_sec",
        "p50", "p95", "p99",
        "p95_ci_low", "p95_ci_high", "p99_ci_low", "p99_ci_high",
        "paired_request_count", "paired_p95_delta_sec",
        "paired_p95_delta_ci_low", "paired_p95_delta_ci_high",
        "paired_ttft_p95_delta_sec",
        "paired_ttft_p95_delta_ci_low",
        "paired_ttft_p95_delta_ci_high",
        "ttft_p50", "ttft_p95", "ttft_p99",
        "ttft_p95_ci_low", "ttft_p95_ci_high",
        "itl_p50", "itl_p95", "itl_p99", "max_itl_p95", "max_itl_max",
        "priority_scheduler_enabled", "priority_ttft_deadline_sec",
        "priority_itl_rescue_sec", "priority_cohort_size",
        "scheduler_preemptions_mean", "scheduler_preemptions_coverage",
        "priority_updates_mean", "priority_transitions_mean",
        "priority_lookup_misses",
        "tokens_per_sec", "prompt_tokens_per_sec",
        "accepted_requests_per_sec", "goodput_requests_per_sec",
        "goodput_fraction", "shed_fraction",
        "predicted_queue_wait_p50", "predicted_queue_wait_p95",
        "predicted_queue_wait_p99",
        "engine_queue_wait_p50", "engine_queue_wait_p95",
        "engine_queue_wait_p99", "engine_queue_metrics_coverage",
        "scheduler_queue_wait_p50", "scheduler_queue_wait_p95",
        "scheduler_queue_wait_p99", "scheduler_queue_metrics_coverage",
        "n_submitted", "n_accepted", "n_rejected", "n_measured_completed",
        "mean_prompt_tokens", "mean_output_tokens",
        "reference_length_comparisons",
        "reference_length_mismatch_fraction",
        "mean_abs_reference_length_error_tokens",
        "mean_gpu_busy", "mean_slot_occupancy",
        "engine_capacity_gpu0", "engine_capacity_gpu1",
        "mean_engine_capacity",
        "assignment_gpu0", "assignment_gpu1",
        "max_pending_gpu0", "max_pending_gpu1",
        "pending_depth_at_arrival_mean",
        "pending_depth_at_arrival_p95",
        "waiting_depth_at_arrival_p95",
        "engine_waiting_depth_at_arrival_p95",
        "engine_running_depth_at_arrival_p95",
        "active_depth_at_arrival_mean",
        "queue_nonempty_at_arrival_fraction",
        "waiting_at_arrival_fraction",
        "queued_on_chosen_fraction", "system_saturated_fraction",
        "mean_score_margin_sec", "mean_helix_confidence",
        "helix_active_snapshot_fraction",
        "helix_fresh_active_snapshot_fraction",
        "helix_estimate_coverage",
        "helix_informative_estimate_fraction", "helix_snapshot_valid",
        "hook_skip_ratio", "hook_true_error_ratio",
        "tracker_slot_exhaustions", "work_override_count",
        "work_override_fraction", "work_candidate_disagreement_count",
        "work_override_acceptance_fraction", "routing_reason_counts",
        "routing_predicted_gain_mean_sec",
        "routing_predicted_gain_p95_sec",
        "routing_predicted_gain_mean_tokens",
        "routing_score_gpu0_mean_sec", "routing_score_gpu1_mean_sec",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in serializable:
            writer.writerow({k: r.get(k) for k in fields})

    request_fields = [
        "experiment_id", "policy", "load_factor", "arrival_seed",
        "arrival_mode", "burstiness", "status", "req_id", "gpu_id",
        "traffic_class", "measure", "arrival_time", "scheduled_arrival_time",
        "finish_time", "latency_sec", "ttft_sec", "n_tokens",
        "prompt_len", "estimated_output_tokens", "output_prior_source",
        "predicted_queue_wait_sec", "engine_queue_time_sec",
        "scheduler_queue_time_sec", "scheduler_preemptions",
        "scheduler_priority_initial", "scheduler_priority_final",
        "scheduler_priority_updates", "scheduler_priority_transitions",
        "scheduler_priority_state_final", "priority_length_stage",
        "priority_length_bucket", "priority_predicted_total_tokens",
        "priority_lookup_misses", "reason",
        "predicted_wait_gpu0_sec", "predicted_wait_gpu1_sec",
        "routing_reason", "routing_queue_choice", "routing_work_choice",
        "routing_overrode_queue", "routing_predicted_gain_sec",
        "routing_predicted_gain_tokens",
        "routing_score_gpu0_sec", "routing_score_gpu1_sec",
        "routing_gpu0_active_work_tokens",
        "routing_gpu1_active_work_tokens",
        "routing_gpu0_total_work_tokens",
        "routing_gpu1_total_work_tokens",
    ]
    with request_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=request_fields)
        writer.writeheader()
        for result in results:
            common = {
                "experiment_id": result.get("experiment_id"),
                "policy": result["policy"],
                "load_factor": result.get("load_factor"),
                "arrival_seed": result.get("arrival_seed"),
                "arrival_mode": result.get("arrival_mode"),
                "burstiness": result.get("burstiness"),
            }
            for rec in result.get("details", []):
                row = {
                    **common,
                    "status": "completed",
                    **{k: rec.get(k) for k in request_fields if k in rec},
                    "latency_sec": rec["finish_time"] - rec["arrival_time"],
                    "ttft_sec": rec.get("ttft"),
                }
                writer.writerow({k: row.get(k) for k in request_fields})
            for rec in result.get("rejections", []):
                row = {
                    **common,
                    "status": "shed",
                    **{k: rec.get(k) for k in request_fields if k in rec},
                }
                writer.writerow({k: row.get(k) for k in request_fields})

    if experiment_config is not None:
        config_path.write_text(
            json.dumps(
                json_safe(experiment_config),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    aggregates = aggregate_seed_results(results)
    aggregate_json_path.write_text(
        json.dumps(json_safe(aggregates), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    aggregate_fields = sorted({
        key for rec in aggregates for key in rec
    })
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    print(
        f"\nSaved summary metrics to {json_path} and {csv_path}\n"
        f"Saved per-request outcomes to {request_csv_path}\n"
        f"Saved cross-seed aggregates to {aggregate_csv_path}"
    )


def parse_float_list(text):
    values = tuple(float(v.strip()) for v in str(text).split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value")
    return values


def parse_int_list(text):
    values = tuple(int(v.strip()) for v in str(text).split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    return values


def parse_policy_list(text):
    aliases = {
        "round_robin": "round_robin",
        "round-robin": "round_robin",
        "rr": "round_robin",
        "queue_size": "queue_size",
        "queue-size": "queue_size",
        "queue": "queue_size",
        "helix_work": "helix_work",
        "helix": "helix_work",
        "oracle_work": "oracle_work",
        "oracle": "oracle_work",
    }
    values = []
    for raw in str(text).split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unknown policy {raw!r}; choose queue_size, helix_work, "
                "round_robin, or oracle_work"
            )
        value = aliases[key]
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one policy")
    return tuple(values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Production-shaped two-replica Helix routing benchmark. Capacity "
            "is measured on the current hardware before normalized loads run."
        )
    )
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=DEFAULT_BENCHMARK_REQUESTS,
        help="Requests per policy and traffic condition (default: 600).",
    )
    parser.add_argument(
        "--policies",
        type=parse_policy_list,
        default=(
            "round_robin", "queue_size", "helix_work", "oracle_work"
        ),
        help=(
            "Comma-separated policies to run. queue_size is required as the "
            "paired baseline; oracle_work also requires round_robin."
        ),
    )
    parser.add_argument(
        "--capacity-rps",
        type=float,
        default=None,
        help=(
            "Reuse a previously printed saturated request capacity and skip "
            "the capacity probe during iteration."
        ),
    )
    parser.add_argument(
        "--calibration-cache",
        default=None,
        help=(
            "Path for reusable Helix basis/trace calibration. Defaults to a "
            "quick- or full-mode cache in the output directory."
        ),
    )
    parser.add_argument(
        "--rebuild-calibration",
        action="store_true",
        help="Ignore and replace an existing calibration cache.",
    )
    parser.add_argument(
        "--capacity-probe-requests",
        type=int,
        default=DEFAULT_CAPACITY_PROBE_REQUESTS,
        help=(
            "Requests in the saturated hardware-capacity calibration "
            "(default: 64)."
        ),
    )
    parser.add_argument(
        "--preflight-requests",
        type=int,
        default=DEFAULT_PREFLIGHT_REQUESTS,
        help="Representative requests per automatic traffic probe (default: 32).",
    )
    parser.add_argument(
        "--preflight-load-levels",
        type=parse_float_list,
        default=DEFAULT_PREFLIGHT_LOAD_LEVELS,
        help=(
            "Paired capacity fractions for automatic traffic calibration "
            "(default: 0.75,0.90,1.00)."
        ),
    )
    parser.add_argument(
        "--preflight-burstiness-levels",
        type=parse_float_list,
        default=DEFAULT_PREFLIGHT_BURSTINESS_LEVELS,
        help=(
            "Paired Gamma shapes for automatic traffic calibration "
            "(default: 1.0,0.5,0.25)."
        ),
    )
    parser.add_argument(
        "--manual-workload",
        action="store_true",
        help=(
            "Skip automatic traffic probes and run the Cartesian product of "
            "--load-levels and --burstiness-levels."
        ),
    )
    parser.add_argument(
        "--load-levels",
        type=parse_float_list,
        default=DEFAULT_LOAD_LEVELS,
        help="Comma-separated fractions of measured saturated capacity.",
    )
    parser.add_argument(
        "--burstiness-levels",
        type=parse_float_list,
        default=(DEFAULT_BURSTINESS,),
        help=(
            "Gamma shape values. 1=Poisson; values below 1 are burstier. "
            "Use 1.0,0.5,0.2 for a full sensitivity sweep."
        ),
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("gamma", "trace"),
        default="gamma",
    )
    parser.add_argument(
        "--trace-path",
        default=os.getenv("BURSTGPT_TRACE_PATH"),
        help="BurstGPT CSV path when --arrival-mode=trace.",
    )
    parser.add_argument("--trace-start-row", type=int, default=0)
    parser.add_argument(
        "--trace-log-type",
        default=None,
        help="Optional exact BurstGPT Log Type filter.",
    )
    parser.add_argument(
        "--queue-wait-budget-sec",
        type=float,
        default=DEFAULT_QUEUE_WAIT_BUDGET_SEC,
    )
    parser.add_argument(
        "--ttft-slo-sec",
        type=float,
        default=DEFAULT_TTFT_SLO_SEC,
    )
    parser.add_argument(
        "--itl-slo-ms",
        type=float,
        default=DEFAULT_ITL_SLO_SEC * 1000.0,
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=(POLICY_SEED,),
        help="Arrival seeds. Use 0,1,2,3,4 for final confidence intervals.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run only queue_size and helix_work with at most 96 policy "
            "requests and 24 capacity requests; skip traffic preflights and "
            "use a smaller separately cached calibration."
        ),
    )
    parser.add_argument(
        "--quick-all-policies",
        action="store_true",
        help=(
            "Use the quick workload/calibration but run round_robin, "
            "queue_size, helix_work, and oracle_work. This is the fastest "
            "real-vLLM Oracle/headroom check."
        ),
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run CPU-only routing regression checks and exit before loading data or models.",
    )
    parser.add_argument(
        "--measure-overhead",
        action="store_true",
        help=(
            "Also run a no-detector saturated control. This doubles capacity "
            "calibration time and is off by default."
        ),
    )
    parser.add_argument(
        "--skip-overhead-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.quick_all_policies:
        args.quick = True
        args.policies = (
            "round_robin", "queue_size", "helix_work", "oracle_work"
        )
    if args.benchmark_requests <= 0:
        parser.error("--benchmark-requests must be positive")
    if args.capacity_probe_requests <= 0:
        parser.error("--capacity-probe-requests must be positive")
    if args.preflight_requests <= 0:
        parser.error("--preflight-requests must be positive")
    if any(v <= 0 for v in args.load_levels):
        parser.error("--load-levels values must be positive")
    if any(v <= 0 for v in args.burstiness_levels):
        parser.error("--burstiness-levels values must be positive")
    if any(v <= 0 for v in args.preflight_load_levels):
        parser.error("--preflight-load-levels values must be positive")
    if any(v <= 0 for v in args.preflight_burstiness_levels):
        parser.error("--preflight-burstiness-levels values must be positive")
    if (
        len(args.preflight_load_levels)
        != len(args.preflight_burstiness_levels)
    ):
        parser.error(
            "--preflight-load-levels and --preflight-burstiness-levels "
            "must contain the same number of paired values"
        )
    if args.queue_wait_budget_sec <= 0:
        parser.error("--queue-wait-budget-sec must be positive")
    if args.capacity_rps is not None and args.capacity_rps <= 0:
        parser.error("--capacity-rps must be positive")
    if "queue_size" not in args.policies:
        parser.error("--policies must include queue_size for paired comparison")
    if (
        "oracle_work" in args.policies
        and "round_robin" not in args.policies
    ):
        parser.error("oracle_work requires round_robin to establish true lengths")
    if args.arrival_mode == "trace" and not args.trace_path:
        parser.error("--trace-path is required for trace arrivals")
    if args.quick:
        args.benchmark_requests = min(args.benchmark_requests, 96)
        args.capacity_probe_requests = min(args.capacity_probe_requests, 24)
        args.preflight_requests = min(args.preflight_requests, 16)
        args.load_levels = (DEFAULT_TARGET_UTILIZATION,)
        args.burstiness_levels = (DEFAULT_BURSTINESS,)
        args.seeds = (args.seeds[0],)
        args.manual_workload = True
        # Fast iteration answers the important question first: does Helix beat
        # the unchanged queue-size baseline? Oracle and RR remain in full runs.
        if not args.quick_all_policies:
            args.policies = ("queue_size", "helix_work")
    if args.skip_overhead_probe:
        args.measure_overhead = False
    return args


def formatted_prompt_length(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(text))


def load_production_shaped_prompt_pool(load_dataset, tokenizer, requested):
    """Build a heterogeneous natural-EOS workload without fake cache reuse."""
    calibration_needed = N_CALIB_PROMPTS + N_TRACE_PROMPTS
    # Oversample so calibration can use memory-safe short inputs while routing
    # still retains the original long-context distribution.
    selection_pool_needed = requested + 4 * calibration_needed
    per_large_class = max(1000, math.ceil(selection_pool_needed * 0.50))

    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
    ds_chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    ds_swe = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    records = [
        {"prompt": row["question"], "traffic_class": "math_reasoning"}
        for row in ds_gsm.select(range(min(len(ds_gsm), per_large_class)))
    ]
    records.extend([
        {
            "prompt": row["messages"][0]["content"],
            "traffic_class": "conversation",
        }
        for row in ds_chat.select(range(min(len(ds_chat), per_large_class)))
    ])
    records.extend([
        {
            "prompt": row["problem_statement"],
            "traffic_class": "software_engineering",
        }
        for row in ds_swe
    ])

    max_allowed_input = MAX_MODEL_LEN - MAX_NEW_TOKENS - 50
    filtered = []
    for rec in records:
        token_len = formatted_prompt_length(tokenizer, rec["prompt"])
        if token_len <= max_allowed_input:
            filtered.append({**rec, "prompt_len": token_len})

    random.Random(42).shuffle(filtered)
    if len(filtered) < selection_pool_needed:
        raise RuntimeError(
            f"Only {len(filtered)} prompts fit the {MAX_MODEL_LEN}-token "
            f"context; {selection_pool_needed} are required. Reduce "
            "--benchmark-requests or increase HELIX_MAX_MODEL_LEN."
        )
    return filtered[:selection_pool_needed]


def build_arrival_schedule(args, n_requests, request_rate, burstiness, seed):
    if args.arrival_mode == "trace":
        return trace_arrival_offsets(
            csv_path=args.trace_path,
            n_requests=n_requests,
            request_rate=request_rate,
            start_row=args.trace_start_row + seed * n_requests,
            log_type=args.trace_log_type,
        )
    return gamma_arrival_offsets(
        n_requests=n_requests,
        request_rate=request_rate,
        burstiness=burstiness,
        seed=seed,
    )


def workload_probe_score(
    result,
    queue_wait_budget_sec,
    ttft_slo_sec,
):
    """Lower is a better production-shaped operating point.

    The target is sustained GPU use with visible (but bounded) queueing. Empty
    queues do not exercise routing; persistent saturation mostly measures
    admission control rather than routing quality.
    """
    busy = float(result.get("mean_gpu_busy", 0.0))
    pending_p95 = float(
        result.get("pending_depth_at_arrival_p95", 0.0)
    )
    queue_fraction = float(
        result.get("queue_nonempty_at_arrival_fraction", 0.0)
    )
    waiting_fraction = float(
        result.get("waiting_at_arrival_fraction", 0.0)
    )
    shed_fraction = float(result.get("shed_fraction", 0.0))
    saturated_fraction = float(
        result.get("system_saturated_fraction", 0.0)
    )
    ttft_p95 = float(result.get("ttft_p95", float("nan")))
    if not np.isfinite(ttft_p95):
        ttft_p95 = float(queue_wait_budget_sec)

    target_busy = 0.88
    total_engine_capacity = max(
        2.0,
        float(result.get("engine_capacity_gpu0", 0.0))
        + float(result.get("engine_capacity_gpu1", 0.0)),
    )
    target_pending = max(4.0, math.sqrt(total_engine_capacity))
    target_ttft = max(
        float(ttft_slo_sec),
        min(3.0, 0.60 * float(queue_wait_budget_sec)),
    )

    score = abs(busy - target_busy) / 0.12
    score += abs(
        math.log1p(max(0.0, pending_p95))
        - math.log1p(target_pending)
    )
    score += abs(
        math.log(max(0.05, ttft_p95) / max(0.05, target_ttft))
    )
    score += 2.0 * abs(queue_fraction - 0.75)
    score += 1.5 * abs(waiting_fraction - 0.10)
    score += 30.0 * shed_fraction
    score += 8.0 * max(0.0, saturated_fraction - 0.25)
    if busy < 0.65:
        score += 3.0
    if ttft_p95 > float(queue_wait_budget_sec):
        score += 2.0 * (
            ttft_p95 / float(queue_wait_budget_sec) - 1.0
        )
    return float(score)


def choose_preflight_workload(
    candidates,
    queue_wait_budget_sec,
    ttft_slo_sec,
):
    if not candidates:
        raise ValueError("At least one preflight candidate is required")

    scored = []
    for candidate in candidates:
        score = workload_probe_score(
            candidate["result"],
            queue_wait_budget_sec=queue_wait_budget_sec,
            ttft_slo_sec=ttft_slo_sec,
        )
        scored.append({**candidate, "selection_score": score})

    zero_shed = [
        candidate for candidate in scored
        if float(candidate["result"].get("shed_fraction", 0.0)) == 0.0
    ]
    eligible = zero_shed or scored
    winner = min(eligible, key=lambda candidate: candidate["selection_score"])
    return winner, scored


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    args = parse_args(argv)
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except (ImportError, ModuleNotFoundError):
        pass

    mp.set_start_method("spawn", force=True)
    run_routing_sanity_checks()
    if args.sanity_only:
        print("Sanity-only validation completed; no models were loaded.")
        return

    print("Loading production-shaped natural-EOS workload...")
    calib_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if calib_tokenizer.pad_token is None:
        calib_tokenizer.pad_token = calib_tokenizer.eos_token

    records = load_production_shaped_prompt_pool(
        load_dataset=load_dataset,
        tokenizer=calib_tokenizer,
        requested=args.benchmark_requests,
    )
    calibration_prompt_count = 12 if args.quick else N_CALIB_PROMPTS
    trace_prompt_count = 12 if args.quick else N_TRACE_PROMPTS
    calibration_count = calibration_prompt_count + trace_prompt_count
    calibration_indices = [
        idx for idx, record in enumerate(records)
        if record["prompt_len"] <= CALIBRATION_MAX_INPUT_TOKENS
    ][:calibration_count]
    if len(calibration_indices) < calibration_count:
        raise RuntimeError(
            f"Only {len(calibration_indices)} prompts fit the "
            f"{CALIBRATION_MAX_INPUT_TOKENS}-token offline calibration limit; "
            f"{calibration_count} are required. Increase "
            "HELIX_CALIBRATION_MAX_INPUT_TOKENS cautiously."
        )
    calibration_index_set = set(calibration_indices)
    calibration_records = [records[idx] for idx in calibration_indices]
    calib_records = calibration_records[:calibration_prompt_count]
    trace_records = calibration_records[calibration_prompt_count:]
    routing_records = [
        record for idx, record in enumerate(records)
        if idx not in calibration_index_set
    ][:args.benchmark_requests]
    if len(routing_records) < args.benchmark_requests:
        raise RuntimeError("Insufficient disjoint records for the routing workload")

    calib_prompts = [r["prompt"] for r in calib_records]
    trace_prompts = [r["prompt"] for r in trace_records]
    routing_prompts = [r["prompt"] for r in routing_records]
    prompt_lengths = [r["prompt_len"] for r in routing_records]
    traffic_classes = [r["traffic_class"] for r in routing_records]
    generation_params = [
        {"max_tokens": MAX_NEW_TOKENS, "temperature": 0.0}
        for _ in routing_records
    ]

    print(
        f"Using {len(routing_prompts)} benchmark requests with context "
        f"limit {MAX_MODEL_LEN}; max_num_seqs is "
        + (
            f"explicitly overridden to {MAX_CONCURRENT_SEQS_OVERRIDE}."
            if MAX_CONCURRENT_SEQS_OVERRIDE is not None
            else "discovered from each vLLM engine."
        )
    )
    print(
        f"Offline calibration inputs are capped at "
        f"{CALIBRATION_MAX_INPUT_TOKENS} tokens; routed inputs are not."
    )
    for label in sorted(set(traffic_classes)):
        count = sum(v == label for v in traffic_classes)
        print(f"  {label}: {count}")

    calibration_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "version": CALIBRATION_CACHE_VERSION,
                "model_id": MODEL_ID,
                "target_layer": TARGET_LAYER,
                "max_new_tokens": MAX_NEW_TOKENS,
                "calibration_prompts": calib_prompts,
                "trace_prompts": trace_prompts,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    calibration_output_dir = (
        "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    )
    calibration_cache_path = args.calibration_cache or os.path.join(
        calibration_output_dir,
        "helix_calibration_quick.pkl"
        if args.quick else "helix_calibration_full.pkl",
    )
    calibration_payload = None
    if not args.rebuild_calibration and os.path.isfile(calibration_cache_path):
        try:
            with open(calibration_cache_path, "rb") as handle:
                candidate = pickle.load(handle)
            if candidate.get("fingerprint") == calibration_fingerprint:
                calibration_payload = candidate
                print(
                    f"Reusing Helix calibration from "
                    f"{calibration_cache_path}."
                )
            else:
                print(
                    "Existing calibration cache does not match this workload; "
                    "rebuilding it."
                )
        except Exception as exc:
            print(f"Could not reuse calibration cache ({exc}); rebuilding it.")

    if calibration_payload is None:
        print("Building Helix basis...")
        calib_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map="cuda:0"
        )
        cache = build_helix_basis_cache(
            calib_model,
            calib_tokenizer,
            calib_prompts,
            target_layer=TARGET_LAYER,
            max_new_tokens=MAX_NEW_TOKENS,
            n_prompts=len(calib_prompts),
            n_contam=N_CONTAM,
        )
        # Release allocator cache from repeated basis generations before the
        # independent trace pass. Live model weights remain resident.
        torch.cuda.empty_cache()

        print("\nCollecting independent historical traces...")
        traces = collect_offline_traces_both(
            calib_model,
            calib_tokenizer,
            trace_prompts,
            cache,
            target_layer=TARGET_LAYER,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        calib_params = calibrate_thresholds_and_gap(traces)
        historical_lengths = [float(t["true"]) for t in traces]
        calibration_payload = {
            "fingerprint": calibration_fingerprint,
            "cache": cache,
            "calib_params": calib_params,
            "historical_lengths": historical_lengths,
        }
        cache_parent = os.path.dirname(
            os.path.abspath(calibration_cache_path)
        )
        os.makedirs(cache_parent, exist_ok=True)
        with open(calibration_cache_path, "wb") as handle:
            pickle.dump(
                calibration_payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved reusable calibration to {calibration_cache_path}.")
        del calib_model
        torch.cuda.empty_cache()
    else:
        cache = calibration_payload["cache"]
        calib_params = calibration_payload["calib_params"]
        historical_lengths = calibration_payload["historical_lengths"]

    historical_mean = float(np.mean(historical_lengths))
    historical_std = float(np.std(historical_lengths))
    historical_priors = build_class_output_priors(
        trace_records, historical_lengths
    )
    print(
        f"Historical output prior: mean={historical_mean:.1f}, "
        f"std={historical_std:.1f} tokens"
    )

    probe_n = min(args.capacity_probe_requests, len(routing_prompts))
    probe_prompts = routing_prompts[:probe_n]
    probe_prompt_lengths = prompt_lengths[:probe_n]
    probe_generation_params = generation_params[:probe_n]
    probe_traffic_classes = traffic_classes[:probe_n]
    preflight_n = min(args.preflight_requests, len(routing_prompts))
    preflight_prompts = routing_prompts[:preflight_n]
    preflight_prompt_lengths = prompt_lengths[:preflight_n]
    preflight_generation_params = generation_params[:preflight_n]
    preflight_traffic_classes = traffic_classes[:preflight_n]
    print(
        f"\nCapacity calibration uses {probe_n} representative requests; "
        f"each traffic preflight uses {preflight_n}; "
        f"policy runs use all {len(routing_prompts)} requests."
    )

    control_probe = None
    detector_overhead_pct = None
    if args.capacity_rps is not None:
        saturated_request_capacity = float(args.capacity_rps)
        probe = {
            "provided_capacity_rps": saturated_request_capacity,
            "capacity_probe_skipped": True,
        }
        print(
            f"\nReusing supplied saturated capacity: "
            f"{saturated_request_capacity:.3f} req/s."
        )
    else:
        if args.measure_overhead:
            print("\nRunning no-detector saturated throughput control...")
            control_probe = run_local_queue_experiment(
                policy_name="round_robin",
                prompts=probe_prompts,
                cache=cache,
                calib_params=calib_params,
                historical_mean=historical_mean,
                historical_std=historical_std,
                historical_priors=historical_priors,
                prompt_lengths=probe_prompt_lengths,
                generation_params=probe_generation_params,
                traffic_classes=probe_traffic_classes,
                arrival_offsets=[0.0] * probe_n,
                queue_wait_budget_sec=None,
                offered_rps=float("inf"),
                load_factor=None,
                arrival_mode="overhead_control",
                burstiness=1.0,
                arrival_seed=POLICY_SEED,
                experiment_id="no-detector-capacity-control",
                use_measurement_window=False,
                enable_helix_tracker=False,
            )

        print("\nRunning warmed saturated capacity probe...")
        probe = run_local_queue_experiment(
            policy_name="round_robin",
            prompts=probe_prompts,
            cache=cache,
            calib_params=calib_params,
            historical_mean=historical_mean,
            historical_std=historical_std,
            historical_priors=historical_priors,
            prompt_lengths=probe_prompt_lengths,
            generation_params=probe_generation_params,
            traffic_classes=probe_traffic_classes,
            arrival_offsets=[0.0] * probe_n,
            queue_wait_budget_sec=None,
            offered_rps=float("inf"),
            load_factor=None,
            arrival_mode="capacity_probe",
            burstiness=1.0,
            arrival_seed=POLICY_SEED,
            experiment_id="capacity-probe",
            use_measurement_window=False,
            enable_helix_tracker=True,
        )
        saturated_request_capacity = probe["n_completed"] / max(
            1e-9, probe["wall_sec"]
        )
        if saturated_request_capacity <= 0:
            raise RuntimeError("Capacity probe completed no measurable work")
        print(
            f"Measured saturated capacity: "
            f"{saturated_request_capacity:.3f} req/s, "
            f"{probe['tokens_per_sec']:.1f} output tok/s."
        )
        if control_probe is not None and control_probe["tokens_per_sec"] > 0:
            detector_overhead_pct = 100.0 * (
                control_probe["tokens_per_sec"] - probe["tokens_per_sec"]
            ) / control_probe["tokens_per_sec"]
            print(
                f"Detector throughput overhead: {detector_overhead_pct:+.2f}% "
                f"(control {control_probe['tokens_per_sec']:.1f} vs detector "
                f"{probe['tokens_per_sec']:.1f} output tok/s)."
            )

    preflight_results = []
    selected_preflight = None
    if args.manual_workload:
        effective_load_levels = tuple(args.load_levels)
        effective_burstiness_levels = tuple(args.burstiness_levels)
        print(
            "\nAutomatic workload calibration disabled; using requested "
            "manual load and burstiness levels."
        )
    else:
        print(
            "\nRunning short traffic preflights to select a measured "
            "operating point..."
        )
        preflight_candidates = []
        for candidate_idx, (load_factor, burstiness) in enumerate(zip(
            args.preflight_load_levels,
            args.preflight_burstiness_levels,
        )):
            effective_burstiness = (
                1.0 if args.arrival_mode == "trace" else burstiness
            )
            request_rate = saturated_request_capacity * load_factor
            arrival_offsets = build_arrival_schedule(
                args=args,
                n_requests=preflight_n,
                request_rate=request_rate,
                burstiness=effective_burstiness,
                seed=POLICY_SEED,
            )
            print(
                f"\nPreflight {candidate_idx + 1}/"
                f"{len(args.preflight_load_levels)}: "
                f"load={load_factor:.2f}, "
                f"burstiness={effective_burstiness:.2f}, "
                f"offered={request_rate:.3f} req/s"
            )
            preflight_result = run_local_queue_experiment(
                policy_name="round_robin",
                prompts=preflight_prompts,
                cache=cache,
                calib_params=calib_params,
                historical_mean=historical_mean,
                historical_std=historical_std,
                historical_priors=historical_priors,
                prompt_lengths=preflight_prompt_lengths,
                generation_params=preflight_generation_params,
                traffic_classes=preflight_traffic_classes,
                arrival_offsets=arrival_offsets,
                queue_wait_budget_sec=args.queue_wait_budget_sec,
                offered_rps=request_rate,
                load_factor=load_factor,
                arrival_mode=f"preflight_{args.arrival_mode}",
                burstiness=effective_burstiness,
                arrival_seed=POLICY_SEED + 1000 + candidate_idx,
                ttft_slo_sec=args.ttft_slo_sec,
                itl_slo_sec=args.itl_slo_ms / 1000.0,
                experiment_id=f"traffic-preflight-{candidate_idx}",
                use_measurement_window=False,
                enable_helix_tracker=True,
            )
            preflight_candidates.append({
                "load_factor": float(load_factor),
                "burstiness": float(effective_burstiness),
                "offered_rps": float(request_rate),
                "result": preflight_result,
            })

        selected_preflight, preflight_results = choose_preflight_workload(
            preflight_candidates,
            queue_wait_budget_sec=args.queue_wait_budget_sec,
            ttft_slo_sec=args.ttft_slo_sec,
        )
        effective_load_levels = (
            float(selected_preflight["load_factor"]),
        )
        effective_burstiness_levels = (
            float(selected_preflight["burstiness"]),
        )
        selected_result = selected_preflight["result"]
        print(
            "\nSelected measured workload: "
            f"load={selected_preflight['load_factor']:.2f}, "
            f"burstiness={selected_preflight['burstiness']:.2f}, "
            f"GPU busy={selected_result['mean_gpu_busy'] * 100:.1f}%, "
            f"pending-depth p95="
            f"{selected_result['pending_depth_at_arrival_p95']:.1f}, "
            f"TTFT p95={selected_result['ttft_p95']:.3f}s, "
            f"shed={selected_result['shed_fraction'] * 100:.1f}%."
        )

    # The first full round-robin run defines a fixed output-length workload
    # trace for the diagnostic Oracle. Do not mix in the saturated capacity
    # probe: different batch shapes can change greedy EOS timing.
    true_lengths = [None] * len(routing_prompts)

    output_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    probe_summary = {
        k: v for k, v in probe.items()
        if k not in ("details", "rejections")
    }
    experiment_config = {
        "model_id": MODEL_ID,
        "target_layer": TARGET_LAYER,
        "calibration_cache_version": CALIBRATION_CACHE_VERSION,
        "max_concurrent_seqs_override": MAX_CONCURRENT_SEQS_OVERRIDE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "calibration_max_input_tokens": CALIBRATION_MAX_INPUT_TOKENS,
        "benchmark_requests": len(routing_prompts),
        "quick_mode": bool(args.quick),
        "quick_all_policies": bool(args.quick_all_policies),
        "policies": list(args.policies),
        "calibration_prompt_count": calibration_prompt_count,
        "trace_prompt_count": trace_prompt_count,
        "calibration_cache_path": calibration_cache_path,
        "capacity_rps_override": args.capacity_rps,
        "capacity_probe_requests": probe_n,
        "trace_batch_size": TRACE_BATCH_SIZE,
        "historical_output_priors": historical_priors,
        "preflight_requests_per_candidate": preflight_n,
        "requested_load_levels": args.load_levels,
        "requested_burstiness_levels": args.burstiness_levels,
        "preflight_load_levels": args.preflight_load_levels,
        "preflight_burstiness_levels": args.preflight_burstiness_levels,
        "automatic_workload_selection": not args.manual_workload,
        "load_levels": effective_load_levels,
        "burstiness_levels": effective_burstiness_levels,
        "arrival_mode": args.arrival_mode,
        "trace_path": args.trace_path,
        "trace_start_row": args.trace_start_row,
        "trace_log_type": args.trace_log_type,
        "queue_wait_budget_sec": args.queue_wait_budget_sec,
        "ttft_slo_sec": args.ttft_slo_sec,
        "itl_slo_ms": args.itl_slo_ms,
        "seeds": args.seeds,
        "measurement_warmup_fraction": MEASUREMENT_WARMUP_FRACTION,
        "measurement_cooldown_fraction": MEASUREMENT_COOLDOWN_FRACTION,
        "min_helix_active_snapshot_fraction": (
            MIN_HELIX_ACTIVE_SNAPSHOT_FRACTION
        ),
        "min_helix_informative_estimate_fraction": (
            MIN_HELIX_INFORMATIVE_ESTIMATE_FRACTION
        ),
        "cache_workload": (
            "single-turn prompts; prefix caching enabled; no synthetic "
            "session affinity or shared system prompt"
        ),
        "saturated_request_capacity": saturated_request_capacity,
        "capacity_probe": probe_summary,
        "traffic_preflights": [
            {
                "load_factor": candidate["load_factor"],
                "burstiness": candidate["burstiness"],
                "offered_rps": candidate["offered_rps"],
                "selection_score": candidate["selection_score"],
                "metrics": {
                    key: value
                    for key, value in candidate["result"].items()
                    if key not in ("details", "rejections")
                },
            }
            for candidate in preflight_results
        ],
        "selected_preflight": (
            {
                "load_factor": selected_preflight["load_factor"],
                "burstiness": selected_preflight["burstiness"],
                "offered_rps": selected_preflight["offered_rps"],
                "selection_score": selected_preflight["selection_score"],
            }
            if selected_preflight is not None else None
        ),
        "detector_overhead_percent": detector_overhead_pct,
        "no_detector_capacity_control": (
            {
                k: v for k, v in control_probe.items()
                if k not in ("details", "rejections")
            }
            if control_probe is not None else None
        ),
        "oracle_truth_source": (
            "fixed output-length trace observed in the first full round-robin "
            "run; later scheduling-dependent EOS differences are reported"
        ),
        "routing_score": (
            "Measured backlog-drain routing. Active work uses confidence-"
            "weighted Helix remaining tokens; unseen and queued work uses a "
            "traffic-class empirical-Bayes output prior. Total work is "
            "normalized by each replica's measured decode step throughput and "
            "prefill time. The only routing guard is a small token deadband; "
            "request-count imbalance is allowed when predicted work justifies "
            "it. Trace Oracle substitutes fixed exact remaining lengths."
        ),
        "routing_deadband_tokens": ROUTING_DEADBAND_TOKENS,
        "routing_deadband_sec": ROUTING_DEADBAND_SEC,
        "helix_forecast_grace_sec": HELIX_FORECAST_GRACE_SEC,
        "helix_forecast_half_life_sec": HELIX_FORECAST_HALF_LIFE_SEC,
        "work_aware_queue_discipline": (
            "The production default leaves vLLM continuous batching FCFS and "
            "changes only replica routing; this avoids real KV recomputation "
            "and priority-heap churn. Setting HELIX_ENABLE_VLLM_PREEMPTION=1 "
            "enables the experimental discipline in which decode work is "
            "ordered within bounded arrival cohorts using stable "
            "predicted-total-length buckets. New requests remain FCFS behind "
            "resident decode work until their calibrated TTFT deadline. Helix "
            "may revise a bucket "
            "only when its early/late trajectory checkpoint advances and the "
            "change clears the configured hysteresis; Oracle's exact bucket is "
            "fixed at admission. Unseen requests are promoted at a deadline "
            "calibrated from the same-trace queue-size TTFT tail, and stalled "
            "streams are rescued using that baseline's max-ITL tail. vLLM "
            "controls the actual preemption and recomputation cost."
        ),
        "priority_ttft_deadline_sec": PRIORITY_TTFT_DEADLINE_SEC,
        "priority_itl_rescue_sec": PRIORITY_ITL_RESCUE_SEC,
        "priority_ttft_baseline_ratio": PRIORITY_TTFT_BASELINE_RATIO,
        "priority_itl_baseline_multiplier": (
            PRIORITY_ITL_BASELINE_MULTIPLIER
        ),
        "priority_itl_rescue_min_sec": PRIORITY_ITL_RESCUE_MIN_SEC,
        "priority_length_bucket_tokens": PRIORITY_LENGTH_BUCKET_TOKENS,
        "priority_length_hysteresis_buckets": (
            PRIORITY_LENGTH_HYSTERESIS_BUCKETS
        ),
        "priority_unstarted_base": PRIORITY_UNSTARTED_BASE,
        "experimental_vllm_preemption_enabled": (
            ENABLE_EXPERIMENTAL_VLLM_PREEMPTION
        ),
        "helix_priority_cohort_size": HELIX_PRIORITY_COHORT_SIZE,
        "oracle_priority_cohort_size": ORACLE_PRIORITY_COHORT_SIZE,
    }

    all_results = []
    canonical_policy_order = (
        "round_robin", "queue_size", "helix_work", "oracle_work"
    )
    policies = tuple(
        policy for policy in canonical_policy_order
        if policy in set(args.policies)
    )
    burstiness_values = (
        (1.0,) if args.arrival_mode == "trace"
        else effective_burstiness_levels
    )

    for arrival_seed in args.seeds:
        for burstiness in burstiness_values:
            for load_factor in effective_load_levels:
                request_rate = saturated_request_capacity * load_factor
                arrival_offsets = build_arrival_schedule(
                    args=args,
                    n_requests=len(routing_prompts),
                    request_rate=request_rate,
                    burstiness=burstiness,
                    seed=arrival_seed,
                )
                experiment_id = (
                    f"{args.arrival_mode}-rho{load_factor:.2f}-"
                    f"k{burstiness:.2f}-seed{arrival_seed}"
                )
                print(
                    f"\nStarting {experiment_id}: {request_rate:.3f} req/s "
                    f"against {saturated_request_capacity:.3f} req/s capacity"
                )
                group = []
                for policy_name in policies:
                    if policy_name == "oracle_work":
                        missing_truth = [
                            rid for rid, value in enumerate(true_lengths)
                            if value is None
                        ]
                        if missing_truth:
                            raise RuntimeError(
                                "Trace-Oracle reference lengths are missing for "
                                f"{len(missing_truth)} requests. Lower the first "
                                "load or increase the queue-wait budget so the "
                                "round-robin reference run completes all prompts."
                            )
                    random.seed(POLICY_SEED)
                    calibrated_priority_ttft = None
                    calibrated_priority_itl = None
                    if (
                        ENABLE_EXPERIMENTAL_VLLM_PREEMPTION
                        and policy_name in ("helix_work", "oracle_work")
                    ):
                        queue_baseline = next(
                            (
                                row for row in group
                                if row["policy"] == "queue_size"
                            ),
                            None,
                        )
                        if queue_baseline is None:
                            raise RuntimeError(
                                "Priority policies require queue_size to run "
                                "first on the same workload."
                            )
                        baseline_ttft_p95 = float(
                            queue_baseline["ttft_p95"]
                        )
                        baseline_max_itl_p95 = float(
                            queue_baseline["max_itl_p95"]
                        )
                        if not np.isfinite(baseline_ttft_p95):
                            baseline_ttft_p95 = (
                                PRIORITY_TTFT_DEADLINE_SEC
                                / PRIORITY_TTFT_BASELINE_RATIO
                            )
                        if not np.isfinite(baseline_max_itl_p95):
                            baseline_max_itl_p95 = (
                                PRIORITY_ITL_RESCUE_SEC
                                / PRIORITY_ITL_BASELINE_MULTIPLIER
                            )
                        calibrated_priority_ttft = min(
                            PRIORITY_TTFT_DEADLINE_SEC,
                            PRIORITY_TTFT_BASELINE_RATIO
                            * baseline_ttft_p95,
                        )
                        calibrated_priority_itl = min(
                            PRIORITY_ITL_RESCUE_SEC,
                            max(
                                PRIORITY_ITL_RESCUE_MIN_SEC,
                                PRIORITY_ITL_BASELINE_MULTIPLIER
                                * baseline_max_itl_p95,
                            ),
                        )
                        print(
                            "  Calibrated priority scheduler: "
                            f"TTFT deadline={calibrated_priority_ttft:.3f}s, "
                            f"ITL rescue={calibrated_priority_itl:.3f}s"
                        )
                    result = run_local_queue_experiment(
                        policy_name=policy_name,
                        prompts=routing_prompts,
                        cache=cache,
                        calib_params=calib_params,
                        historical_mean=historical_mean,
                        historical_std=historical_std,
                        historical_priors=historical_priors,
                        prompt_lengths=prompt_lengths,
                        true_lengths=(
                            true_lengths if policy_name == "oracle_work" else None
                        ),
                        generation_params=generation_params,
                        traffic_classes=traffic_classes,
                        arrival_offsets=arrival_offsets,
                        queue_wait_budget_sec=args.queue_wait_budget_sec,
                        offered_rps=request_rate,
                        load_factor=load_factor,
                        arrival_mode=args.arrival_mode,
                        burstiness=burstiness,
                        arrival_seed=arrival_seed,
                        ttft_slo_sec=args.ttft_slo_sec,
                        itl_slo_sec=args.itl_slo_ms / 1000.0,
                        experiment_id=experiment_id,
                        use_measurement_window=True,
                        enable_helix_tracker=(policy_name == "helix_work"),
                        priority_ttft_deadline_sec=(
                            calibrated_priority_ttft
                        ),
                        priority_itl_rescue_sec=calibrated_priority_itl,
                    )
                    group.append(result)
                    all_results.append(result)
                    if (
                        policy_name == "round_robin"
                        and not any(v is not None for v in true_lengths)
                    ):
                        for record in result["details"]:
                            rid = int(record["req_id"])
                            true_lengths[rid] = int(record["n_tokens"])

                    comparable_errors = [
                        abs(int(record["n_tokens"]) - true_lengths[int(record["req_id"])])
                        for record in result["details"]
                        if true_lengths[int(record["req_id"])] is not None
                    ]
                    result["reference_length_comparisons"] = len(comparable_errors)
                    result["reference_length_mismatch_fraction"] = (
                        sum(error > 0 for error in comparable_errors)
                        / max(1, len(comparable_errors))
                    )
                    result["mean_abs_reference_length_error_tokens"] = (
                        float(np.mean(comparable_errors))
                        if comparable_errors else float("nan")
                    )
                    # Persist every completed run. A long sweep can therefore be
                    # inspected as useful partial output instead of producing
                    # its first CSV only after all four policies.
                    experiment_config["oracle_truth_coverage_fraction"] = (
                        sum(v is not None for v in true_lengths)
                        / max(1, len(true_lengths))
                    )
                    save_results(
                        all_results,
                        output_dir,
                        experiment_config=experiment_config,
                    )
                    if (
                        policy_name == "helix_work"
                        and not result["helix_snapshot_valid"]
                    ):
                        raise RuntimeError(
                            "Invalid Helix run: active snapshot coverage="
                            f"{result['helix_active_snapshot_fraction']:.3f}, "
                            "fresh coverage="
                            f"{result['helix_fresh_active_snapshot_fraction']:.3f}, "
                            "estimate coverage="
                            f"{result['helix_estimate_coverage']:.3f}, "
                            "informative fraction="
                            f"{result['helix_informative_estimate_fraction']:.3f}. "
                            "Results were checkpointed, but Oracle was not run."
                        )

                attach_paired_queue_comparisons(group)
                print_comparison_report(group)
                save_results(
                    all_results,
                    output_dir,
                    experiment_config=experiment_config,
                )

    print(f"\nCompleted {len(all_results)} policy-condition runs.")


if __name__ == "__main__":
    main()
