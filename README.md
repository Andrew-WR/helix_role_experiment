# Helix router benchmark

This repository contains a production-shaped, two-replica vLLM benchmark for
the online remaining-length detector and its routing policies.

The benchmark does **not** claim to reproduce a Fireworks fleet. It measures
the warmed capacity of the current two-T4 setup, then expresses offered load
as a fraction of that capacity. Every policy receives the same prompts,
generation parameters, arrival timestamps, admission controller, and SLOs.

## What changed

- Both vLLM workers signal readiness after a real batched warmup. Model loading
  and kernel warmup are excluded from request TTFT.
- A short saturated reference run (256 requests by default) measures request
  capacity for this model, hardware, context distribution, and detector.
- The optional `--measure-overhead` control disables the detector hook and
  reports output-throughput overhead. It is separate from policy comparison
  because it doubles capacity-calibration time.
- Open-loop traffic defaults to `0.95`, `1.00`, and `1.05` times measured
  request capacity, where waiting-time differences can emerge.
- Queue + Helix uses exactly `historical mean × waiting requests + sum(active
  Helix remaining-token estimates)`. There is no throughput conversion,
  prefill term, queue-count guard, or score deadband.
- Active decode rows are mapped through vLLM V1's
  `model_runner.input_batch.req_ids`, with older `req_states` layouts retained
  only as a compatibility fallback.
- The evaluated policies are round robin, queue size, queue + Helix, and a
  trace Oracle. The first full round-robin run defines the Oracle’s fixed
  output-length workload trace; later EOS differences are reported rather than
  assumed impossible. A separate `queue + mean` policy is intentionally
  omitted: on
  identical replicas, assigning every unfinished request the same mean cost
  is exactly the same ordering as queue size.
- Offline Transformers calibration uses prompts of at most 2,048 input tokens
  to avoid quadratic-attention OOMs on a T4. This does not truncate the routed
  vLLM workload, which still uses the configured 8,192-token context limit.
- Synthetic arrivals use Gamma inter-arrivals. Shape `1.0` is Poisson; `0.5`
  is burstier; `0.2` is a severe burst sensitivity test.
- A BurstGPT CSV can supply real arrival timing. Only timestamps are replayed;
  prompts retain natural EOS so the length detector remains valid.
- A policy-neutral admission controller sheds a request only when both
  replicas exceed the predicted queue-wait budget, which defaults to five
  seconds.
- Queue reporting separates vLLM's measured engine queue time, measured
  router-to-first-schedule delay, and the admission model's predicted wait.
- Helix runs report active-snapshot, fresh-snapshot, estimate, and informative
  forecast coverage. Runs below the validity thresholds are checkpointed and
  stopped before Oracle rather than presented as policy comparisons.
- The first and last 10% of requests supply warmup/cooldown traffic. Metrics
  use the steady-state middle 80%.
- Results include accepted RPS, output and prompt TPS, queue-wait percentiles,
  shedding, goodput, per-class results, moving-block bootstrap confidence
  intervals, paired policy-minus-queue p95 intervals, per-request outcomes,
  and cross-seed aggregates.
- `MAX_MODEL_LEN` defaults to 8192 so the workload is not artificially limited
  to sub-1k prompts. Override it if Kaggle memory requires a smaller value.

## Cache interpretation

The default workload is deliberately single-turn and has no synthetic session
affinity or shared system prompt. Prefix caching remains enabled, but the
benchmark does not force a target cache-hit rate.

Multi-turn conversations are an important source of prefix reuse, but not the
only one. Long shared system prompts, repeated RAG templates, few-shot
examples, and repeated code prefixes can also produce cache hits. None should
be fabricated unless they represent the product workload being claimed.

## Kaggle commands

Fast integration run:

```bash
python router.py --quick
```

Main load sweep (the defaults are intentionally sized for two T4s):

```bash
python router.py \
  --benchmark-requests 600 \
  --capacity-probe-requests 256 \
  --load-levels 0.95,1.00,1.05 \
  --burstiness-levels 0.5 \
  --seeds 0 \
  --queue-wait-budget-sec 5.0
```

Every completed policy run is saved immediately, so a stopped notebook keeps
useful partial CSV and JSON output. The first full round-robin run supplies the
fixed output-length trace for the trace Oracle; the saturated capacity probe
is not used because its different batch shapes can change natural EOS timing.

After a single-seed sweep identifies the load where policies separate, run
replicates only at that load:

```bash
python router.py \
  --benchmark-requests 1000 \
  --load-levels 1.00 \
  --burstiness-levels 0.5 \
  --seeds 0,1,2
```

Keep the five-second queue-wait budget for the first sweep so the round-robin
reference run completes every prompt. If heavy shedding leaves unseen
reference lengths, lower the first load or increase the budget.

Measure detector overhead separately when needed:

```bash
python router.py --quick --measure-overhead
```

BurstGPT timestamp replay:

```bash
python router.py \
  --arrival-mode trace \
  --trace-path /kaggle/input/burstgpt/BurstGPT_without_fails_1.csv \
  --trace-log-type "API log" \
  --benchmark-requests 600 \
  --load-levels 0.95,1.00,1.05 \
  --seeds 0
```

## Environment controls

These must be set before Python starts because multiprocessing uses `spawn`:

```bash
HELIX_MAX_CONCURRENT_SEQS=24
HELIX_MAX_MODEL_LEN=8192
HELIX_CALIBRATION_MAX_INPUT_TOKENS=2048
HELIX_N_CALIB_PROMPTS=40
HELIX_N_TRACE_PROMPTS=160
```

For the concurrency sensitivity test, run separate jobs with
`HELIX_MAX_CONCURRENT_SEQS=16`, `24`, and `32`. Compare throughput, ITL,
shedding, and routing improvement; do not assume the largest batch is best.

## Outputs

Kaggle writes these under `/kaggle/working`:

- `router_local_queue_results.json`: complete summary metrics.
- `router_local_queue_results.csv`: flat policy-condition summaries.
- `router_local_queue_requests.csv`: completed and shed request outcomes.
- `router_seed_aggregates.json` / `.csv`: means and 95% cross-seed intervals.
- `router_experiment_config.json`: the exact configuration and capacity probe.
