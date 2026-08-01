# Router correction plan

## Objective

Make the two-T4 experiment a defensible test of work-aware routing without
pretending to reproduce Fireworks' fleet. Queue size remains unchanged. Helix
and Oracle use the same work-aware placement equation so the Oracle is a clean
upper bound on the value of remaining-length information.

## Problems found in the full-file audit

1. `max_num_seqs=24` was simultaneously an engine limit, tracker allocation
   limit, queue-depth definition, utilization denominator, and routing-model
   lane count. Below 24 pending requests, that lane model made active remaining
   work almost irrelevant.
2. Early Helix forecasts were confidence-weighted in the worker and then
   weighted a second time in the parent. A nominal confidence of 0.55 therefore
   contributed only 0.3025 of the routing signal.
3. The Oracle did not install the hidden-state hook, so mixed prefill/decode
   steps were incorrectly entered into its decode timing table as pure decode.
   Helix and Oracle were therefore scored with different, biased timing models.
4. Forecasts older than a fixed cutoff were discarded completely, even when a
   long vLLM iteration was the reason no newer snapshot existed.
5. The online tracker projected every configured slot on every step, including
   unused slots. Increasing `max_num_seqs` therefore increased detector overhead
   even when the active batch did not grow.
6. Historical traces were batched but completed rows remained in every decode
   step. A single long trace forced all short rows to keep consuming compute.
7. Queue and saturation diagnostics inferred waiting from the old 24-slot
   constant rather than the live scheduler and active-request state.
8. A dead worker could leave the parent waiting indefinitely for results.
9. Only a global output-length prior was used for queued/unseen requests even
   though the workload already contains traffic-class labels.
10. The CSV did not expose why Helix chose a GPU, how often it disagreed with
    queue size, whether the deadband suppressed it, or what active work it saw.

## Implemented design

### Capacity and state

- Omit `max_num_seqs` by default and read each vLLM engine's effective
  scheduler capacity after construction.
- Retain `HELIX_MAX_CONCURRENT_SEQS` only as an explicit experiment override.
- Size shared tracker/publication storage from the actual benchmark workload.
- Record effective engine capacity plus live scheduler running/waiting counts.

### Work-aware placement

For each GPU, calculate:

`score = (active remaining + queued prior + target prior) / measured output throughput + queued/target prefill time`

- Helix supplies confidence-blended remaining tokens for active requests.
- Oracle substitutes exact fixed trace lengths everywhere.
- Queued and unseen Helix requests use empirical-Bayes traffic-class priors.
- The candidate may accept a larger request count when it has less predicted
  work; only a small token/time deadband suppresses noise.
- Queue size and round robin retain their existing implementations.

### Detector correctness and cost

- Return raw early Helix estimates with confidence and blend exactly once.
- Decay stale confidence smoothly after a grace interval.
- Run the projection/PLL kernel only for active hidden-state rows.
- Add a dense-versus-sparse tracker regression check.

### Timing and calibration

- Detect mixed prefill/decode iterations from first-token output deltas for
  every policy, not only from the Helix hook.
- Compact finished rows and KV-cache entries during offline trace collection.
- Version the calibration cache so corrected trace logic cannot silently reuse
  incompatible cached data.

### Reliability and evidence

- Poll worker health during startup, arrival scheduling, and completion; abort
  immediately with the failed GPU and exit code.
- Persist per-decision scores, predicted gains, queue/work candidates, override
  reasons, active/total work, prior source, and engine capacities.
- Aggregate override and Helix-coverage diagnostics across seeds.

## Validation sequence

1. Run CPU regression checks:

   `python router.py --sanity-only`

2. Run a short placement A/B test:

   `python router.py --quick`

3. Reuse the printed saturated capacity during iterations:

   `python router.py --quick --capacity-rps <measured_req_per_sec>`

4. Run the shortest real-vLLM Oracle/headroom check:

   `python router.py --quick-all-policies --capacity-rps <measured_req_per_sec>`

5. Inspect these fields before interpreting latency:

   - `work_candidate_disagreement_count`
   - `work_override_fraction`
   - `work_override_acceptance_fraction`
   - `routing_reason_counts`
   - `routing_predicted_gain_mean_sec`
   - `helix_active_snapshot_fraction`
   - `helix_informative_estimate_fraction`
   - `engine_capacity_gpu0`, `engine_capacity_gpu1`

6. Run the full four-policy experiment only after the short A/B shows that
   Helix is active and the hook diagnostics are valid.

## Success criteria

- Oracle must choose the lower exact-work GPU in the synthetic regression.
- Helix must override queue size when the work advantage clears the deadband.
- Tracker slot exhaustion and true hook errors must be zero.
- Oracle should provide positive headroom over queue size before Helix results
  are treated as evidence for the detector.
- The observed gain must be reported with paired request bootstrap intervals.
  A 25% simulation result is a target, not a guaranteed real-vLLM outcome.
