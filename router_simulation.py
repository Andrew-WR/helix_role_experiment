"""Deterministic CPU simulation of two-replica continuous-batching routing.

The simulator deliberately separates routing from GPU/numerical noise:

* Every policy receives the same immutable arrivals, prompt lengths, and output
  lengths.
* Each replica owns a local FCFS queue and admits at most ``max_num_seqs``.
* Prefill time and batch-size/context-dependent decode-step time are modeled
  explicitly.
* Exact and noisy-length policies can be compared against queue size without
  changing the service trace.
* Round-robin and queue-size retain immediate binding and local FCFS exactly;
  experimental Helix/Oracle policies may use deferred binding, bounded queue
  reordering, or decode-step-boundary preemption with vLLM-style recomputation.

Examples
--------
Calibrate from the aggregate CSV produced by router.py:

    python router_simulation.py --calibration-csv router_local_queue_results.csv

For the most faithful replay, also provide the per-request CSV:

    python router_simulation.py \
      --calibration-csv router_local_queue_results.csv \
      --request-csv router_local_queue_requests.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


EPS = 1e-12


def percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * float(q) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def bounded_lognormal(rng, mean, cv, lower, upper):
    if cv <= 0:
        return int(round(min(upper, max(lower, mean))))
    sigma2 = math.log1p(cv * cv)
    mu = math.log(max(EPS, mean)) - 0.5 * sigma2
    value = rng.lognormvariate(mu, math.sqrt(sigma2))
    return int(round(min(upper, max(lower, value))))


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int
    helix_error_z: float


@dataclass
class RequestState:
    spec: RequestSpec
    replica: int
    generated: int = 0
    prefill_start: Optional[float] = None
    first_token: Optional[float] = None
    finish: Optional[float] = None
    preemptions: int = 0
    last_token: Optional[float] = None
    max_token_gap: float = 0.0


@dataclass
class StepPlan:
    start: float
    end: float
    participants: tuple[int, ...]
    admitted: tuple[int, ...]


@dataclass
class Replica:
    replica_id: int
    active: list[int] = field(default_factory=list)
    waiting: deque[int] = field(default_factory=deque)
    plan: Optional[StepPlan] = None
    busy_seconds: float = 0.0
    decode_slot_seconds: float = 0.0
    max_pending: int = 0
    max_waiting: int = 0

    def pending_count(self):
        return len(self.active) + len(self.waiting)


@dataclass
class TimingModel:
    max_num_seqs: int = 24
    prefill_fixed_sec: float = 0.006
    prefill_tokens_per_sec: float = 15000.0
    decode_fixed_sec: float = 0.0075
    decode_linear_sec: float = 0.00095
    decode_quadratic_sec: float = 0.0
    context_scale_tokens: float = 2048.0
    context_slowdown: float = 0.15

    def decode_step_sec(self, batch_size, mean_context_tokens):
        batch_size = max(1, int(batch_size))
        base = (
            self.decode_fixed_sec
            + self.decode_linear_sec * batch_size
            + self.decode_quadratic_sec * batch_size * batch_size
        )
        context_factor = 1.0 + self.context_slowdown * (
            max(0.0, float(mean_context_tokens))
            / max(1.0, self.context_scale_tokens)
        )
        return max(1e-6, base * context_factor)

    def step_sec(self, states, participants, admitted):
        contexts = [
            states[rid].spec.prompt_tokens + states[rid].generated
            for rid in participants
        ]
        decode = self.decode_step_sec(
            len(participants),
            statistics.fmean(contexts) if contexts else 0.0,
        )
        if not admitted:
            return decode
        prefill_tokens = sum(
            states[rid].spec.prompt_tokens for rid in admitted
        )
        prefill = (
            self.prefill_fixed_sec
            + prefill_tokens / max(EPS, self.prefill_tokens_per_sec)
        )
        # vLLM can mix decode tokens with chunked prefill. Additive time is a
        # conservative approximation and is exposed through CLI parameters.
        return prefill + decode


@dataclass(frozen=True)
class Policy:
    name: str
    knowledge: str
    objective: str = "completion"
    min_gain_sec: float = 0.0
    max_count_imbalance: int = 10**9


def read_aggregate_calibration(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in calibration CSV: {path}")
    baseline = next(
        (row for row in rows if row.get("policy") == "queue_size"),
        rows[0],
    )

    def number(name, default):
        raw = baseline.get(name)
        try:
            value = float(raw)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    # The CSV reports active depth across both replicas; ITL is a per-replica
    # decode-step interval, so calibrate against half of the cluster depth.
    active = max(1.0, number("active_depth_at_arrival_mean", 24.0) / 2.0)
    itl = number("itl_p50", 0.030)
    fixed = min(0.008, 0.35 * itl)
    linear = max(1e-6, (itl - fixed) / active)
    return {
        "n_requests": int(number("n_submitted", 300)),
        "offered_rps": number("offered_rps", 3.4),
        "burstiness": number("burstiness", 0.05),
        "mean_prompt_tokens": number("mean_prompt_tokens", 230.0),
        "mean_output_tokens": number("mean_output_tokens", 345.0),
        "target_tokens_per_sec": number("tokens_per_sec", 700.0),
        "target_ttft_p95": number("ttft_p95", 5.0),
        "target_pending_p95": number(
            "pending_depth_at_arrival_p95", 48.0
        ),
        "target_waiting_p95": number(
            "waiting_depth_at_arrival_p95", 4.0
        ),
        "decode_reference_batch": active,
        "decode_reference_step_sec": itl,
        "decode_fixed_sec": fixed,
        "decode_linear_sec": linear,
    }


def read_request_trace(path, policy="round_robin"):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("policy") == policy
            and row.get("status", "completed") == "completed"
        ]
    if not rows:
        raise ValueError(f"No completed {policy} rows in {path}")
    rows.sort(key=lambda row: int(row["req_id"]))
    first_arrival = min(float(row["arrival_time"]) for row in rows)
    return [
        {
            "request_id": int(row["req_id"]),
            "arrival": float(row["arrival_time"]) - first_arrival,
            "prompt_tokens": int(float(row["prompt_len"])),
            "output_tokens": int(float(row["n_tokens"])),
        }
        for row in rows
    ]


def generate_workload(args, calibration, seed):
    workload_rng = random.Random(args.workload_seed)
    arrival_rng = random.Random(seed)
    if args.request_csv:
        trace = read_request_trace(args.request_csv)
        return [
            RequestSpec(
                request_id=index,
                arrival=float(row["arrival"]),
                prompt_tokens=int(row["prompt_tokens"]),
                output_tokens=int(row["output_tokens"]),
                helix_error_z=workload_rng.gauss(0.0, 1.0),
            )
            for index, row in enumerate(trace[: args.requests])
        ]

    n_requests = args.requests
    rate = args.offered_rps or calibration["offered_rps"]
    burstiness = args.burstiness or calibration["burstiness"]
    mean_prompt = calibration["mean_prompt_tokens"]
    mean_output = calibration["mean_output_tokens"]
    try:
        import numpy as np
        arrival_gaps = np.random.default_rng(seed).gamma(
            shape=burstiness,
            scale=1.0 / (rate * burstiness),
            size=max(0, n_requests - 1),
        ).astype(float).tolist()
    except ImportError:
        arrival_gaps = [
            arrival_rng.gammavariate(
                burstiness, 1.0 / (rate * burstiness)
            )
            for _ in range(max(0, n_requests - 1))
        ]

    arrival = 0.0
    specs = []
    for request_id in range(n_requests):
        if request_id:
            arrival += arrival_gaps[request_id - 1]
        specs.append(RequestSpec(
            request_id=request_id,
            arrival=arrival,
            prompt_tokens=bounded_lognormal(
                workload_rng,
                mean_prompt,
                args.prompt_cv,
                8,
                args.max_prompt_tokens,
            ),
            output_tokens=bounded_lognormal(
                workload_rng,
                mean_output,
                args.output_cv,
                4,
                args.max_output_tokens,
            ),
            helix_error_z=workload_rng.gauss(0.0, 1.0),
        ))
    return specs


class Simulator:
    def __init__(self, specs, timing, policy, args):
        self.specs = specs
        self.timing = timing
        self.policy = policy
        self.args = args
        self.replicas = [Replica(0), Replica(1)]
        self.states = {}
        self.rr_next = 0
        self.now = 0.0
        self.pending_samples = []
        self.waiting_samples = []
        self.override_count = 0
        self.rebind_count = 0

    def estimated_remaining(self, state, knowledge):
        true_remaining = max(
            0.0, state.spec.output_tokens - state.generated
        )
        if knowledge == "oracle":
            return true_remaining
        prior_remaining = max(
            1.0,
            self.args.prior_output_tokens - state.generated,
        )
        if knowledge != "helix":
            return prior_remaining
        if state.generated < self.args.helix_reveal_tokens:
            return prior_remaining
        progress = state.generated / max(1.0, state.spec.output_tokens)
        error_scale = self.args.helix_relative_error
        if progress >= 0.50:
            error_scale *= 0.35
        elif progress >= 0.25:
            error_scale *= 0.65
        predicted_total = state.spec.output_tokens * (
            1.0 + error_scale * state.spec.helix_error_z
        )
        return max(1.0, predicted_total - state.generated)

    def start_step(self, replica, now):
        if replica.plan is not None:
            return
        admitted = []
        while (
            replica.waiting
            and len(replica.active) < self.timing.max_num_seqs
        ):
            rid = replica.waiting.popleft()
            replica.active.append(rid)
            state = self.states[rid]
            if state.prefill_start is None:
                state.prefill_start = now
            admitted.append(rid)
        if not replica.active:
            return
        participants = tuple(replica.active)
        new_prefills = tuple(
            rid for rid in admitted
            if self.states[rid].generated == 0
            and self.states[rid].first_token is None
        )
        resumed = len(admitted) - len(new_prefills)
        duration = self.timing.step_sec(
            self.states, participants, new_prefills
        )
        if resumed and self.args.preemption_mode == "recompute":
            # vLLM V1 normally discards a preempted request's KV blocks and
            # recomputes its full prompt plus generated prefix on resumption.
            # Batch those context tokens into one conservative prefill charge.
            recompute_tokens = sum(
                self.states[rid].spec.prompt_tokens
                + self.states[rid].generated
                for rid in admitted
                if rid not in new_prefills
            )
            duration += (
                self.timing.prefill_fixed_sec
                + recompute_tokens
                / max(EPS, self.timing.prefill_tokens_per_sec)
            )
        duration += (
            resumed * max(0.0, self.args.preemption_ms) / 1000.0
        )
        replica.plan = StepPlan(
            start=now,
            end=now + duration,
            participants=participants,
            admitted=tuple(admitted),
        )
        replica.busy_seconds += duration
        replica.decode_slot_seconds += duration * len(participants)

    def finish_step(self, replica):
        plan = replica.plan
        if plan is None:
            return False
        self.now = plan.end
        finished = []
        for rid in plan.participants:
            state = self.states[rid]
            state.generated += 1
            if state.first_token is None:
                state.first_token = plan.end
            if state.last_token is not None:
                state.max_token_gap = max(
                    state.max_token_gap,
                    plan.end - state.last_token,
                )
            state.last_token = plan.end
            if state.generated >= state.spec.output_tokens:
                state.finish = plan.end
                finished.append(rid)
        if finished:
            finished_set = set(finished)
            replica.active = [
                rid for rid in replica.active if rid not in finished_set
            ]
        replica.plan = None
        if (
            not self.policy.objective.startswith("preemptive")
            and (
                not self.policy.objective.startswith("trajectory_queue")
                or not finished
            )
        ):
            self.start_step(replica, plan.end)
        return bool(finished)

    def pending_count(self, replica):
        return replica.pending_count()

    def queue_choice(self):
        counts = [self.pending_count(replica) for replica in self.replicas]
        if counts[0] < counts[1]:
            return 0
        if counts[1] < counts[0]:
            return 1
        return self.rr_next

    def predicted_target_times(self, replica, target, knowledge):
        """Simulate one replica forward with no unknown future arrivals."""
        active = list(replica.active)
        waiting = deque(replica.waiting)
        remaining = {}
        prompt_tokens = {}
        context_tokens = {}
        first_pending = {}

        for rid in active + list(waiting):
            state = self.states[rid]
            remaining[rid] = self.estimated_remaining(state, knowledge)
            prompt_tokens[rid] = state.spec.prompt_tokens
            context_tokens[rid] = state.spec.prompt_tokens + state.generated
            first_pending[rid] = state.first_token is None

        target_id = target.request_id
        remaining[target_id] = (
            float(target.output_tokens)
            if knowledge == "oracle"
            else float(self.args.prior_output_tokens)
        )
        prompt_tokens[target_id] = target.prompt_tokens
        context_tokens[target_id] = target.prompt_tokens
        first_pending[target_id] = True
        waiting.append(target_id)

        now = self.now
        target_first = None
        if replica.plan is not None:
            plan = replica.plan
            now = max(now, plan.end)
            for rid in plan.participants:
                if rid not in remaining:
                    continue
                remaining[rid] -= 1.0
                context_tokens[rid] += 1.0
                if first_pending[rid]:
                    first_pending[rid] = False
            active = [rid for rid in active if remaining[rid] > EPS]

        while True:
            admitted = []
            while waiting and len(active) < self.timing.max_num_seqs:
                rid = waiting.popleft()
                active.append(rid)
                admitted.append(rid)
            if not active:
                raise RuntimeError("Prediction lost the target request")

            contexts = [context_tokens[rid] for rid in active]
            decode = self.timing.decode_step_sec(
                len(active), statistics.fmean(contexts)
            )
            if admitted:
                prefill = (
                    self.timing.prefill_fixed_sec
                    + sum(prompt_tokens[rid] for rid in admitted)
                    / max(EPS, self.timing.prefill_tokens_per_sec)
                )
                duration = prefill + decode
                steps = 1
            else:
                steps = max(
                    1,
                    int(math.ceil(min(remaining[rid] for rid in active))),
                )
                # Context grows by one token per sequence per decode iteration.
                context_increment = max(0.0, steps - 1) / 2.0
                duration = steps * self.timing.decode_step_sec(
                    len(active),
                    statistics.fmean(contexts) + context_increment,
                )

            now += duration
            for rid in active:
                remaining[rid] -= steps
                context_tokens[rid] += steps
                if first_pending[rid]:
                    first_pending[rid] = False
                    if rid == target_id:
                        # With a decode leap the first token occurs after one
                        # iteration, not at the end of all coalesced iterations.
                        target_first = now - duration + duration / steps
            if remaining[target_id] <= EPS:
                return {
                    "ttft": (
                        target_first - self.now
                        if target_first is not None else 0.0
                    ),
                    "completion": now - self.now,
                }
            active = [rid for rid in active if remaining[rid] > EPS]

    def predicted_replica_trajectory(
        self, replica, knowledge, target=None
    ):
        """Forecast first-token and completion times for every pending request.

        Unlike ``predicted_target_times``, this exposes the externality of an
        assignment: admitting one more sequence changes batch size, decode-step
        duration, context growth, and therefore the trajectory of every other
        request on that replica.
        """
        active = list(replica.active)
        waiting = deque(replica.waiting)
        remaining = {}
        prompt_tokens = {}
        context_tokens = {}
        first_pending = {}
        trajectory = {}

        for rid in active + list(waiting):
            state = self.states[rid]
            remaining[rid] = self.estimated_remaining(state, knowledge)
            prompt_tokens[rid] = state.spec.prompt_tokens
            context_tokens[rid] = (
                state.spec.prompt_tokens + state.generated
            )
            first_pending[rid] = state.first_token is None

        target_id = None
        if target is not None:
            target_id = target.request_id
            remaining[target_id] = (
                float(target.output_tokens)
                if knowledge == "oracle"
                else float(self.args.prior_output_tokens)
            )
            prompt_tokens[target_id] = target.prompt_tokens
            context_tokens[target_id] = target.prompt_tokens
            first_pending[target_id] = True
            waiting.append(target_id)

        now = self.now
        if replica.plan is not None:
            plan = replica.plan
            now = max(now, plan.end)
            for rid in plan.participants:
                if rid not in remaining:
                    continue
                remaining[rid] -= 1.0
                context_tokens[rid] += 1.0
                if first_pending[rid]:
                    first_pending[rid] = False
                    trajectory.setdefault(rid, {})["ttft"] = (
                        plan.end - self.now
                    )
                if remaining[rid] <= EPS:
                    trajectory.setdefault(rid, {})["completion"] = (
                        plan.end - self.now
                    )
            active = [rid for rid in active if remaining[rid] > EPS]

        while active or waiting:
            admitted = []
            while waiting and len(active) < self.timing.max_num_seqs:
                rid = waiting.popleft()
                active.append(rid)
                admitted.append(rid)

            contexts = [context_tokens[rid] for rid in active]
            if admitted:
                prefill = (
                    self.timing.prefill_fixed_sec
                    + sum(prompt_tokens[rid] for rid in admitted)
                    / max(EPS, self.timing.prefill_tokens_per_sec)
                )
                steps = 1
                duration = prefill + self.timing.decode_step_sec(
                    len(active), statistics.fmean(contexts)
                )
            else:
                steps = max(
                    1,
                    int(math.ceil(min(remaining[rid] for rid in active))),
                )
                context_increment = max(0.0, steps - 1) / 2.0
                duration = steps * self.timing.decode_step_sec(
                    len(active),
                    statistics.fmean(contexts) + context_increment,
                )

            step_start = now
            now += duration
            for rid in active:
                remaining[rid] -= steps
                context_tokens[rid] += steps
                record = trajectory.setdefault(rid, {})
                if first_pending[rid]:
                    first_pending[rid] = False
                    record["ttft"] = (
                        step_start + duration / steps - self.now
                    )
                if remaining[rid] <= EPS:
                    record["completion"] = now - self.now
            active = [rid for rid in active if remaining[rid] > EPS]

        if target_id is not None and target_id not in trajectory:
            raise RuntimeError("Trajectory prediction lost the target request")
        return trajectory

    def predicted_system_tail_score(
        self, target, target_replica, knowledge
    ):
        """Return smooth tail-risk statistics after a hypothetical assignment."""
        projected_ttfts = []

        # Completed first tokens anchor the online estimate to the latency
        # distribution the policy has actually produced so far.
        for state in self.states.values():
            if state.first_token is not None:
                projected_ttfts.append(
                    state.first_token - state.spec.arrival
                )

        target_ttft = None
        for replica_id, replica in enumerate(self.replicas):
            extra = target if replica_id == target_replica else None
            forecast = self.predicted_replica_trajectory(
                replica, knowledge, target=extra
            )
            for rid, prediction in forecast.items():
                if "ttft" not in prediction:
                    continue
                spec = target if rid == target.request_id else self.states[rid].spec
                latency = self.now + prediction["ttft"] - spec.arrival
                projected_ttfts.append(latency)
                if rid == target.request_id:
                    target_ttft = latency

        ordered = sorted(projected_ttfts)
        if not ordered or target_ttft is None:
            raise RuntimeError("Incomplete projected TTFT trajectory")
        p95 = percentile(ordered, 95)
        tail_5 = ordered[max(0, int(math.floor(0.95 * len(ordered)))):]
        tail_10 = ordered[max(0, int(math.floor(0.90 * len(ordered)))):]
        return {
            "p95": p95,
            "cvar95": statistics.fmean(tail_5),
            "cvar90": statistics.fmean(tail_10),
            "mean": statistics.fmean(ordered),
            "target": target_ttft,
        }

    def _migration_enabled(self):
        # Only length-aware policies get to use deferred binding / continuous
        # rebalancing. queue_size and round_robin are left untouched so they
        # remain a fair, unmodified baseline.
        return (
            self.policy.knowledge in ("helix", "oracle")
            and not self.policy.objective.startswith("trajectory_queue")
        )

    def trajectory_dispatch(self):
        """Centrally rebind unstarted work using predicted request trajectories.

        Requests that have begun prefill never move. The remaining pool is
        ordered by predicted size, with optional quadratic aging to prevent
        long requests from starving, and then greedily placed where its exact
        predicted TTFT is smallest. This models a practical deferred-binding
        router rather than GPU-side preemption.
        """
        if not self.policy.objective.startswith("trajectory_queue"):
            return
        pool = []
        for replica in self.replicas:
            pool.extend(replica.waiting)
            replica.waiting.clear()
        if not pool:
            for replica in self.replicas:
                self.start_step(replica, self.now)
            return

        def priority(rid):
            state = self.states[rid]
            remaining = self.estimated_remaining(
                state, self.policy.knowledge
            )
            if self.policy.objective.endswith("_ljf"):
                return (-remaining, state.spec.arrival, rid)
            if self.policy.objective == "trajectory_queue_sjf":
                return (remaining, state.spec.arrival, rid)
            wait_age = max(0.0, self.now - state.spec.arrival)
            if "_deadline" in self.policy.objective:
                deadline = float(
                    self.policy.objective.rsplit("_deadline", 1)[1]
                )
                if wait_age >= deadline:
                    return (0, state.spec.arrival, rid)
                return (1, remaining, rid)
            aging = (1.0 + wait_age / 5.0) ** 2
            return (remaining / aging, state.spec.arrival, rid)

        import heapq

        # Each heap entry is the predicted time at which one decode lane can
        # admit another request. This is the same list-scheduling abstraction
        # used by production load balancers, augmented with measured
        # batch/context-dependent step duration.
        lane_heaps = []
        step_seconds = []
        for replica in self.replicas:
            contexts = [
                self.states[rid].spec.prompt_tokens
                + self.states[rid].generated
                for rid in replica.active
            ]
            expected_batch = min(
                self.timing.max_num_seqs,
                max(1, len(replica.active) + len(pool)),
            )
            step_sec = self.timing.decode_step_sec(
                expected_batch,
                statistics.fmean(contexts) if contexts else 0.0,
            )
            step_seconds.append(step_sec)
            in_flight = (
                max(0.0, replica.plan.end - self.now)
                if replica.plan is not None else 0.0
            )
            lanes = [
                in_flight
                + max(
                    0.0,
                    self.estimated_remaining(
                        self.states[rid], self.policy.knowledge
                    ) - (1.0 if replica.plan is not None else 0.0),
                ) * step_sec
                for rid in replica.active
            ]
            lanes.extend(
                [0.0] * (self.timing.max_num_seqs - len(lanes))
            )
            heapq.heapify(lanes)
            lane_heaps.append(lanes)

        if self.policy.objective.startswith("trajectory_queue_protect"):
            protected_percent = int(
                self.policy.objective.removeprefix(
                    "trajectory_queue_protect"
                )
            )
            by_age = sorted(
                pool,
                key=lambda rid: (
                    self.states[rid].spec.arrival,
                    rid,
                ),
            )
            protected_count = max(
                1,
                int(math.ceil(
                    len(by_age) * protected_percent / 100.0
                )),
            )
            ordered_pool = (
                by_age[:protected_count]
                + sorted(by_age[protected_count:], key=priority)
            )
        elif self.policy.objective.startswith("trajectory_queue_window"):
            window = int(
                self.policy.objective.removeprefix(
                    "trajectory_queue_window"
                ).split("_", 1)[0]
            )
            by_age = sorted(
                pool,
                key=lambda rid: (
                    self.states[rid].spec.arrival,
                    rid,
                ),
            )
            ordered_pool = []
            for start in range(0, len(by_age), window):
                cohort = by_age[start:start + window]
                ordered_pool.extend(sorted(cohort, key=priority))
        else:
            ordered_pool = sorted(pool, key=priority)

        for rid in ordered_pool:
            state = self.states[rid]
            admission_times = [
                lane_heaps[replica_id][0]
                + self.timing.prefill_fixed_sec
                + state.spec.prompt_tokens
                / max(EPS, self.timing.prefill_tokens_per_sec)
                for replica_id in (0, 1)
            ]
            choice = 0 if admission_times[0] < admission_times[1] else (
                1 if admission_times[1] < admission_times[0]
                else self.queue_choice()
            )
            self.replicas[choice].waiting.append(rid)
            if state.replica != choice:
                self.rebind_count += 1
            state.replica = choice
            available = heapq.heappop(lane_heaps[choice])
            finish = (
                available
                + self.timing.prefill_fixed_sec
                + state.spec.prompt_tokens
                / max(EPS, self.timing.prefill_tokens_per_sec)
                + self.estimated_remaining(
                    state, self.policy.knowledge
                ) * step_seconds[choice]
            )
            heapq.heappush(lane_heaps[choice], finish)

        for replica in self.replicas:
            self.start_step(replica, self.now)

    def preemptive_schedule(self):
        """Run a length-aware, first-token-first decode scheduler.

        Scheduling happens only at decode-step boundaries. Requests that have
        not produced a token outrank decode work; the remainder use an aged
        shortest-remaining-time priority. In the default vLLM-like mode, a
        resumed request pays batched recomputation for its prompt and generated
        prefix plus the explicit residual ``preemption_ms`` overhead.
        """
        if not self.policy.objective.startswith("preemptive"):
            return

        # Deferred binding keeps unstarted work balanced even when one replica
        # finishes a run of short requests earlier than the other.
        while True:
            counts = [replica.pending_count() for replica in self.replicas]
            donor_id = 0 if counts[0] > counts[1] + 1 else (
                1 if counts[1] > counts[0] + 1 else None
            )
            if donor_id is None:
                break
            receiver_id = 1 - donor_id
            donor = self.replicas[donor_id]
            if not donor.waiting:
                break
            rid = donor.waiting.pop()
            self.replicas[receiver_id].waiting.append(rid)
            self.states[rid].replica = receiver_id
            self.rebind_count += 1

        for replica in self.replicas:
            if replica.plan is not None:
                continue
            pool = list(dict.fromkeys(
                replica.active + list(replica.waiting)
            ))
            if not pool:
                continue
            decode_cohort = {}
            window_prefix = next(
                (
                    prefix for prefix in (
                        "preemptive_first_token_window",
                        "preemptive_slo_window",
                        "preemptive_guarded_window",
                    )
                    if self.policy.objective.startswith(prefix)
                ),
                None,
            )
            if window_prefix is not None:
                window = int(
                    self.policy.objective.removeprefix(window_prefix)
                )
                decode_ids = sorted(
                    (
                        rid for rid in pool
                        if self.states[rid].first_token is not None
                    ),
                    key=lambda rid: (
                        self.states[rid].spec.arrival,
                        rid,
                    ),
                )
                decode_cohort = {
                    rid: rank // window
                    for rank, rid in enumerate(decode_ids)
                }
            old_active = set(replica.active)

            def priority(rid):
                state = self.states[rid]
                if self.policy.objective.startswith("preemptive_guarded"):
                    if (
                        state.first_token is not None
                        and self.now - state.last_token
                        >= self.args.preemptive_itl_deadline_ms / 1000.0
                    ):
                        return (
                            0, state.last_token, 0, 0,
                            state.spec.arrival, rid,
                        )
                    if state.first_token is None:
                        overdue = (
                            self.now - state.spec.arrival
                            >= self.args.preemptive_ttft_deadline_ms / 1000.0
                        )
                        return (
                            1 if overdue else 4,
                            state.spec.arrival, 0, 0,
                            state.spec.arrival, rid,
                        )
                    remaining = self.estimated_remaining(
                        state, self.policy.knowledge
                    )
                    return (
                        2 if rid in old_active else 3,
                        decode_cohort.get(rid, -1), remaining, 0,
                        state.spec.arrival, rid,
                    )
                if self.policy.objective.startswith("preemptive_slo"):
                    if state.first_token is None:
                        deadline = (
                            state.spec.arrival
                            + self.args.preemptive_ttft_deadline_ms / 1000.0
                        )
                        request_class = 0
                    else:
                        deadline = (
                            state.last_token
                            + self.args.preemptive_itl_deadline_ms / 1000.0
                        )
                        request_class = 1
                    remaining = self.estimated_remaining(
                        state, self.policy.knowledge
                    )
                    overdue = deadline <= self.now + EPS
                    return (
                        0 if overdue else 1,
                        deadline if overdue else 0.0,
                        0 if rid in old_active else 1,
                        request_class,
                        decode_cohort.get(rid, -1),
                        remaining,
                        state.spec.arrival,
                        rid,
                    )
                if state.first_token is None:
                    return (0, state.spec.arrival, rid)
                if self.policy.objective == "preemptive_first_token_fcfs":
                    return (1, state.spec.arrival, rid)
                remaining = self.estimated_remaining(
                    state, self.policy.knowledge
                )
                if decode_cohort:
                    return (
                        1,
                        decode_cohort[rid],
                        remaining,
                        state.spec.arrival,
                        rid,
                    )
                age = max(0.0, self.now - state.spec.arrival)
                aged_remaining = remaining / (1.0 + age / 10.0) ** 2
                return (1, aged_remaining, state.spec.arrival, rid)

            selected = set(sorted(pool, key=priority)[
                :self.timing.max_num_seqs
            ])
            preempted = old_active - selected
            for rid in preempted:
                self.states[rid].preemptions += 1

            # Keep continuously selected sequences resident. Newly selected or
            # resumed sequences are admitted through start_step(), which
            # charges prefill or resume cost as appropriate.
            replica.active = [
                rid for rid in replica.active if rid in selected
            ]
            selected_waiting = [
                rid for rid in sorted(pool, key=priority)
                if rid in selected and rid not in old_active
            ]
            deferred = [
                rid for rid in sorted(pool, key=priority)
                if rid not in selected
            ]
            replica.waiting = deque(selected_waiting + deferred)
            self.start_step(replica, self.now)

    def steal_idle(self):
        """Move an unstarted request when its assigned replica is idle."""
        if not self._migration_enabled():
            return
        for idle_idx, donor_idx in ((0, 1), (1, 0)):
            idle = self.replicas[idle_idx]
            donor = self.replicas[donor_idx]
            while not idle.active and idle.plan is None and donor.waiting:
                rid = donor.waiting.popleft()
                idle.waiting.append(rid)
                self.states[rid].replica = idle_idx
                self.rebind_count += 1
                self.start_step(idle, self.now)

    def predictive_migrate(self, max_moves_per_side=6):
        """Re-evaluate still-unstarted requests using current length estimates."""
        if not self._migration_enabled():
            return
        knowledge = self.policy.knowledge
        key = (
            "ttft"
            if self.policy.objective == "ttft"
            or self.policy.objective.startswith("trajectory")
            else "completion"
        )
        for src_idx, dst_idx in ((0, 1), (1, 0)):
            src = self.replicas[src_idx]
            dst = self.replicas[dst_idx]
            moved = 0
            checked = 0
            while (
                src.waiting
                and moved < max_moves_per_side
                and checked < max_moves_per_side + 2
            ):
                checked += 1
                candidate_rid = src.waiting[-1]
                spec = self.states[candidate_rid].spec
                src.waiting.pop()
                try:
                    pred_stay = self.predicted_target_times(
                        src, spec, knowledge
                    )
                    pred_move = self.predicted_target_times(
                        dst, spec, knowledge
                    )
                except RuntimeError:
                    src.waiting.append(candidate_rid)
                    break
                gain = pred_stay[key] - pred_move[key]
                count_after_diff = (
                    (dst.pending_count() + 1) - src.pending_count()
                )
                if (
                    gain >= self.policy.min_gain_sec
                    and count_after_diff <= self.policy.max_count_imbalance
                ):
                    dst.waiting.append(candidate_rid)
                    self.states[candidate_rid].replica = dst_idx
                    self.rebind_count += 1
                    self.start_step(dst, self.now)
                    moved += 1
                else:
                    src.waiting.append(candidate_rid)
                    break

    def choose_replica(self, spec):
        queue_choice = self.queue_choice()
        if self.policy.name == "round_robin":
            choice = self.rr_next
            self.rr_next = 1 - self.rr_next
            return choice
        if self.policy.name == "queue_size":
            self.rr_next = 1 - self.rr_next
            return queue_choice
        if self.policy.objective.startswith("trajectory_queue"):
            # Binding is intentionally deferred until trajectory_dispatch(),
            # after the request has joined the movable central pool.
            self.rr_next = 1 - self.rr_next
            return queue_choice
        if self.policy.objective.startswith("preemptive"):
            self.rr_next = 1 - self.rr_next
            return queue_choice
        if self.policy.name == "least_work":
            scores = []
            for replica in self.replicas:
                work = sum(
                    self.estimated_remaining(
                        self.states[rid], self.policy.knowledge
                    )
                    for rid in replica.active + list(replica.waiting)
                )
                scores.append(work)
            choice = 0 if scores[0] < scores[1] else (
                1 if scores[1] < scores[0] else queue_choice
            )
        elif self.policy.objective.startswith("trajectory"):
            projected = [
                self.predicted_system_tail_score(
                    spec, replica_id, self.policy.knowledge
                )
                for replica_id in (0, 1)
            ]
            if self.policy.objective == "trajectory_p95":
                scores = [
                    (
                        prediction["p95"],
                        prediction["cvar95"],
                        prediction["cvar90"],
                        prediction["target"],
                    )
                    for prediction in projected
                ]
            else:
                scores = [
                    (
                        prediction["cvar90"],
                        prediction["p95"],
                        prediction["target"],
                    )
                    for prediction in projected
                ]
            best = 0 if scores[0] < scores[1] else (
                1 if scores[1] < scores[0] else queue_choice
            )
            # Tail scores are lexicographic. The request-count guard remains
            # available, but a seconds-valued minimum-gain threshold does not
            # apply to the secondary tuple components.
            count_difference = (
                self.pending_count(self.replicas[best])
                - self.pending_count(self.replicas[queue_choice])
            )
            choice = (
                best
                if count_difference <= self.policy.max_count_imbalance
                else queue_choice
            )
        else:
            predicted = [
                self.predicted_target_times(
                    replica, spec, self.policy.knowledge
                )
                for replica in self.replicas
            ]
            key = "ttft" if self.policy.objective == "ttft" else "completion"
            scores = [prediction[key] for prediction in predicted]
            best = 0 if scores[0] < scores[1] else (
                1 if scores[1] < scores[0] else queue_choice
            )
            gain = scores[queue_choice] - scores[best]
            count_difference = (
                self.pending_count(self.replicas[best])
                - self.pending_count(self.replicas[queue_choice])
            )
            choice = (
                best
                if gain >= self.policy.min_gain_sec
                and count_difference <= self.policy.max_count_imbalance
                else queue_choice
            )
        if choice != queue_choice:
            self.override_count += 1
        self.rr_next = 1 - self.rr_next
        return choice

    def run(self):
        next_arrival = 0
        while next_arrival < len(self.specs) or any(
            replica.plan is not None for replica in self.replicas
        ):
            arrival_time = (
                self.specs[next_arrival].arrival
                if next_arrival < len(self.specs) else math.inf
            )
            next_step_time = min(
                (
                    replica.plan.end
                    for replica in self.replicas
                    if replica.plan is not None
                ),
                default=math.inf,
            )
            if next_step_time <= arrival_time + EPS:
                self.now = next_step_time
                slot_freed = False
                for replica in self.replicas:
                    if (
                        replica.plan is not None
                        and replica.plan.end <= next_step_time + EPS
                    ):
                        if self.finish_step(replica):
                            slot_freed = True
                if self.policy.objective.startswith("preemptive"):
                    self.preemptive_schedule()
                elif self.policy.objective.startswith("trajectory_queue"):
                    if slot_freed:
                        self.trajectory_dispatch()
                else:
                    self.steal_idle()
                    if slot_freed:
                        self.predictive_migrate()
                continue

            spec = self.specs[next_arrival]
            self.now = spec.arrival
            self.pending_samples.append(sum(
                replica.pending_count() for replica in self.replicas
            ))
            self.waiting_samples.append(sum(
                len(replica.waiting) for replica in self.replicas
            ))
            replica_id = self.choose_replica(spec)
            state = RequestState(spec=spec, replica=replica_id)
            self.states[spec.request_id] = state
            replica = self.replicas[replica_id]
            replica.waiting.append(spec.request_id)
            replica.max_pending = max(
                replica.max_pending, replica.pending_count()
            )
            replica.max_waiting = max(
                replica.max_waiting, len(replica.waiting)
            )
            if self.policy.objective.startswith("preemptive"):
                self.preemptive_schedule()
            elif self.policy.objective.startswith("trajectory_queue"):
                self.trajectory_dispatch()
            else:
                self.start_step(replica, self.now)
                self.steal_idle()
                self.predictive_migrate()
            next_arrival += 1

        return self.summarize()

    def summarize(self):
        ordered = [self.states[spec.request_id] for spec in self.specs]
        warm = int(len(ordered) * self.args.measurement_trim)
        measured = ordered[warm: len(ordered) - warm] if warm else ordered
        latencies = [
            state.finish - state.spec.arrival for state in measured
        ]
        ttfts = [
            state.first_token - state.spec.arrival for state in measured
        ]
        mean_itls = [
            (state.finish - state.first_token)
            / max(1, state.spec.output_tokens - 1)
            for state in measured
        ]
        max_itls = [state.max_token_gap for state in measured]
        start = min(state.spec.arrival for state in measured)
        end = max(state.finish for state in measured)
        wall = max(EPS, end - start)
        output_tokens = sum(state.spec.output_tokens for state in measured)
        return {
            "policy": self.policy.name,
            "knowledge": self.policy.knowledge,
            "objective": self.policy.objective,
            "binding_mode": (
                "central_preemptive"
                if self.policy.objective.startswith("preemptive")
                else (
                    "central_deferred"
                    if self.policy.objective.startswith("trajectory_queue")
                    else (
                        "predictive_rebinding"
                        if self._migration_enabled() else "immediate"
                    )
                )
            ),
            "queue_discipline": (
                self.policy.objective
                if self.policy.objective.startswith(
                    ("trajectory_queue", "preemptive")
                )
                else "local_fcfs"
            ),
            "min_gain_sec": self.policy.min_gain_sec,
            "max_count_imbalance": self.policy.max_count_imbalance,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies),
            "ttft_p50": percentile(ttfts, 50),
            "ttft_p95": percentile(ttfts, 95),
            "ttft_p99": percentile(ttfts, 99),
            "ttft_max": max(ttfts),
            "itl_p95": percentile(mean_itls, 95),
            "max_itl_p95": percentile(max_itls, 95),
            "measured_output_tokens": output_tokens,
            "tokens_per_sec": output_tokens / wall,
            "mean_gpu_busy": statistics.fmean(
                replica.busy_seconds / max(EPS, self.now)
                for replica in self.replicas
            ),
            "mean_slot_occupancy": statistics.fmean(
                replica.decode_slot_seconds
                / max(EPS, self.now * self.timing.max_num_seqs)
                for replica in self.replicas
            ),
            "pending_p95": percentile(self.pending_samples, 95),
            "waiting_p95": percentile(self.waiting_samples, 95),
            "waiting_fraction": sum(
                value > 0 for value in self.waiting_samples
            ) / max(1, len(self.waiting_samples)),
            "assignments": [
                sum(state.replica == replica_id for state in ordered)
                for replica_id in (0, 1)
            ],
            "max_pending": [
                replica.max_pending for replica in self.replicas
            ],
            "overrides": self.override_count,
            "rebindings": self.rebind_count,
            "preemptions": sum(
                state.preemptions for state in ordered
            ),
        }


def policy_candidates(args):
    candidates = [
        Policy("round_robin", "prior"),
        Policy("queue_size", "prior"),
        Policy("least_work", "oracle"),
        Policy("completion_oracle", "oracle", "completion"),
        Policy("ttft_oracle", "oracle", "ttft"),
        Policy("completion_helix", "helix", "completion"),
        Policy("ttft_helix", "helix", "ttft"),
        Policy(
            "trajectory_tail_oracle",
            "oracle",
            "trajectory_p95",
        ),
        Policy(
            "trajectory_window6_oracle",
            "oracle",
            "trajectory_queue_window6",
        ),
        Policy(
            "trajectory_deadline10_oracle",
            "oracle",
            "trajectory_queue_window6_deadline10",
        ),
        Policy(
            "trajectory_deadline10_helix",
            "helix",
            "trajectory_queue_window6_deadline10",
        ),
        Policy(
            "preemptive_trajectory_oracle",
            "oracle",
            "preemptive_guarded_window12",
        ),
        Policy(
            "preemptive_trajectory_helix",
            "helix",
            "preemptive_guarded_window6",
        ),
        Policy(
            "aggressive_ttft_oracle",
            "oracle",
            "preemptive_first_token_window12",
        ),
        Policy(
            "aggressive_ttft_helix",
            "helix",
            "preemptive_first_token_window6",
        ),
    ]
    for objective in ("ttft", "completion"):
        for imbalance in args.search_imbalances:
            for gain in args.search_gains:
                candidates.append(Policy(
                    name=f"guarded_{objective}_helix",
                    knowledge="helix",
                    objective=objective,
                    min_gain_sec=gain,
                    max_count_imbalance=imbalance,
                ))
                candidates.append(Policy(
                    name=f"guarded_{objective}_oracle",
                    knowledge="oracle",
                    objective=objective,
                    min_gain_sec=gain,
                    max_count_imbalance=imbalance,
                ))
    return candidates


def scaled_timing(timing, scale):
    return TimingModel(
        max_num_seqs=timing.max_num_seqs,
        prefill_fixed_sec=timing.prefill_fixed_sec * scale,
        prefill_tokens_per_sec=timing.prefill_tokens_per_sec / scale,
        decode_fixed_sec=timing.decode_fixed_sec * scale,
        decode_linear_sec=timing.decode_linear_sec * scale,
        decode_quadratic_sec=timing.decode_quadratic_sec * scale,
        context_scale_tokens=timing.context_scale_tokens,
        context_slowdown=timing.context_slowdown,
    )


def calibrate_timing_scale(specs, timing, args, target_tokens_per_sec):
    """Match the measured queue-size throughput before comparing policies."""
    low, high = 0.20, 2.00
    best = None
    queue_policy = Policy("queue_size", "prior")
    for _ in range(10):
        scale = 0.5 * (low + high)
        candidate_timing = scaled_timing(timing, scale)
        result = Simulator(
            specs=specs,
            timing=candidate_timing,
            policy=queue_policy,
            args=args,
        ).run()
        error = abs(result["tokens_per_sec"] - target_tokens_per_sec)
        if best is None or error < best[0]:
            best = (error, scale, candidate_timing, result)
        if result["tokens_per_sec"] < target_tokens_per_sec:
            high = scale
        else:
            low = scale
    return best[2], best[1], best[3]


def scaled_arrivals(specs, scale):
    return [
        RequestSpec(
            request_id=spec.request_id,
            arrival=spec.arrival * scale,
            prompt_tokens=spec.prompt_tokens,
            output_tokens=spec.output_tokens,
            helix_error_z=spec.helix_error_z,
        )
        for spec in specs
    ]


def aggregate_results(rows):
    grouped = {}
    for row in rows:
        key = (
            row["policy"],
            row["knowledge"],
            row["objective"],
            row["min_gain_sec"],
            row["max_count_imbalance"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for key, values in grouped.items():
        record = {
            "policy": key[0],
            "knowledge": key[1],
            "objective": key[2],
            "binding_mode": values[0]["binding_mode"],
            "queue_discipline": values[0]["queue_discipline"],
            "min_gain_sec": key[3],
            "max_count_imbalance": key[4],
            "n_seeds": len(values),
        }
        for metric in (
            "p50", "p95", "p99", "max",
            "ttft_p50", "ttft_p95", "ttft_p99", "ttft_max",
            "itl_p95", "max_itl_p95",
            "measured_output_tokens", "tokens_per_sec",
            "mean_gpu_busy", "mean_slot_occupancy",
            "pending_p95", "waiting_p95", "waiting_fraction",
            "overrides", "rebindings", "preemptions",
        ):
            record[metric] = statistics.fmean(
                float(value[metric]) for value in values
            )
        aggregates.append(record)
    baseline = next(
        row for row in aggregates if row["policy"] == "queue_size"
    )
    for row in aggregates:
        row["p95_delta_vs_queue_sec"] = row["p95"] - baseline["p95"]
        row["ttft_p95_delta_vs_queue_sec"] = (
            row["ttft_p95"] - baseline["ttft_p95"]
        )
        row["ttft_p95_improvement_pct_vs_queue"] = (
            100.0
            * (baseline["ttft_p95"] - row["ttft_p95"])
            / max(EPS, baseline["ttft_p95"])
        )
    return aggregates


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def parse_int_list(text):
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CPU discrete-event simulator for Helix routing policies"
    )
    parser.add_argument("--calibration-csv")
    parser.add_argument("--request-csv")
    parser.add_argument("--output-dir", default="simulation_results")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--seeds", type=parse_int_list, default=(0, 1, 2))
    parser.add_argument("--offered-rps", type=float)
    parser.add_argument("--burstiness", type=float)
    parser.add_argument("--max-num-seqs", type=int, default=24)
    parser.add_argument("--prefill-fixed-ms", type=float, default=6.0)
    parser.add_argument("--prefill-tokens-per-sec", type=float, default=15000.0)
    parser.add_argument("--decode-fixed-ms", type=float)
    parser.add_argument("--decode-linear-ms", type=float)
    parser.add_argument("--decode-quadratic-ms", type=float, default=0.0)
    parser.add_argument("--context-slowdown", type=float, default=0.15)
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Optional multiplier for synthetic arrival offsets.",
    )
    parser.add_argument(
        "--no-auto-calibrate",
        action="store_true",
        help="Do not scale timing to measured queue-size output throughput.",
    )
    parser.add_argument("--prompt-cv", type=float, default=1.0)
    parser.add_argument("--output-cv", type=float, default=0.9)
    parser.add_argument("--workload-seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--prior-output-tokens", type=float)
    parser.add_argument("--helix-reveal-tokens", type=int, default=8)
    parser.add_argument("--helix-relative-error", type=float, default=0.10)
    parser.add_argument(
        "--preemption-ms",
        type=float,
        default=5.0,
        help=(
            "Residual scheduler penalty per resumed sequence, in addition to "
            "context recomputation when --preemption-mode=recompute."
        ),
    )
    parser.add_argument(
        "--preemption-mode",
        choices=("recompute", "retained"),
        default="recompute",
        help=(
            "Model vLLM V1 recomputation on resume (default), or retain KV "
            "state as an optimistic research-only counterfactual."
        ),
    )
    parser.add_argument(
        "--preemptive-ttft-deadline-ms",
        type=float,
        default=20000.0,
        help=(
            "Waiting time after which a guarded policy may preempt decode "
            "work to protect TTFT."
        ),
    )
    parser.add_argument(
        "--preemptive-itl-deadline-ms",
        type=float,
        default=5000.0,
        help=(
            "Maximum intended token gap before guarded policies prioritize "
            "resuming a paused sequence."
        ),
    )
    parser.add_argument(
        "--search-gains",
        type=parse_float_list,
        default=(0.0, 0.1, 0.25, 0.5),
    )
    parser.add_argument(
        "--search-imbalances",
        type=parse_int_list,
        default=(0, 2, 4),
    )
    parser.add_argument("--measurement-trim", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.requests <= 0 or args.max_num_seqs <= 0:
        parser.error("requests and max-num-seqs must be positive")
    if (
        args.preemption_ms < 0
        or args.preemptive_ttft_deadline_ms <= 0
        or args.preemptive_itl_deadline_ms <= 0
    ):
        parser.error("preemption cost must be non-negative and deadlines positive")
    if not 0 <= args.measurement_trim < 0.5:
        parser.error("measurement-trim must be in [0, 0.5)")
    return args


def main(argv=None):
    args = parse_args(argv)
    calibration = {
        "n_requests": args.requests,
        "offered_rps": 3.4,
        "burstiness": 0.05,
        "mean_prompt_tokens": 230.0,
        "mean_output_tokens": 345.0,
        "target_tokens_per_sec": 700.0,
        "target_ttft_p95": 5.0,
        "target_pending_p95": 48.0,
        "target_waiting_p95": 4.0,
        "decode_reference_batch": 12.0,
        "decode_reference_step_sec": 0.030,
        "decode_fixed_sec": 0.0075,
        "decode_linear_sec": 0.001875,
    }
    if args.calibration_csv:
        calibration.update(read_aggregate_calibration(args.calibration_csv))
    if args.requests == 300 and args.request_csv:
        args.requests = len(read_request_trace(args.request_csv))
    if args.prior_output_tokens is None:
        args.prior_output_tokens = calibration["mean_output_tokens"]

    timing = TimingModel(
        max_num_seqs=args.max_num_seqs,
        prefill_fixed_sec=args.prefill_fixed_ms / 1000.0,
        prefill_tokens_per_sec=args.prefill_tokens_per_sec,
        decode_fixed_sec=(
            args.decode_fixed_ms / 1000.0
            if args.decode_fixed_ms is not None
            else calibration["decode_fixed_sec"]
        ),
        decode_linear_sec=(
            args.decode_linear_ms / 1000.0
            if args.decode_linear_ms is not None
            else calibration["decode_linear_sec"]
        ),
        decode_quadratic_sec=args.decode_quadratic_ms / 1000.0,
        context_slowdown=args.context_slowdown,
    )

    timing_scale = 1.0
    arrival_scale = args.arrival_scale
    calibration_check = None
    if args.calibration_csv and not args.no_auto_calibrate:
        calibration_specs = generate_workload(args, calibration, args.seeds[0])
        if arrival_scale != 1.0:
            calibration_specs = scaled_arrivals(
                calibration_specs, arrival_scale
            )
        timing, timing_scale, calibration_check = calibrate_timing_scale(
            specs=calibration_specs,
            timing=timing,
            args=args,
            target_tokens_per_sec=calibration["target_tokens_per_sec"],
        )

    print("Calibration:")
    print(json.dumps({
        **calibration,
        "timing_model": asdict(timing),
        "timing_scale": timing_scale,
        "arrival_scale": arrival_scale,
        "queue_size_calibration_check": calibration_check,
        "prior_output_tokens": args.prior_output_tokens,
        "preemption_ms": args.preemption_ms,
        "preemption_mode": args.preemption_mode,
        "preemptive_ttft_deadline_ms": (
            args.preemptive_ttft_deadline_ms
        ),
        "preemptive_itl_deadline_ms": (
            args.preemptive_itl_deadline_ms
        ),
    }, indent=2))

    policies = policy_candidates(args)
    rows = []
    for seed in args.seeds:
        specs = generate_workload(args, calibration, seed)
        if arrival_scale != 1.0:
            specs = scaled_arrivals(specs, arrival_scale)
        seed_rows = []
        for index, policy in enumerate(policies, start=1):
            result = Simulator(
                specs=specs,
                timing=timing,
                policy=policy,
                args=args,
            ).run()
            result["seed"] = seed
            rows.append(result)
            seed_rows.append(result)
        measured_work = {
            int(result["measured_output_tokens"]) for result in seed_rows
        }
        if len(measured_work) != 1:
            raise AssertionError(
                f"Policies received different token work for seed {seed}: "
                f"{sorted(measured_work)}"
            )
        print(f"Completed seed {seed}: {len(policies)} policies")

    aggregates = aggregate_results(rows)
    ranked = sorted(
        aggregates,
        key=lambda row: (row["ttft_p95"], row["p95"]),
    )
    queue = next(row for row in aggregates if row["policy"] == "queue_size")
    best_helix = next(
        row for row in ranked if row["knowledge"] == "helix"
    )
    best_oracle = next(
        row for row in ranked if row["knowledge"] == "oracle"
    )
    balanced_oracles = [
        row for row in ranked
        if row["knowledge"] == "oracle"
        and row["ttft_p99"] <= queue["ttft_p99"] + EPS
        and row["p95"] <= queue["p95"] + EPS
    ]
    best_balanced_oracle = (
        balanced_oracles[0] if balanced_oracles else None
    )
    streaming_safe_oracles = [
        row for row in ranked
        if row["knowledge"] == "oracle"
        and row["ttft_p99"] <= queue["ttft_p99"] + EPS
        and row["p99"] <= queue["p99"] + EPS
        and row["max_itl_p95"] <= (
            args.preemptive_itl_deadline_ms / 1000.0 + 0.25
        )
    ]
    best_streaming_safe_oracle = (
        streaming_safe_oracles[0] if streaming_safe_oracles else None
    )
    streaming_safe_helix = [
        row for row in ranked
        if row["knowledge"] == "helix"
        and row["ttft_p99"] <= queue["ttft_p99"] + EPS
        and row["p99"] <= queue["p99"] + EPS
        and row["max_itl_p95"] <= (
            args.preemptive_itl_deadline_ms / 1000.0 + 0.25
        )
    ]
    best_streaming_safe_helix = (
        streaming_safe_helix[0] if streaming_safe_helix else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "simulation_runs.csv", rows)
    write_csv(output_dir / "simulation_policy_search.csv", ranked)
    report = {
        "calibration": calibration,
        "timing_model": asdict(timing),
        "timing_scale": timing_scale,
        "arrival_scale": arrival_scale,
        "queue_size_calibration_check": calibration_check,
        "preemptive_scheduler": {
            "resume_cost_ms": args.preemption_ms,
            "ttft_deadline_ms": args.preemptive_ttft_deadline_ms,
            "itl_rescue_ms": args.preemptive_itl_deadline_ms,
        },
        "queue_size": queue,
        "best_helix": best_helix,
        "best_oracle": best_oracle,
        "best_balanced_oracle": best_balanced_oracle,
        "best_streaming_safe_oracle": best_streaming_safe_oracle,
        "best_streaming_safe_helix": best_streaming_safe_helix,
        "ranking": ranked,
    }
    (output_dir / "simulation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBest policies by mean TTFT p95:")
    for row in ranked[:12]:
        print(
            f"{row['policy']:<30} {row['knowledge']:<7} "
            f"obj={row['objective']:<10} "
            f"gain={row['min_gain_sec']:<4.2f} "
            f"imb={row['max_count_imbalance']:<3} "
            f"TTFT95={row['ttft_p95']:.3f}s "
            f"d={row['ttft_p95_delta_vs_queue_sec']:+.3f}s "
            f"improve={row['ttft_p95_improvement_pct_vs_queue']:+.1f}% "
            f"TTFT99={row['ttft_p99']:.3f}s "
            f"lat95={row['p95']:.3f}s "
            f"maxITL95={row['max_itl_p95']:.3f}s"
        )
    print(f"\nQueue size TTFT p95: {queue['ttft_p95']:.3f}s")
    recommended_helix = best_streaming_safe_helix or best_helix
    print(
        "Recommended streaming-safe Helix policy: "
        f"{recommended_helix['policy']}, "
        f"objective={recommended_helix['objective']}, "
        f"TTFT95 improvement="
        f"{recommended_helix['ttft_p95_improvement_pct_vs_queue']:.1f}%"
    )
    print(
        "Best Oracle policy: "
        f"{best_oracle['policy']}, objective={best_oracle['objective']}, "
        f"min_gain={best_oracle['min_gain_sec']}, "
        f"max_count_imbalance={best_oracle['max_count_imbalance']}"
    )
    if best_balanced_oracle is not None:
        print(
            "Best completion-tail-safe Oracle (streaming gaps ignored): "
            f"{best_balanced_oracle['policy']}, "
            f"TTFT95 improvement="
            f"{best_balanced_oracle['ttft_p95_improvement_pct_vs_queue']:.1f}%"
        )
    if best_streaming_safe_oracle is not None:
        print(
            "Best streaming-guarded Oracle: "
            f"{best_streaming_safe_oracle['policy']}, "
            f"TTFT95 improvement="
            f"{best_streaming_safe_oracle['ttft_p95_improvement_pct_vs_queue']:.1f}%, "
            f"max-ITL p95={best_streaming_safe_oracle['max_itl_p95']:.3f}s"
        )
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
