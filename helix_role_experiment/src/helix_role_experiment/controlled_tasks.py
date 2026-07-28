from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import deterministic_id


@dataclass(frozen=True)
class ComputationalState:
    step: int
    total_steps: int
    state_payload: dict[str, Any]
    valid_next: tuple[str, ...]
    remaining_distance: int
    resolved_dependencies: int
    total_dependencies: int
    invalid_commitments: int = 0
    answer_known: bool = False
    verification_remaining: bool = True

    @property
    def structural_progress(self) -> float:
        if self.total_steps <= 0:
            return 1.0
        return float(np.clip(1.0 - self.remaining_distance / self.total_steps, 0.0, 1.0))

    @property
    def dependency_progress(self) -> float:
        return self.resolved_dependencies / max(1, self.total_dependencies)

    @property
    def state_id(self) -> str:
        return deterministic_id(
            self.step,
            self.total_steps,
            sorted(self.state_payload.items()),
            self.remaining_distance,
            self.invalid_commitments,
            self.answer_known,
            self.verification_remaining,
        )


@dataclass
class ControlledProblem:
    problem_id: str
    family: str
    prompt: str
    states: list[ComputationalState]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        for index, state in enumerate(self.states):
            value["states"][index]["state_id"] = state.state_id
            value["states"][index]["structural_progress"] = state.structural_progress
            value["states"][index]["dependency_progress"] = state.dependency_progress
        return value


def _entity_names(rng: np.random.Generator, count: int) -> list[str]:
    syllables = ("ka", "zu", "mi", "tor", "vek", "pal", "rin", "sho", "da", "fen")
    names: list[str] = []
    while len(names) < count:
        name = "".join(rng.choice(syllables, size=2)).capitalize()
        if name not in names:
            names.append(name)
    return names


def generate_iterative_problem(index: int, rng: np.random.Generator) -> ControlledProblem:
    modulus = int(rng.integers(7, 20))
    multiplier = int(rng.integers(2, modulus))
    increment = int(rng.integers(1, modulus))
    start = int(rng.integers(0, modulus))
    steps = int(rng.integers(4, 8))
    values = [start]
    for _ in range(steps):
        values.append((multiplier * values[-1] + increment) % modulus)
    states = []
    for step, value in enumerate(values):
        remaining = steps - step
        states.append(
            ComputationalState(
                step=step,
                total_steps=steps,
                state_payload={"register": value},
                valid_next=(
                    f"apply r <- ({multiplier}*{value}+{increment}) mod {modulus}",
                )
                if remaining
                else ("verify final register",),
                remaining_distance=remaining,
                resolved_dependencies=step,
                total_dependencies=steps,
                answer_known=remaining == 0,
                verification_remaining=True,
            )
        )
    prompt = (
        f"Start with register r={start}. Apply r <- ({multiplier}r+{increment}) "
        f"mod {modulus}, exactly {steps} times. Report and verify the final r."
    )
    return ControlledProblem(
        problem_id=f"iterative-{index:05d}",
        family="iterative_state_machine",
        prompt=prompt,
        states=states,
        answer=str(values[-1]),
        metadata={
            "modulus": modulus,
            "multiplier": multiplier,
            "increment": increment,
            "values": values,
        },
    )


def generate_ontology_problem(index: int, rng: np.random.Generator) -> ControlledProblem:
    steps = int(rng.integers(3, 7))
    entities = _entity_names(rng, steps + 3)
    relation = str(rng.choice(["orbits", "mentors", "signals", "guards"]))
    edges = [(entities[i], entities[i + 1]) for i in range(steps)]
    distractors = [(entities[-1], entities[0]), (entities[-2], entities[1])]
    facts = [f"{a} {relation} {b}" for a, b in edges]
    facts += [f"{a} does not {relation} {b}" for a, b in distractors]
    states = []
    for step in range(steps + 1):
        remaining = steps - step
        payload = {
            "chain_start": entities[0],
            "current": entities[step],
            "relation": relation,
            "used_edges": step,
        }
        states.append(
            ComputationalState(
                step=step,
                total_steps=steps,
                state_payload=payload,
                valid_next=(
                    f"follow {entities[step]} -> {entities[step + 1]}",
                )
                if remaining
                else ("state the chain endpoint",),
                remaining_distance=remaining,
                resolved_dependencies=step,
                total_dependencies=steps,
                answer_known=remaining == 0,
                verification_remaining=remaining == 0,
            )
        )
    prompt = (
        "In this fictional ontology, follow only the positive relation chain. "
        + "; ".join(facts)
        + f". Starting at {entities[0]}, what endpoint is reached after {steps} links?"
    )
    return ControlledProblem(
        problem_id=f"ontology-{index:05d}",
        family="fictional_ontology",
        prompt=prompt,
        states=states,
        answer=entities[steps],
        metadata={"entities": entities, "relation": relation, "edges": edges},
    )


def generate_dependency_problem(index: int, rng: np.random.Generator) -> ControlledProblem:
    node_count = int(rng.integers(5, 9))
    nodes = [f"D{index}_{i}" for i in range(node_count)]
    parents: dict[str, list[str]] = {node: [] for node in nodes}
    for child_index in range(1, node_count):
        candidates = nodes[:child_index]
        parent_count = int(rng.integers(1, min(3, len(candidates)) + 1))
        parents[nodes[child_index]] = sorted(
            str(value) for value in rng.choice(candidates, size=parent_count, replace=False)
        )
    order = nodes.copy()
    states = []
    for step in range(node_count + 1):
        completed = set(order[:step])
        available = tuple(
            node
            for node in nodes
            if node not in completed and set(parents[node]).issubset(completed)
        )
        remaining = node_count - step
        states.append(
            ComputationalState(
                step=step,
                total_steps=node_count,
                state_payload={
                    "completed": tuple(sorted(completed)),
                    "unresolved": tuple(sorted(set(nodes) - completed)),
                },
                valid_next=available if available else ("emit conclusion",),
                remaining_distance=remaining,
                resolved_dependencies=step,
                total_dependencies=node_count,
                answer_known=remaining == 0,
                verification_remaining=remaining == 0,
            )
        )
    clauses = [
        f"{node} requires {', '.join(parents[node])}" if parents[node] else f"{node} is initially available"
        for node in nodes
    ]
    prompt = (
        "Resolve the fictional dependency graph in a valid order, then report DONE. "
        + "; ".join(clauses)
        + "."
    )
    return ControlledProblem(
        problem_id=f"dependency-{index:05d}",
        family="dependency_graph",
        prompt=prompt,
        states=states,
        answer="DONE",
        metadata={"nodes": nodes, "parents": parents},
    )


def generate_suite(per_family: int, seed: int) -> list[ControlledProblem]:
    rng = np.random.default_rng(seed)
    problems: list[ControlledProblem] = []
    for index in range(per_family):
        problems.append(generate_iterative_problem(index, rng))
        problems.append(generate_ontology_problem(index, rng))
        problems.append(generate_dependency_problem(index, rng))
    return problems


def rollback_state(problem: ControlledProblem, advanced_step: int) -> ComputationalState:
    if advanced_step <= 0 or advanced_step >= len(problem.states):
        raise ValueError("advanced_step must identify a noninitial state")
    base = problem.states[max(0, advanced_step - 1)]
    return ComputationalState(
        step=base.step,
        total_steps=base.total_steps,
        state_payload={**base.state_payload, "rollback_from": advanced_step},
        valid_next=base.valid_next,
        remaining_distance=base.remaining_distance,
        resolved_dependencies=base.resolved_dependencies,
        total_dependencies=base.total_dependencies,
        invalid_commitments=1,
        answer_known=False,
        verification_remaining=True,
    )

