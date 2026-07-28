from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import deterministic_id
from .controlled_tasks import ComputationalState, ControlledProblem, rollback_state


@dataclass
class PrefixVariant:
    variant_id: str
    problem_id: str
    family: str
    condition: str
    text: str
    state_id: str
    structural_progress: float
    remaining_distance: int
    operation: str
    termination_allowed: bool
    confidence: float
    token_count_proxy: int
    semantic_parent_state_id: str
    exact_state_valid: bool
    metadata: dict[str, Any]


def _step_sentence(problem: ControlledProblem, state: ComputationalState) -> str:
    if problem.family == "iterative_state_machine":
        return f"After {state.step} updates, the register is {state.state_payload['register']}."
    if problem.family == "fictional_ontology":
        return (
            f"After {state.step} positive links, the chain currently reaches "
            f"{state.state_payload['current']}."
        )
    completed = state.state_payload.get("completed", ())
    return f"Resolved dependencies so far: {', '.join(completed) if completed else 'none'}."


def _history(problem: ControlledProblem, state: ComputationalState) -> str:
    return " ".join(_step_sentence(problem, item) for item in problem.states[1 : state.step + 1])


def _pad_to_words(text: str, target: int) -> str:
    fillers = (
        "This restates the already established state without applying a new transition.",
        "For clarity, no unresolved dependency is changed by this confirmation.",
        "The same intermediate result remains in force.",
    )
    output = text
    cursor = 0
    while len(output.split()) < target:
        output += " " + fillers[cursor % len(fillers)]
        cursor += 1
    return " ".join(output.split()[:target])


def _variant(
    problem: ControlledProblem,
    state: ComputationalState,
    condition: str,
    text: str,
    operation: str,
    termination_allowed: bool,
    confidence: float,
    parent_state_id: str | None = None,
    **metadata: Any,
) -> PrefixVariant:
    return PrefixVariant(
        variant_id=deterministic_id(problem.problem_id, state.state_id, condition, text),
        problem_id=problem.problem_id,
        family=problem.family,
        condition=condition,
        text=text,
        state_id=state.state_id,
        structural_progress=state.structural_progress,
        remaining_distance=state.remaining_distance,
        operation=operation,
        termination_allowed=termination_allowed,
        confidence=confidence,
        token_count_proxy=len(text.split()),
        semantic_parent_state_id=parent_state_id or state.state_id,
        exact_state_valid=True,
        metadata=metadata,
    )


def build_progress_position_cross(problem: ControlledProblem) -> list[PrefixVariant]:
    variants: list[PrefixVariant] = []
    for state in problem.states:
        base = f"{problem.prompt} {_history(problem, state)} {_step_sentence(problem, state)}"
        concise = _variant(
            problem,
            state,
            "concise",
            base,
            "calculation" if state.remaining_distance else "checking",
            False,
            0.55 + 0.35 * state.structural_progress,
        )
        variants.append(concise)
        for condition, extra in (
            ("verbose_paraphrase", "In other words, this is exactly the current intermediate state."),
            ("redundant_valid", "Rechecking the arithmetic confirms the same intermediate state."),
            ("repeated_summary", _step_sentence(problem, state)),
            ("confirmation", "Confirmed: no additional transition has been executed."),
            ("plausible_digression", "A general solution could use a table, but that observation changes no state."),
        ):
            variants.append(
                _variant(
                    problem,
                    state,
                    condition,
                    f"{base} {extra} {extra}",
                    "checking" if condition != "plausible_digression" else "planning",
                    False,
                    concise.confidence,
                    padding_type=condition,
                )
            )

    # Same proxy length, distinct exact state.
    target_words = max(variant.token_count_proxy for variant in variants)
    for state in problem.states:
        text = _pad_to_words(
            f"{problem.prompt} {_history(problem, state)} {_step_sentence(problem, state)}",
            target_words,
        )
        variants.append(
            _variant(
                problem,
                state,
                "length_matched_progress",
                text,
                "calculation" if state.remaining_distance else "consolidation",
                False,
                0.55 + 0.35 * state.structural_progress,
                target_word_count=target_words,
            )
        )

    if len(problem.states) >= 4:
        early = problem.states[1]
        late = problem.states[-2]
        teleport_text = (
            f"{problem.prompt} At this early point a verified lemma is supplied: "
            f"{_step_sentence(problem, late)}"
        )
        variants.append(
            _variant(
                problem,
                late,
                "teleport",
                teleport_text,
                "calculation",
                False,
                0.85,
                parent_state_id=early.state_id,
                supplied_state=late.state_id,
            )
        )
        rolled = rollback_state(problem, len(problem.states) - 2)
        rollback_text = (
            f"{problem.prompt} {_history(problem, late)} A check invalidates the prior "
            f"step. Return to: {_step_sentence(problem, problem.states[rolled.step])}"
        )
        variants.append(
            _variant(
                problem,
                rolled,
                "rollback",
                rollback_text,
                "backtracking",
                False,
                0.25,
                parent_state_id=late.state_id,
                invalidated_state=late.state_id,
            )
        )
        loop_text = (
            f"{problem.prompt} {_history(problem, early)} "
            f"{_step_sentence(problem, early)} {_step_sentence(problem, early)} "
            "No new transition is made."
        )
        variants.append(
            _variant(
                problem,
                early,
                "loop",
                loop_text,
                "calculation",
                False,
                0.60,
                parent_state_id=early.state_id,
            )
        )

    final = problem.states[-1]
    before_final = problem.states[-2]
    variants.extend(
        [
            _variant(
                problem,
                final,
                "answer_known_verification_unfinished",
                f"{problem.prompt} {_history(problem, final)} The answer is {problem.answer}; verify it before responding.",
                "checking",
                False,
                0.95,
            ),
            _variant(
                problem,
                before_final,
                "work_nearly_complete_answer_unknown",
                f"{problem.prompt} {_history(problem, before_final)} The decisive final transition remains.",
                "calculation",
                False,
                0.70,
            ),
            _variant(
                problem,
                problem.states[0],
                "confident_guess",
                f"{problem.prompt} I confidently guess the answer is {problem.answer}, without deriving it.",
                "final_emission",
                True,
                0.95,
            ),
            _variant(
                problem,
                final,
                "complete_answer_forbidden",
                f"{problem.prompt} {_history(problem, final)} The derivation is complete, but do not emit the final answer.",
                "consolidation",
                False,
                0.95,
            ),
            _variant(
                problem,
                final,
                "complete_answer_allowed",
                f"{problem.prompt} {_history(problem, final)} The derivation is complete; emit the final answer.",
                "final_emission",
                True,
                0.95,
            ),
        ]
    )
    return variants


def validate_variant(variant: PrefixVariant, problem: ControlledProblem) -> tuple[bool, str | None]:
    known = {state.state_id: state for state in problem.states}
    if variant.condition == "rollback":
        if variant.metadata.get("invalidated_state") is None:
            return False, "rollback_missing_invalidated_state"
    elif variant.state_id not in known:
        return False, "unknown_state_id"
    if variant.condition in {
        "verbose_paraphrase",
        "redundant_valid",
        "repeated_summary",
        "confirmation",
        "plausible_digression",
        "loop",
    } and variant.state_id != variant.semantic_parent_state_id:
        return False, "same_state_variant_changed_state"
    if not variant.text.strip():
        return False, "empty_prefix"
    return True, None


def build_all_counterfactuals(problems: list[ControlledProblem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in problems:
        for variant in build_progress_position_cross(problem):
            valid, reason = validate_variant(variant, problem)
            variant.exact_state_valid = valid
            row = asdict(variant)
            row["exclusion_reason"] = reason
            rows.append(row)
    return rows

