from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from .behavioral import SentenceSpan, split_sentence_spans


CATEGORY_WORDS = {
    "final_answer_emission": [
        "\\boxed",
        "\\fbox",
        "final answer",
        "answer is",
        "the answer is",
        "therefore the answer",
        "thus the answer",
        "hence the answer",
        "so the answer",
        "we conclude",
        "final:",
        "answer:",
        "boxed answer",
        "final result",
    ],
    "problem_setup": [
        "need to",
        "we need",
        "we are given",
        "given that",
        "the problem asks",
        "we want to",
        "we must find",
        "the goal is",
        "suppose",
        "assume",
        "let",
        "define",
        "denote",
        "set",
        "constraint",
        "condition",
        "rewrite",
        "represent",
        "consider the equation",
    ],
    "fact_retrieval": [
        "remember",
        "recall",
        "formula",
        "identity",
        "theorem",
        "property",
        "rule",
        "definition",
        "by definition",
        "we know",
        "known that",
        "standard result",
        "fact",
        "using the identity",
        "according to",
        "it is known",
    ],
    "active_computation": [
        "calculate",
        "compute",
        "solve",
        "substitute",
        "simplify",
        "expand",
        "factor",
        "evaluate",
        "derive",
        "rearrange",
        "differentiate",
        "integrate",
        "divide",
        "multiply",
        "subtract",
        "add",
        "equate",
        "equals",
        "result",
        "giving",
        "gives",
        "yields",
        "obtains",
        "becomes",
        "=",
    ],
    "uncertainty_management": [
        "wait",
        "hold on",
        "let me",
        "double check",
        "double-check",
        "hmm",
        "maybe",
        "perhaps",
        "possibly",
        "not sure",
        "actually",
        "reconsider",
        "mistake",
        "incorrect",
        "wrong",
        "however",
        "but",
        "alternatively",
        "instead",
        "is that correct",
        "need to check",
        "unless",
    ],
    "result_consolidation": [
        "summarize",
        "in summary",
        "therefore",
        "thus",
        "hence",
        "so",
        "consequently",
        "we get",
        "we have",
        "it follows",
        "this means",
        "combining",
        "putting this together",
        "overall",
        "conclude",
        "conclusion",
        "result is",
    ],
    "self_checking": [
        "verify",
        "check",
        "confirm",
        "validate",
        "cross-check",
        "sanity check",
        "substitute back",
        "recalculate",
        "ensure",
        "consistent",
        "correct",
        "satisfies",
        "matches",
        "indeed",
        "test the result",
        "check the sign",
        "check the arithmetic",
    ],
    "plan_generation": [
        "plan",
        "approach",
        "strategy",
        "method",
        "first",
        "next",
        "then",
        "finally",
        "will",
        "we'll",
        "i'll",
        "try",
        "start by",
        "proceed",
        "consider",
        "break into",
        "split into",
        "case",
        "steps are",
        "to solve this",
    ],
}

WEAK_TERMS = {"=", "so", "will", "given", "let", "set", "result", "correct"}
CATEGORY_PROTOTYPES = {
    "final_answer_emission": (
        "The solver states the final answer and concludes the solution.",
        "The final numeric or symbolic result is emitted.",
    ),
    "problem_setup": (
        "The solver restates variables, assumptions, constraints, or the goal.",
        "The mathematical problem is represented and initialized.",
    ),
    "fact_retrieval": (
        "The solver recalls a formula, theorem, identity, or known fact.",
        "Relevant mathematical knowledge is retrieved.",
    ),
    "active_computation": (
        "The solver performs algebra, arithmetic, substitution, or derivation.",
        "A concrete mathematical operation advances the solution.",
    ),
    "uncertainty_management": (
        "The solver notices uncertainty, reconsiders a step, or changes course.",
        "A possible mistake or alternative is examined.",
    ),
    "result_consolidation": (
        "Intermediate results are combined into a conclusion.",
        "The solver summarizes what follows from completed calculations.",
    ),
    "self_checking": (
        "The solver verifies a calculation or substitutes an answer back.",
        "A derived result is checked for correctness or consistency.",
    ),
    "plan_generation": (
        "The solver proposes an approach or sequences future reasoning steps.",
        "A strategy for solving the problem is formed.",
    ),
}

CORE_STAGE = {
    "problem_setup": 0,
    "plan_generation": 1,
    "fact_retrieval": 1,
    "active_computation": 2,
    "result_consolidation": 3,
    "self_checking": 4,
    "final_answer_emission": 5,
}


def _contains_term(text: str, term: str) -> bool:
    if term in {"=", "\\boxed", "\\fbox", "final:", "answer:"}:
        return term in text
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            text,
        )
    )


def lexical_scores(sentence: str) -> dict[str, float]:
    lowered = sentence.casefold()
    output = {}
    for category, terms in CATEGORY_WORDS.items():
        score = 0.0
        for term in terms:
            normalized = term.casefold()
            if _contains_term(lowered, normalized):
                score += 0.25 if normalized in WEAK_TERMS else 1.0
        output[category] = score
    return output


def load_sentence_embedder(model_id: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic analysis requires sentence-transformers>=2.7. "
            "Install it before running file 06c."
        ) from exc
    return SentenceTransformer(
        model_id,
        device=device,
        tokenizer_kwargs={"padding_side": "left"},
    )


def embed_reasoning_sentences(model, sentences: list[str]) -> np.ndarray:
    task = (
        "Represent this mathematical reasoning sentence for semantic "
        "similarity and reasoning-stage classification"
    )
    inputs = [f"Instruct: {task}\nQuery: {value}" for value in sentences]
    return np.asarray(
        model.encode(
            inputs,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def prototype_embeddings(model) -> tuple[list[str], np.ndarray]:
    categories = []
    texts = []
    for category, prototypes in CATEGORY_PROTOTYPES.items():
        for prototype in prototypes:
            categories.append(category)
            texts.append(prototype)
    values = embed_reasoning_sentences(model, texts)
    centroids = []
    ordered = list(CATEGORY_PROTOTYPES)
    for category in ordered:
        centroid = values[
            [index for index, value in enumerate(categories) if value == category]
        ].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    return ordered, np.asarray(centroids, dtype=np.float32)


def token_span_for_char_span(
    offsets: list[tuple[int, int]],
    span: SentenceSpan,
) -> tuple[int, int]:
    covered = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > span.start and start < span.end
    ]
    if not covered:
        return 0, 0
    return covered[0], covered[-1] + 1


def analyze_condition_sentences(
    model,
    tokenizer,
    condition: str,
    text: str,
    output_tokens: int,
    thresholds: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spans = split_sentence_spans(text)
    if not spans:
        return [], {
            "control": condition,
            "sentence_count": 0,
            "redundant_sentence_count": 0,
            "redundant_sentence_rate": 0.0,
        }
    sentence_embeddings = embed_reasoning_sentences(
        model,
        [span.text for span in spans],
    )
    categories, prototypes = prototype_embeddings(model)
    semantic_scores = sentence_embeddings @ prototypes.T
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    similarity = sentence_embeddings @ sentence_embeddings.T
    rows = []
    previous_core_stage: int | None = None
    forward = 0
    backward = 0
    self_transitions = 0
    stage_revisits = 0
    seen_stages: set[int] = set()
    for index, span in enumerate(spans):
        lexical = lexical_scores(span.text)
        combined = {}
        for category_index, category in enumerate(categories):
            lexical_bonus = min(0.25 * lexical[category], 0.5)
            combined[category] = float(
                semantic_scores[index, category_index] + lexical_bonus
            )
        primary = max(combined, key=combined.get)
        best = combined[primary]
        labels = sorted(
            category
            for category, score in combined.items()
            if lexical[category] > 0 or (score >= 0.35 and score >= best - 0.05)
        )
        if not labels:
            labels = [primary]
        core_candidates = [
            category for category in labels if category in CORE_STAGE
        ]
        core_category = (
            max(core_candidates, key=lambda value: combined[value])
            if core_candidates
            else None
        )
        core_stage = CORE_STAGE.get(core_category) if core_category else None
        if core_stage is not None and previous_core_stage is not None:
            if core_stage > previous_core_stage:
                forward += 1
            elif core_stage < previous_core_stage:
                backward += 1
                if core_stage in seen_stages:
                    stage_revisits += 1
            else:
                self_transitions += 1
        if core_stage is not None:
            previous_core_stage = core_stage
            seen_stages.add(core_stage)
        token_start, token_end = token_span_for_char_span(offsets, span)
        prior_max = (
            float(np.max(similarity[index, :index]))
            if index
            else float("nan")
        )
        row: dict[str, Any] = {
            "control": condition,
            "sentence_index": index,
            "char_start": span.start,
            "char_end": span.end,
            "token_start": token_start,
            "token_end": token_end,
            "normalized_token_start": token_start / max(output_tokens, 1),
            "normalized_token_end": token_end / max(output_tokens, 1),
            "sentence": span.text,
            "primary_category": primary,
            "categories": "|".join(labels),
            "core_stage": core_stage,
            "prior_max_cosine_similarity": prior_max,
        }
        for threshold in thresholds:
            row[f"redundant_at_{threshold:.2f}"] = int(
                index > 0 and prior_max > threshold
            )
        for category in categories:
            row[f"score_{category}"] = combined[category]
            row[f"lexical_{category}"] = lexical[category]
        rows.append(row)

    summary: dict[str, Any] = {
        "control": condition,
        "sentence_count": len(rows),
        "forward_stage_transitions": forward,
        "backward_stage_transitions": backward,
        "self_stage_transitions": self_transitions,
        "stage_revisits": stage_revisits,
    }
    for threshold in thresholds:
        column = f"redundant_at_{threshold:.2f}"
        count = sum(int(row[column]) for row in rows)
        redundant_tokens = sum(
            int(row["token_end"]) - int(row["token_start"])
            for row in rows
            if row[column]
        )
        summary[f"redundant_sentence_count_{threshold:.2f}"] = count
        summary[f"redundant_sentence_rate_{threshold:.2f}"] = (
            count / max(len(rows) - 1, 1)
        )
        summary[f"redundant_token_fraction_{threshold:.2f}"] = (
            redundant_tokens / max(output_tokens, 1)
        )
    category_positions: dict[str, list[float]] = defaultdict(list)
    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for category in str(row["categories"]).split("|"):
            category_counts[category] += 1
            category_positions[category].append(
                float(row["normalized_token_end"])
            )
    for category in categories:
        positions = category_positions[category]
        summary[f"count_{category}"] = category_counts[category]
        summary[f"fraction_{category}"] = (
            category_counts[category] / len(rows)
        )
        summary[f"first_position_{category}"] = (
            min(positions) if positions else float("nan")
        )
        summary[f"last_position_{category}"] = (
            max(positions) if positions else float("nan")
        )
    return rows, summary


def add_baseline_deltas(
    summaries: list[dict[str, Any]],
    baseline_control: str = "baseline",
) -> list[dict[str, Any]]:
    baseline = next(
        value for value in summaries if value["control"] == baseline_control
    )
    output = []
    for value in summaries:
        row = dict(value)
        for metric in (
            "sentence_count",
            "redundant_sentence_count_0.65",
            "redundant_sentence_rate_0.65",
            "redundant_token_fraction_0.65",
            "backward_stage_transitions",
            "stage_revisits",
        ):
            difference = float(value[metric]) - float(baseline[metric])
            row[f"delta_{metric}"] = difference
            denominator = float(baseline[metric])
            row[f"percent_delta_{metric}"] = (
                100.0 * difference / denominator
                if denominator
                else float("nan")
            )
        output.append(row)
    return output
