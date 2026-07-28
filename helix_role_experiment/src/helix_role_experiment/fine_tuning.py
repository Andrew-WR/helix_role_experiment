from __future__ import annotations

import re
from collections import Counter


CATEGORY_PATTERNS = {
    "setup": (r"\bgiven\b", r"\bwe need\b", r"\blet\b"),
    "planning": (r"\bplan\b", r"\bfirst\b", r"\bapproach\b"),
    "productive_computation": (r"[=+\-*/]", r"\bcalculate\b", r"\btherefore\b"),
    "irrelevant_exploration": (r"\balternatively\b", r"\btangent\b", r"\bunrelated\b"),
    "repeated_computation": (r"\bagain\b", r"\brecompute\b", r"\brepeat\b"),
    "backtracking": (r"\bbacktrack\b", r"\bwrong branch\b", r"\breturn to\b"),
    "correction": (r"\bcorrection\b", r"\bmistake\b", r"\binvalid\b"),
    "verification": (r"\bverify\b", r"\bcheck\b", r"\bconfirm\b"),
    "consolidation": (r"\bsummar", r"\bcombine\b", r"\bconclude\b"),
    "rhetorical_restatement": (r"\bin other words\b", r"\brestat", r"\bclearly\b"),
    "final_answer_detail": (r"\bfinal answer\b", r"\banswer is\b", r"\bboxed\b"),
}


def sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def classify_sentence(sentence: str) -> str:
    lowered = sentence.lower()
    scores = {
        category: sum(bool(re.search(pattern, lowered)) for pattern in patterns)
        for category, patterns in CATEGORY_PATTERNS.items()
    }
    winner = max(scores, key=scores.get)
    return winner if scores[winner] > 0 else "productive_computation"


def category_durations(text: str) -> dict[str, int]:
    counts = Counter(classify_sentence(sentence) for sentence in sentence_split(text))
    return dict(counts)


def removed_content_categories(base_text: str, tuned_text: str) -> dict[str, int]:
    """Approximate removed sentence categories using normalized exact matching.

    Confirmatory use should replace this deterministic baseline with blinded
    human or verifier labels; the method remains useful as an auditable floor.
    """

    normalize = lambda value: re.sub(r"\s+", " ", value.lower()).strip()
    tuned = {normalize(sentence) for sentence in sentence_split(tuned_text)}
    removed = [
        sentence for sentence in sentence_split(base_text) if normalize(sentence) not in tuned
    ]
    return dict(Counter(classify_sentence(sentence) for sentence in removed))

