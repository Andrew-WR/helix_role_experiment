from __future__ import annotations

import re
from collections import Counter


FINAL_PATTERN = re.compile(
    r"(?im)^\s*\**final(?:\s+answer)?\**\s*:\**\s*(.+?)\s*$"
)
ANCHOR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bplan\b|\bapproach\b|\bfirst\b",
        r"\bwait\b|\breconsider\b|\bmistake\b|\bwrong\b|\bbacktrack\b",
        r"\bcheck\b|\bverify\b|\bconfirm\b|\bmake sure\b",
        r"\bmaybe\b|\buncertain\b|\binstead\b|\btry again\b",
    )
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_answer(value: str) -> str:
    return " ".join(WORD_PATTERN.findall(value.casefold()))


def extract_final_answer(text: str) -> tuple[str | None, int | None]:
    matches = list(FINAL_PATTERN.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    return match.group(1).strip(), match.start()


def final_answer_is_correct(text: str, expected: str) -> bool:
    answer, _ = extract_final_answer(text)
    if answer is None:
        return False
    observed = normalize_answer(answer)
    target = normalize_answer(expected)
    if not observed or not target:
        return False
    for prefix in ("the answer is ", "answer is "):
        if observed.startswith(prefix):
            observed = observed[len(prefix) :]
            break
    return observed == target


def split_sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in SENTENCE_PATTERN.split(text.strip())
        if value.strip()
    ]


def anchor_sentence_fraction(text: str) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    return sum(
        any(pattern.search(sentence) for pattern in ANCHOR_PATTERNS)
        for sentence in sentences
    ) / len(sentences)


def repeated_sentence_fraction(text: str) -> float:
    sentences = [normalize_answer(value) for value in split_sentences(text)]
    sentences = [value for value in sentences if value]
    if not sentences:
        return 0.0
    counts = Counter(sentences)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(sentences)


def repeated_ngram_fraction(token_ids: list[int], n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    total = len(token_ids) - n + 1
    for index in range(total):
        value = tuple(token_ids[index : index + n])
        if value in seen:
            repeated += 1
        seen.add(value)
    return repeated / total
