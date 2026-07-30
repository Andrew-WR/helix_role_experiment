from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


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
WORD_PATTERN = re.compile(r"[a-z0-9]+")
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|<>]+\|>")
BOX_START_PATTERN = re.compile(r"\\(?:boxed|fbox)\{")
ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "fig.",
    "eq.",
    "no.",
}


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


def normalize_answer(value: str) -> str:
    value = SPECIAL_TOKEN_PATTERN.sub("", value)
    return " ".join(WORD_PATTERN.findall(value.casefold()))


def _terminal_boxed_answer(text: str) -> tuple[str | None, int | None]:
    lowered = text.casefold()
    thinking_open = lowered.rfind("<think>")
    thinking_close = lowered.rfind("</think>")
    if thinking_open >= 0 and thinking_close < thinking_open:
        return None, None
    answer_region_start = (
        thinking_close + len("</think>") if thinking_close >= 0 else 0
    )
    candidates: list[tuple[str, int]] = []
    for match in BOX_START_PATTERN.finditer(text, answer_region_start):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            continue
        suffix = SPECIAL_TOKEN_PATTERN.sub("", text[index:])
        suffix = re.sub(r"</?s>", "", suffix, flags=re.IGNORECASE)
        suffix = suffix.replace(r"\)", "").replace(r"\]", "")
        if suffix.strip(" \t\r\n$.,;:!?*_`"):
            continue
        line_start = text.rfind("\n", answer_region_start, match.start()) + 1
        candidates.append((text[match.end() : index - 1].strip(), line_start))
    return candidates[-1] if candidates else (None, None)


def extract_final_answer(text: str) -> tuple[str | None, int | None]:
    matches = list(FINAL_PATTERN.finditer(text))
    lowered = text.casefold()
    thinking_open = lowered.rfind("<think>")
    thinking_close = lowered.rfind("</think>")
    if thinking_open >= 0:
        if thinking_close < thinking_open:
            matches = []
        else:
            matches = [
                match for match in matches if match.start() >= thinking_close
            ]
    if matches:
        match = matches[-1]
        return match.group(1).strip(), match.start()
    return _terminal_boxed_answer(text)


def final_answer_is_correct(text: str, expected: str) -> bool:
    answer, _ = extract_final_answer(text)
    if answer is None:
        return False
    observed = normalize_answer(answer)
    target = normalize_answer(expected)
    if not observed or not target:
        return False
    for prefix in ("the answer is ", "answer is ", "boxed "):
        if observed.startswith(prefix):
            observed = observed[len(prefix) :]
            break
    return observed == target


def _period_is_terminal(text: str, index: int, start: int) -> bool:
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    if text[index : index + 3] == "..." or text[max(0, index - 2) : index + 1] == "...":
        return False
    prefix = text[start : index + 1].rstrip().casefold()
    if re.fullmatch(r"\s*(?:\d+|[a-z])\.", prefix):
        return False
    last = prefix.split()[-1] if prefix.split() else ""
    if last in ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[a-z]\.){1,4}", last):
        return False
    return not following or following.isspace() or following == "<"


def split_sentence_spans(text: str) -> list[SentenceSpan]:
    """Split reasoning text without breaking decimals, LaTeX, or code."""

    spans: list[SentenceSpan] = []
    start = 0
    index = 0
    dollar_mode = 0
    latex_closer: str | None = None
    in_inline_code = False
    in_fenced_code = False
    environment_depth = 0

    def append_span(end: int) -> None:
        nonlocal start
        left = start
        right = end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            spans.append(SentenceSpan(text[left:right], left, right))
        start = end

    while index < len(text):
        if text.startswith("```", index):
            in_fenced_code = not in_fenced_code
            index += 3
            continue
        character = text[index]
        if in_fenced_code:
            index += 1
            continue
        if character == "`":
            in_inline_code = not in_inline_code
            index += 1
            continue
        if in_inline_code:
            index += 1
            continue
        if text.startswith("\\begin{", index):
            environment_depth += 1
        elif text.startswith("\\end{", index) and environment_depth:
            environment_depth -= 1
        if latex_closer is not None:
            if text.startswith(latex_closer, index):
                index += len(latex_closer)
                latex_closer = None
            else:
                index += 1
            continue
        if text.startswith("\\(", index):
            latex_closer = "\\)"
            index += 2
            continue
        if text.startswith("\\[", index):
            latex_closer = "\\]"
            index += 2
            continue
        if character == "$" and (index == 0 or text[index - 1] != "\\"):
            marker_length = 2 if text.startswith("$$", index) else 1
            dollar_mode = 0 if dollar_mode == marker_length else marker_length
            index += marker_length
            continue
        protected = bool(dollar_mode or environment_depth)
        terminal = False
        if not protected and character in "!?;":
            terminal = True
        elif not protected and character == ".":
            terminal = _period_is_terminal(text, index, start)
        if terminal:
            append_span(index + 1)
        elif character == "\n":
            current = text[start:index].strip().casefold()
            if current.startswith("final:") or "\\boxed" in current:
                append_span(index)
        index += 1
    append_span(len(text))
    return spans


def split_sentences(text: str) -> list[str]:
    return [span.text for span in split_sentence_spans(text)]


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
