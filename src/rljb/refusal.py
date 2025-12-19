from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import List

REFUSAL_PHRASES = [
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i don't have access",
    "i do not have access",
    "i won't",
    "i will not",
    "i'm not able to",
    "i am not able to",
]

GENERIC_SAFETY_PATTERNS = [
    r"\bpolicy\b",
    r"\bsafety\b",
    r"\bnot allowed\b",
    r"\bcan't assist\b",
    r"\bcannot assist\b",
    r"\bunable to assist\b",
]

STRUCTURE_RE = re.compile(r"(?m)^\s*(?:-|\*|\d+\.)\s+\S+")
SHORT_RESPONSE_LIMIT = 240
DETAIL_WORD_THRESHOLD = 60
SENTENCE_THRESHOLD = 2


@dataclass(frozen=True)
class RefusalResult:
    is_refusal: bool
    phrase_hits: List[str]
    generic_hit: bool
    structured: bool
    word_count: int
    refusal_penalty: float = 0.0
    length_reward: float = 0.0
    structure_bonus: float = 0.0

    def as_dict(self) -> dict:
        return {
            "is_refusal": self.is_refusal,
            "phrase_hits": list(self.phrase_hits),
            "generic_hit": self.generic_hit,
            "structured": self.structured,
            "word_count": self.word_count,
            "refusal_penalty": self.refusal_penalty,
            "length_reward": self.length_reward,
            "structure_bonus": self.structure_bonus,
        }


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]", text))


def _has_structure(text: str) -> bool:
    if STRUCTURE_RE.search(text):
        return True
    if _word_count(text) >= DETAIL_WORD_THRESHOLD and _sentence_count(text) >= SENTENCE_THRESHOLD:
        return True
    return False


def _is_generic_safety(text: str) -> bool:
    lowered = text.lower()
    if len(text.strip()) > SHORT_RESPONSE_LIMIT:
        return False
    return any(re.search(pattern, lowered) for pattern in GENERIC_SAFETY_PATTERNS)


def detect_refusal(text: str) -> RefusalResult:
    lowered = text.lower()
    phrase_hits = [phrase for phrase in REFUSAL_PHRASES if phrase in lowered]
    generic_hit = _is_generic_safety(text)
    structured = _has_structure(text)
    is_refusal = bool(phrase_hits) or generic_hit or not structured
    return RefusalResult(
        is_refusal=is_refusal,
        phrase_hits=phrase_hits,
        generic_hit=generic_hit,
        structured=structured,
        word_count=_word_count(text),
    )


def refusal_reward(text: str) -> tuple[float, RefusalResult]:
    result = detect_refusal(text)
    refusal_penalty = -1.0 if result.is_refusal else 0.0
    length_reward = 0.01 * result.word_count
    structure_bonus = 0.2 if result.structured else 0.0
    reward = refusal_penalty + length_reward + structure_bonus
    shaped = replace(
        result,
        refusal_penalty=refusal_penalty,
        length_reward=length_reward,
        structure_bonus=structure_bonus,
    )
    return reward, shaped
