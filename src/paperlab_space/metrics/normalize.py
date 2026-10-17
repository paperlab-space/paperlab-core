"""Text normalisation and tokenisation shared by metrics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Normalizer:
    """How to normalise text before comparing a prediction to a reference.

    Defaults suit short answers and labels: NFKC (so full-width ``ＡＢＣ`` equals ``ABC``),
    case-insensitive, whitespace collapsed, and quotes/punctuation trimmed from the ends
    (so ``"Positive."`` matches ``positive``).
    """

    nfkc: bool = True
    ignore_case: bool = True
    trim_punctuation: bool = True
    """Strip punctuation and quotes from the start and end only."""
    ignore_punctuation: bool = False
    """Remove all punctuation, including inside the text."""
    remove_articles: bool = False
    """Remove English articles (a, an, the), as in SQuAD-style evaluation."""

    def __call__(self, text: str) -> str:
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.ignore_case:
            text = text.casefold()
        if self.ignore_punctuation:
            text = "".join(" " if _is_punct(ch) else ch for ch in text)
        if self.remove_articles:
            text = _ARTICLES.sub(" ", text)
        text = " ".join(text.split())
        if self.trim_punctuation:
            start, end = 0, len(text)
            while start < end and (_is_punct(text[start]) or text[start].isspace()):
                start += 1
            while end > start and (_is_punct(text[end - 1]) or text[end - 1].isspace()):
                end -= 1
            text = text[start:end]
        return text


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith(("P", "S")) and ch not in "+-%$#@&"


_NO_SPACE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN")


def tokenize(text: str) -> list[str]:
    """Split on whitespace, and into single characters for scripts written without spaces
    between words (Chinese, Japanese, Thai, Lao, Khmer, Burmese, Tibetan).

    Korean uses spaces between words, so Hangul is split on whitespace like Latin text.
    """
    tokens: list[str] = []
    for word in text.split():
        buffer = ""
        for ch in word:
            if not ch.isascii() and unicodedata.name(ch, "").startswith(_NO_SPACE_SCRIPTS):
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                if unicodedata.category(ch).startswith("M") and tokens:
                    tokens[-1] += ch  # keep combining marks (e.g. Thai vowels) with their base
                else:
                    tokens.append(ch)
            else:
                buffer += ch
        if buffer:
            tokens.append(buffer)
    return tokens


def extract_label(text: str, labels: Sequence[str], normalize: Normalizer) -> str:
    """Map a free-form answer to one of ``labels``.

    Returns the matching label if the normalised answer equals one, or if exactly one label
    appears in it (``"The sentiment is positive."`` -> ``positive``). Otherwise the
    normalised answer is returned unchanged and will not match any label.
    """
    answer = normalize(text)
    normalized = {normalize(label): label for label in labels}
    if answer in normalized:
        return normalize(normalized[answer])
    found = [n for n in normalized if n and _contains(answer, n)]
    # Prefer the longest match so "very positive" wins over "positive".
    found = [f for f in found if not any(f != g and f in g for g in found)]
    return found[0] if len(found) == 1 else answer


def _contains(text: str, label: str) -> bool:
    if label.isascii():
        return re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text) is not None
    return label in text
