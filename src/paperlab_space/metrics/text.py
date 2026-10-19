"""Text-generation metrics: BLEU and chrF (through sacreBLEU), and ROUGE."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Literal

import numpy as np

from paperlab_space.metrics.base import (
    Indices,
    Metric,
    Predictions,
    References,
    Scorer,
    register,
)
from paperlab_space.metrics.normalize import _NO_SPACE_SCRIPTS, Normalizer, tokenize
from paperlab_space.results import MetricResult

BleuTokenizer = Literal[
    "auto", "13a", "intl", "zh", "char", "none", "ja-mecab", "ko-mecab", "flores200"
]


class _SacreBLEU(Metric):
    """Shared code for metrics computed with sacreBLEU's corpus statistics."""

    def __init__(self, count_failed: bool = True) -> None:
        self.count_failed = count_failed
        """Score failed requests as empty outputs (``True``) or leave them out (``False``)."""

    def _sacrebleu(self, references: list[str], sentence: bool = False) -> Any:
        """The sacreBLEU metric object; ``sentence`` for per-row scores."""
        raise NotImplementedError

    def _prepare(self, texts: list[str]) -> list[str]:
        return texts

    def _rows(
        self, predictions: Predictions, references: References
    ) -> tuple[list[int], list[str], list[list[str]], int]:
        """Rows that are scored, with their hypothesis and references."""
        keep, hypotheses, refs, failed = [], [], [], 0
        for i, (prediction, row_refs) in enumerate(zip(predictions, references, strict=True)):
            if not row_refs:
                continue
            if prediction is None:
                failed += 1
                if not self.count_failed:
                    continue
            keep.append(i)
            hypotheses.append(prediction or "")
            refs.append(row_refs)
        return keep, hypotheses, refs, failed

    def _streams(self, refs: list[list[str]]) -> list[list[str | None]]:
        # sacreBLEU takes one stream per reference position; rows with fewer references
        # are padded with None.
        width = max((len(r) for r in refs), default=0)
        return [[r[k] if k < len(r) else None for r in refs] for k in range(width)]

    def _setup(
        self, predictions: Predictions, references: References
    ) -> tuple[Any, list[int], list[str], list[list[str | None]], list[list[str]], int]:
        keep, hypotheses, refs, failed = self._rows(predictions, references)
        # Settings depend on the references only, so every model is scored the same way.
        metric = self._sacrebleu([r for row in refs for r in row])
        hypotheses = self._prepare(hypotheses)
        refs = [self._prepare(row) for row in refs]
        return metric, keep, hypotheses, self._streams(refs), refs, failed

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        metric, keep, hypotheses, streams, refs, failed = self._setup(predictions, references)
        extra: dict[str, float] = {"failed": float(failed)}
        if not keep:
            return self.result(None, n_scored=0, extra=extra)
        corpus = metric.corpus_score(hypotheses, streams)
        sentence = self._sacrebleu([r for row in refs for r in row], sentence=True)
        per_row: list[float | None] = [None] * len(predictions)
        for i, hypothesis, row_refs in zip(keep, hypotheses, refs, strict=True):
            per_row[i] = _unit(sentence.sentence_score(hypothesis, row_refs).score)
        extra.update(self._extra(corpus))
        return self.result(
            _unit(corpus.score),
            per_row=per_row,
            n_scored=len(keep),
            extra=extra,
            # Read after scoring: the signature records the number of references.
            info={"signature": str(metric.get_signature())},
        )

    def _extra(self, corpus: Any) -> dict[str, float]:
        return {}

    def scorer(self, predictions: Predictions, references: References) -> Scorer:
        metric, keep, hypotheses, streams, _, _ = self._setup(predictions, references)
        stats = np.zeros((len(predictions), 0))
        if keep:
            row_stats = np.array(metric._extract_corpus_statistics(hypotheses, streams))
            stats = np.zeros((len(predictions), row_stats.shape[1]))
            stats[keep] = row_stats
        scored = np.zeros(len(predictions), dtype=bool)
        scored[keep] = True

        def score(indices: Indices) -> float | None:
            if not scored[indices].any():
                return None
            total = stats[indices].sum(axis=0)
            return _unit(float(metric._compute_score_from_stats(list(total)).score))

        return score


@register
class BLEU(_SacreBLEU):
    name = "bleu"
    description = (
        "Corpus BLEU from sacreBLEU, on a 0-1 scale. Text in scripts without spaces "
        "(Chinese, Japanese, Thai, Burmese, ...) is split into characters by default."
    )

    def __init__(
        self,
        tokenize: BleuTokenizer = "auto",
        lowercase: bool = False,
        smooth_method: Literal["exp", "floor", "add-k", "none"] = "exp",
        count_failed: bool = True,
    ) -> None:
        super().__init__(count_failed)
        self.tokenize = tokenize
        """sacreBLEU tokenizer. ``auto`` uses ``13a``, or splits scripts without spaces into
        characters (as ``token_f1`` does) when the references contain them."""
        self.lowercase = lowercase
        """Compare case-insensitively."""
        self.smooth_method = smooth_method
        """Smoothing for n-grams with no match; sacreBLEU's default is ``exp``."""
        self._pretokenize = False

    def _sacrebleu(self, references: list[str], sentence: bool = False) -> Any:
        from sacrebleu.metrics.bleu import BLEU as SacreBLEU

        tokenizer: str = self.tokenize
        self._pretokenize = False
        if tokenizer == "auto":
            self._pretokenize = any(_has_no_space_script(t) for t in references)
            tokenizer = "none" if self._pretokenize else "13a"
        return SacreBLEU(
            tokenize=tokenizer,
            lowercase=self.lowercase,
            smooth_method=self.smooth_method,
            # sacreBLEU recommends effective order for sentence BLEU, but not for corpus BLEU.
            effective_order=sentence,
        )

    def _prepare(self, texts: list[str]) -> list[str]:
        if not self._pretokenize:
            return texts
        return [" ".join(tokenize(unicodedata.normalize("NFKC", t))) for t in texts]

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        result = super().compute(predictions, references)
        if self._pretokenize:
            result.info["tokenize"] = "paperlab-space script-aware tokenizer, then sacreBLEU none"
        return result

    def _extra(self, corpus: Any) -> dict[str, float]:
        extra = {f"precision_{n}": p / 100 for n, p in enumerate(corpus.precisions, start=1)}
        extra["brevity_penalty"] = corpus.bp
        return extra


@register
class ChrF(_SacreBLEU):
    name = "chrf"
    description = (
        "Character n-gram F-score (chrF) from sacreBLEU, on a 0-1 scale. Set `word_order: 2` "
        "for chrF++. Works for every script without tokenisation."
    )

    def __init__(
        self,
        char_order: int = 6,
        word_order: int = 0,
        beta: int = 2,
        lowercase: bool = False,
        count_failed: bool = True,
    ) -> None:
        if char_order < 1 or word_order < 0 or beta < 1:
            raise ValueError("char_order and beta must be at least 1, word_order at least 0")
        super().__init__(count_failed)
        self.char_order = char_order
        """Longest character n-gram."""
        self.word_order = word_order
        """Longest word n-gram; 2 gives chrF++."""
        self.beta = beta
        """Weight of recall relative to precision."""
        self.lowercase = lowercase
        """Compare case-insensitively."""

    def _sacrebleu(self, references: list[str], sentence: bool = False) -> Any:
        from sacrebleu.metrics.chrf import CHRF

        return CHRF(
            char_order=self.char_order,
            word_order=self.word_order,
            beta=self.beta,
            lowercase=self.lowercase,
        )


def _unit(score: float) -> float:
    """A sacreBLEU score (0-100) on a 0-1 scale, without float overshoot past 1."""
    return min(max(score / 100, 0.0), 1.0)


def _has_no_space_script(text: str) -> bool:
    return any(
        not ch.isascii() and unicodedata.name(ch, "").startswith(_NO_SPACE_SCRIPTS) for ch in text
    )


class _Rouge(Metric):
    averages_rows = True
    n: int = 0
    """n-gram size; 0 means longest common subsequence (ROUGE-L)."""

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        nfkc: bool = True,
        count_failed: bool = True,
    ) -> None:
        self.normalize = Normalizer(
            nfkc=nfkc,
            ignore_case=ignore_case,
            trim_punctuation=True,
            ignore_punctuation=ignore_punctuation,
        )
        self.count_failed = count_failed
        """Score failed requests as 0 (``True``) or leave them out (``False``)."""

    def _tokens(self, text: str) -> list[str]:
        return tokenize(self.normalize(text))

    def _prf(self, prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
        if self.n:
            pred = _ngrams(prediction, self.n)
            ref = _ngrams(reference, self.n)
            overlap = sum((pred & ref).values())
            n_pred, n_ref = sum(pred.values()), sum(ref.values())
        else:
            overlap = _lcs(prediction, reference)
            n_pred, n_ref = len(prediction), len(reference)
        precision = overlap / n_pred if n_pred else 0.0
        recall = overlap / n_ref if n_ref else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        per_row: list[float | None] = []
        precisions: list[float] = []
        recalls: list[float] = []
        failed = 0
        for prediction, refs in zip(predictions, references, strict=True):
            if not refs:
                per_row.append(None)
                continue
            if prediction is None:
                failed += 1
                if self.count_failed:
                    per_row.append(0.0)
                    precisions.append(0.0)
                    recalls.append(0.0)
                else:
                    per_row.append(None)
                continue
            tokens = self._tokens(prediction)
            # With several references, the best-matching one counts (as in rouge-score).
            p, r, f = max(
                (self._prf(tokens, self._tokens(ref)) for ref in refs), key=lambda x: x[2]
            )
            per_row.append(f)
            precisions.append(p)
            recalls.append(r)
        values = [v for v in per_row if v is not None]
        extra = {"failed": float(failed)}
        if values:
            extra["precision"] = sum(precisions) / len(precisions)
            extra["recall"] = sum(recalls) / len(recalls)
        return self.result(
            sum(values) / len(values) if values else None,
            per_row=per_row,
            n_scored=len(values),
            extra=extra,
        )


@register
class Rouge1(_Rouge):
    name = "rouge1"
    description = "ROUGE-1: unigram-overlap F-measure with the reference, averaged over rows."
    n = 1


@register
class Rouge2(_Rouge):
    name = "rouge2"
    description = "ROUGE-2: bigram-overlap F-measure with the reference, averaged over rows."
    n = 2


@register
class RougeL(_Rouge):
    name = "rougeL"
    description = (
        "ROUGE-L: longest-common-subsequence F-measure with the reference, averaged over rows."
    )
    n = 0


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]
