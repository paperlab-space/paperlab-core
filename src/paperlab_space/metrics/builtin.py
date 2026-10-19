"""Built-in reference-based metrics: exact match, accuracy, classification F1, token F1."""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np

from paperlab_space.metrics.base import (
    Indices,
    Metric,
    Predictions,
    References,
    Scorer,
    register,
)
from paperlab_space.metrics.normalize import Normalizer, extract_label, tokenize
from paperlab_space.results import MetricResult

Average = Literal["macro", "micro", "weighted"]


class _Normalizing(Metric):
    averages_rows = True

    def __init__(
        self,
        ignore_case: bool = True,
        trim_punctuation: bool = True,
        ignore_punctuation: bool = False,
        nfkc: bool = True,
        count_failed: bool = True,
    ) -> None:
        self.normalize = Normalizer(
            nfkc=nfkc,
            ignore_case=ignore_case,
            trim_punctuation=trim_punctuation,
            ignore_punctuation=ignore_punctuation,
        )
        self.count_failed = count_failed
        """Score failed requests as wrong (``True``) or leave them out (``False``)."""

    def _per_row(
        self, predictions: Predictions, references: References
    ) -> tuple[list[float | None], int]:
        scores: list[float | None] = []
        failed = 0
        for prediction, refs in zip(predictions, references, strict=True):
            if not refs:
                scores.append(None)
            elif prediction is None:
                failed += 1
                scores.append(0.0 if self.count_failed else None)
            else:
                scores.append(self.score_row(prediction, refs))
        return scores, failed

    def score_row(self, prediction: str, references: list[str]) -> float:
        raise NotImplementedError

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        scores, failed = self._per_row(predictions, references)
        return self.result(
            _mean(scores), per_row=scores, n_scored=_count(scores), extra={"failed": float(failed)}
        )


@register
class ExactMatch(_Normalizing):
    name = "exact_match"
    description = "Share of rows where the output equals a reference answer after normalisation."

    def score_row(self, prediction: str, references: list[str]) -> float:
        answer = self.normalize(prediction)
        return float(any(answer == self.normalize(ref) for ref in references))


@register
class Accuracy(_Normalizing):
    name = "accuracy"
    description = (
        "Classification accuracy. With `labels`, a free-form answer that mentions "
        "exactly one label counts as that label."
    )

    def __init__(
        self,
        labels: list[str] | None = None,
        ignore_case: bool = True,
        trim_punctuation: bool = True,
        ignore_punctuation: bool = False,
        nfkc: bool = True,
        count_failed: bool = True,
    ) -> None:
        super().__init__(ignore_case, trim_punctuation, ignore_punctuation, nfkc, count_failed)
        self.labels = labels

    def score_row(self, prediction: str, references: list[str]) -> float:
        answer = (
            extract_label(prediction, self.labels, self.normalize)
            if self.labels
            else self.normalize(prediction)
        )
        return float(any(answer == self.normalize(ref) for ref in references))


@register
class F1(Accuracy):
    name = "f1"
    description = (
        "Classification precision, recall, and F1 over the gold labels (macro-averaged by default)."
    )
    averages_rows = False

    def __init__(
        self,
        average: Average = "macro",
        labels: list[str] | None = None,
        ignore_case: bool = True,
        trim_punctuation: bool = True,
        ignore_punctuation: bool = False,
        nfkc: bool = True,
        count_failed: bool = True,
    ) -> None:
        if average not in ("macro", "micro", "weighted"):
            raise ValueError("average must be 'macro', 'micro', or 'weighted'")
        super().__init__(
            labels, ignore_case, trim_punctuation, ignore_punctuation, nfkc, count_failed
        )
        self.average = average

    def _pairs(
        self, predictions: Predictions, references: References
    ) -> tuple[list[str | None], list[str | None], int]:
        """Gold and predicted label per row (``None`` where the row is not scored)."""
        gold: list[str | None] = []
        predicted: list[str | None] = []
        failed = 0
        for prediction, refs in zip(predictions, references, strict=True):
            if not refs:
                gold.append(None)
                predicted.append(None)
                continue
            if prediction is None:
                failed += 1
                if not self.count_failed:
                    gold.append(None)
                    predicted.append(None)
                    continue
                answer = ""
            elif self.labels:
                answer = extract_label(prediction, self.labels, self.normalize)
            else:
                answer = self.normalize(prediction)
            normalized_refs = [self.normalize(r) for r in refs]
            # With several valid answers, a prediction matching any of them is correct.
            gold.append(answer if answer in normalized_refs else normalized_refs[0])
            predicted.append(answer)
        return gold, predicted, failed

    def scorer(self, predictions: Predictions, references: References) -> Scorer:
        gold_all, predicted_all, _ = self._pairs(predictions, references)
        fixed = [self.normalize(label) for label in self.labels] if self.labels else None
        names = sorted({*(fixed or []), *(g for g in gold_all if g is not None)})
        index = {name: i for i, name in enumerate(names)}
        other = len(names)  # predictions that are no class

        def code(label: str | None) -> int:
            return -1 if label is None else index.get(label, other)

        gold = np.array([code(g) for g in gold_all])
        predicted = np.array([code(p) for p in predicted_all])
        allowed = np.zeros(other + 1, dtype=bool)
        if fixed is not None:
            allowed[[index[c] for c in fixed]] = True

        def score(indices: Indices) -> float | None:
            g, p = gold[indices], predicted[indices]
            scored = g >= 0
            g, p = g[scored], p[scored]
            if not len(g):
                return None
            size = other + 1
            support = np.bincount(g, minlength=size)
            tp = np.bincount(g[g == p], minlength=size)
            pred_count = np.bincount(p, minlength=size)
            classes = allowed.copy() if fixed is not None else support > 0
            classes[other] = False
            tp, support, pred_count = tp[classes], support[classes], pred_count[classes]
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.where(pred_count > 0, tp / pred_count, 0.0)
                recall = np.where(support > 0, tp / support, 0.0)
                f1 = np.where(
                    precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
                )
            if self.average == "macro":
                return float(f1.mean()) if len(f1) else 0.0
            n_gold = support.sum()
            if self.average == "weighted":
                return float((f1 * support).sum() / n_gold) if n_gold else 0.0
            fp_total = (pred_count - tp).sum()
            micro_p = tp.sum() / (tp.sum() + fp_total) if tp.sum() + fp_total else 0.0
            micro_r = tp.sum() / n_gold if n_gold else 0.0
            return _f1(float(micro_p), float(micro_r))

        return score

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        pairs = self._pairs(predictions, references)
        gold = [g for g in pairs[0] if g is not None]
        predicted = [p for p in pairs[1] if p is not None]
        failed = pairs[2]

        if not gold:
            return self.result(None, n_scored=0, extra={"failed": float(failed)})
        classes = (
            [self.normalize(label) for label in self.labels] if self.labels else sorted(set(gold))
        )
        support = Counter(gold)
        per_class: dict[str, tuple[float, float, float]] = {}
        tp_total = fp_total = 0
        for cls in classes:
            tp = sum(g == p == cls for g, p in zip(gold, predicted, strict=True))
            fp = sum(p == cls and g != cls for g, p in zip(gold, predicted, strict=True))
            fn = support[cls] - tp
            tp_total += tp
            fp_total += fp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            per_class[cls] = (precision, recall, _f1(precision, recall))

        n_gold = sum(support[c] for c in classes)
        micro_p = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
        micro_r = tp_total / n_gold if n_gold else 0.0
        averages = {
            "macro": tuple(sum(v[i] for v in per_class.values()) / len(classes) for i in range(3)),
            "micro": (micro_p, micro_r, _f1(micro_p, micro_r)),
            "weighted": tuple(
                sum(per_class[c][i] * support[c] for c in classes) / n_gold if n_gold else 0.0
                for i in range(3)
            ),
        }
        precision, recall, f1 = averages[self.average]
        extra = {
            "precision": precision,
            "recall": recall,
            **{f"f1_{name}": value[2] for name, value in averages.items()},
            **{f"f1[{cls}]": scores[2] for cls, scores in per_class.items()},
            "failed": float(failed),
        }
        return self.result(f1, n_scored=len(gold), extra=extra)


@register
class TokenF1(_Normalizing):
    name = "token_f1"
    description = (
        "Token-overlap F1 between output and reference (SQuAD-style), averaged over "
        "rows. Characters count as tokens in Chinese, Japanese, Thai, and Burmese."
    )

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        remove_articles: bool = False,
        nfkc: bool = True,
        count_failed: bool = True,
    ) -> None:
        super().__init__(ignore_case, True, ignore_punctuation, nfkc, count_failed)
        self.normalize = Normalizer(
            nfkc=nfkc,
            ignore_case=ignore_case,
            trim_punctuation=True,
            ignore_punctuation=ignore_punctuation,
            remove_articles=remove_articles,
        )

    def score_row(self, prediction: str, references: list[str]) -> float:
        pred_tokens = tokenize(self.normalize(prediction))
        return max(_token_f1(pred_tokens, tokenize(self.normalize(ref))) for ref in references)


def _token_f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    return _f1(common / len(pred), common / len(ref))


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(scores: list[float | None]) -> float | None:
    values = [s for s in scores if s is not None]
    return sum(values) / len(values) if values else None


def _count(scores: list[float | None]) -> int:
    return sum(s is not None for s in scores)
