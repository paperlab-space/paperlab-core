"""Built-in reference-based metrics: exact match, accuracy, classification F1, token F1."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from paperlab_space.metrics.base import Metric, Predictions, References, register
from paperlab_space.metrics.normalize import Normalizer, extract_label, tokenize
from paperlab_space.results import MetricResult

Average = Literal["macro", "micro", "weighted"]


class _Normalizing(Metric):
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

    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        gold: list[str] = []
        predicted: list[str] = []
        failed = 0
        for prediction, refs in zip(predictions, references, strict=True):
            if not refs:
                continue
            if prediction is None:
                failed += 1
                if not self.count_failed:
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
