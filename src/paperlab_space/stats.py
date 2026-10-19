"""Confidence intervals and paired significance tests for scores.

Both resample the data rows, so they work for every metric, including corpus-level ones
such as BLEU and F1 that are not an average of row scores:

- Confidence intervals are percentile bootstrap intervals (Efron and Tibshirani, 1993).
- ``permutation`` is the paired approximate randomisation test (Riezler and Maxwell, 2005):
  each row's two outputs are swapped between the models at random, and the p-value is the
  share of shuffles with a difference at least as large as the observed one.
- ``bootstrap`` is the paired bootstrap test (Koehn, 2004; Berg-Kirkpatrick et al., 2012),
  which also gives an interval for the difference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from paperlab_space.config import RunConfig, StatisticsSettings
from paperlab_space.metrics.base import Indices, Metric, Predictions, References, Scorer
from paperlab_space.results import Comparison, MetricResult, RunResults


def settings_seed(config: RunConfig) -> int:
    seed = config.statistics.seed
    if seed is None:
        seed = config.seed
    return seed if seed is not None else 0


def row_scorer(
    metric: Metric, result: MetricResult, predictions: Predictions, references: References
) -> Scorer:
    """How to score a resample of the rows of one model."""
    if metric.averages_rows and result.per_row is not None:
        values = np.array([np.nan if v is None else v for v in result.per_row], dtype=float)
        return _mean_scorer(values)
    return metric.scorer(predictions, references)


def _mean_scorer(values: npt.NDArray[np.float64]) -> Scorer:
    def score(indices: Indices) -> float | None:
        sample = values[indices]
        sample = sample[~np.isnan(sample)]
        return float(sample.mean()) if len(sample) else None

    return score


def confidence_interval(
    scorer: Scorer, n: int, *, confidence: float = 0.95, resamples: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of the score over ``n`` rows."""
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = [scorer(rng.integers(0, n, n)) for _ in range(resamples)]
    values = np.array([s for s in samples if s is not None], dtype=float)
    if not len(values):
        return None
    alpha = (1 - confidence) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)


def add_intervals(
    results: RunResults, metrics: Sequence[Metric], scores: dict[str, dict[str, MetricResult]]
) -> None:
    """Set ``ci_low`` and ``ci_high`` on every score in ``scores``."""
    settings = results.config.statistics
    if not settings.intervals:
        return
    seed = settings_seed(results.config)
    references = [record.expected for record in results.data.records]
    n = len(references)
    for name, model_run in results.runs.items():
        predictions = [row.output for row in model_run.rows]
        for spec, metric in zip(results.config.metrics, metrics, strict=True):
            result = scores[name][spec.key]
            if result.value is None:
                continue
            interval = confidence_interval(
                row_scorer(metric, result, predictions, references),
                n,
                confidence=settings.confidence,
                resamples=settings.resamples,
                seed=seed,
            )
            if interval is not None:
                result.ci_low, result.ci_high = interval


def paired_test(
    metric: Metric,
    result_a: MetricResult,
    result_b: MetricResult,
    predictions_a: Predictions,
    predictions_b: Predictions,
    references: References,
    settings: StatisticsSettings,
    seed: int = 0,
) -> tuple[float | None, float | None, tuple[float, float] | None]:
    """The difference ``a - b``, its two-sided p-value, and (bootstrap only) its interval."""
    n = len(references)
    if result_a.value is None or result_b.value is None or n == 0:
        return None, None, None
    observed = result_a.value - result_b.value
    rng = np.random.default_rng(seed)
    rows = np.arange(n)

    if settings.significance == "bootstrap":
        score_a = row_scorer(metric, result_a, predictions_a, references)
        score_b = row_scorer(metric, result_b, predictions_b, references)
        differences = []
        for _ in range(settings.resamples):
            sample = rng.integers(0, n, n)
            a, b = score_a(sample), score_b(sample)
            if a is not None and b is not None:
                differences.append(a - b)
        if not differences:
            return observed, None, None
        deltas = np.array(differences)
        # Centre the bootstrap distribution on zero to get the null distribution.
        extreme = int(np.sum(np.abs(deltas - deltas.mean()) >= abs(observed) - 1e-12))
        alpha = (1 - settings.confidence) / 2
        low, high = np.quantile(deltas, [alpha, 1 - alpha])
        return observed, float((extreme + 1) / (len(deltas) + 1)), (float(low), float(high))

    # Approximate randomisation: score both models' rows as one list of 2n rows, where
    # row i is model a's and row n + i is model b's, and swap a random subset of pairs.
    if metric.averages_rows and result_a.per_row is not None and result_b.per_row is not None:
        values = np.array(
            [np.nan if v is None else v for v in [*result_a.per_row, *result_b.per_row]],
            dtype=float,
        )
        combined = _mean_scorer(values)
    else:
        combined = metric.scorer([*predictions_a, *predictions_b], [*references, *references])
    extreme = valid = 0
    for _ in range(settings.resamples):
        swap = rng.random(n) < 0.5
        x = combined(rows + n * swap)
        y = combined(rows + n * ~swap)
        if x is None or y is None:
            continue
        valid += 1
        extreme += int(abs(x - y) >= abs(observed) - 1e-12)
    if not valid:
        return observed, None, None
    return observed, (extreme + 1) / (valid + 1), None


def compare(results: RunResults) -> list[Comparison]:
    """Paired significance tests between models, as set in ``config.statistics``.

    Needs ``results.scores``. Compares every model with ``baseline`` if one is set, else
    every pair of models, on every metric. p-values are corrected per metric.
    """
    config = results.config
    settings = config.statistics
    names = list(results.runs)
    if settings.significance == "none" or len(names) < 2 or not results.scores:
        return []
    from paperlab_space.metrics import get_metric

    if settings.baseline is not None:
        pairs = [(m, settings.baseline) for m in names if m != settings.baseline]
    else:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    references = [record.expected for record in results.data.records]
    predictions = {name: [row.output for row in run.rows] for name, run in results.runs.items()}
    seed = settings_seed(config)
    comparisons: list[Comparison] = []
    for spec in config.metrics:
        metric = get_metric(spec)
        group: list[Comparison] = []
        for a, b in pairs:
            result_a = results.scores.get(a, {}).get(spec.key)
            result_b = results.scores.get(b, {}).get(spec.key)
            if result_a is None or result_b is None:
                continue
            difference, p_value, interval = paired_test(
                metric,
                result_a,
                result_b,
                predictions[a],
                predictions[b],
                references,
                settings,
                seed,
            )
            group.append(
                Comparison(
                    metric=spec.key,
                    model_a=a,
                    model_b=b,
                    difference=difference,
                    p_value=p_value,
                    p_adjusted=None,
                    significant=False,
                    method=settings.significance,
                    ci_low=interval[0] if interval else None,
                    ci_high=interval[1] if interval else None,
                )
            )
        adjusted = adjust([c.p_value for c in group], settings.correction)
        for comparison, p in zip(group, adjusted, strict=True):
            comparison.p_adjusted = p
            comparison.significant = p is not None and p < 1 - settings.confidence
        comparisons += group
    return comparisons


def adjust(p_values: Sequence[float | None], correction: str) -> list[float | None]:
    """Correct p-values for multiple comparisons (``holm``, ``bonferroni``, or ``none``)."""
    known = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adjusted: list[float | None] = list(p_values)
    m = len(known)
    if correction == "bonferroni":
        for i, p in known:
            adjusted[i] = min(1.0, p * m)
    elif correction == "holm":
        running = 0.0
        for rank, (i, p) in enumerate(sorted(known, key=lambda x: x[1])):
            running = max(running, min(1.0, (m - rank) * p))
            adjusted[i] = running
    elif correction != "none":
        raise ValueError(f"Unknown correction {correction!r}")
    return [None if p is None or math.isnan(p) else p for p in adjusted]
