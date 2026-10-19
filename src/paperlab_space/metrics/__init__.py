"""Scoring model outputs."""

from __future__ import annotations

from paperlab_space.config import RunConfig
from paperlab_space.data.mapping import MappedDataset
from paperlab_space.errors import ConfigError
from paperlab_space.metrics import builtin as _builtin  # noqa: F401  (registers metrics)
from paperlab_space.metrics import text as _text  # noqa: F401
from paperlab_space.metrics.base import (
    Indices,
    Metric,
    Predictions,
    References,
    Scorer,
    available_metrics,
    get_metric,
    register,
)
from paperlab_space.metrics.judge import LLMJudge
from paperlab_space.metrics.normalize import Normalizer, extract_label, tokenize
from paperlab_space.results import MetricResult, RunResults


def check_metrics(config: RunConfig, data: MappedDataset) -> list[Metric]:
    """Instantiate the configured metrics and check they can run on ``data``."""
    metrics = [get_metric(spec) for spec in config.metrics]
    if not data.has_answer_key:
        needs = [
            spec.key
            for spec, m in zip(config.metrics, metrics, strict=True)
            if m.requires_reference
        ]
        if needs:
            raise ConfigError(
                f"Metric(s) {', '.join(needs)} need an answer key. Map an expected-output "
                "column, or pick metrics that work without one."
            )
    for metric in metrics:
        metric.check(data)
    return metrics


def score(results: RunResults) -> dict[str, dict[str, MetricResult]]:
    """Compute every configured metric for every model in ``results``, with confidence
    intervals if ``config.statistics.intervals`` is on.

    ``llm_judge`` metrics are scored from ``results.judgments``; a model without judgments
    gets the value ``None``.
    """
    from paperlab_space.stats import add_intervals

    metrics = check_metrics(results.config, results.data)
    references = [record.expected for record in results.data.records]
    record_ids = [record.id for record in results.data.records]
    scores: dict[str, dict[str, MetricResult]] = {}
    for name, model_run in results.runs.items():
        predictions = [row.output for row in model_run.rows]
        scores[name] = {}
        for spec, metric in zip(results.config.metrics, metrics, strict=True):
            if isinstance(metric, LLMJudge):
                judged = results.judgments.get(spec.key, {}).get(name)
                refs = references if metric.uses_reference(results.data) else None
                result = metric.from_judgments(record_ids, predictions, judged, refs)
            else:
                result = metric.compute(predictions, references)
            result.name = spec.key
            scores[name][spec.key] = result
    add_intervals(results, metrics, scores)
    return scores


__all__ = [
    "Indices",
    "LLMJudge",
    "Metric",
    "MetricResult",
    "Normalizer",
    "Predictions",
    "References",
    "Scorer",
    "available_metrics",
    "check_metrics",
    "extract_label",
    "get_metric",
    "register",
    "score",
    "tokenize",
]
