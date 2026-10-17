"""Scoring model outputs."""

from __future__ import annotations

from paperlab_core.config import RunConfig
from paperlab_core.data.mapping import MappedDataset
from paperlab_core.errors import ConfigError
from paperlab_core.metrics import builtin as _builtin  # noqa: F401  (registers metrics)
from paperlab_core.metrics.base import (
    Metric,
    Predictions,
    References,
    available_metrics,
    get_metric,
    register,
)
from paperlab_core.metrics.normalize import Normalizer, extract_label, tokenize
from paperlab_core.results import MetricResult, RunResults


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
    return metrics


def score(results: RunResults) -> dict[str, dict[str, MetricResult]]:
    """Compute every configured metric for every model in ``results``."""
    metrics = check_metrics(results.config, results.data)
    references = [record.expected for record in results.data.records]
    scores: dict[str, dict[str, MetricResult]] = {}
    for name, model_run in results.runs.items():
        predictions = [row.output for row in model_run.rows]
        scores[name] = {}
        for spec, metric in zip(results.config.metrics, metrics, strict=True):
            result = metric.compute(predictions, references)
            result.name = spec.key
            scores[name][spec.key] = result
    return scores


__all__ = [
    "Metric",
    "MetricResult",
    "Normalizer",
    "Predictions",
    "References",
    "available_metrics",
    "check_metrics",
    "extract_label",
    "get_metric",
    "register",
    "score",
    "tokenize",
]
