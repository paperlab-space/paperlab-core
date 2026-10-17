"""The metric interface and registry."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from paperlab_core.config import MetricSpec
from paperlab_core.errors import ConfigError
from paperlab_core.results import MetricResult

Predictions = list[str | None]
"""Model outputs; ``None`` marks a failed request."""
References = list[list[str] | None]
"""Valid answers per row; ``None`` marks a row without an answer key entry."""


class Metric(ABC):
    """Base class for metrics. Subclasses declare their options as ``__init__`` arguments."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    requires_reference: ClassVar[bool] = True
    higher_is_better: ClassVar[bool] = True

    @abstractmethod
    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        """Score all rows of one model."""

    def result(self, value: float | None, **kwargs: Any) -> MetricResult:
        return MetricResult(
            name=self.name, value=value, higher_is_better=self.higher_is_better, **kwargs
        )


_REGISTRY: dict[str, type[Metric]] = {}


def register(cls: type[Metric]) -> type[Metric]:
    """Class decorator that makes a metric available by name in run configs."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"A metric named {cls.name!r} is already registered.")
    _REGISTRY[cls.name] = cls
    return cls


def available_metrics() -> dict[str, type[Metric]]:
    return dict(_REGISTRY)


def get_metric(spec: MetricSpec | str) -> Metric:
    """Instantiate a metric from its config entry."""
    if isinstance(spec, str):
        spec = MetricSpec(name=spec)
    cls = _REGISTRY.get(spec.name)
    if cls is None:
        raise ConfigError(
            f"Unknown metric {spec.name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        )
    allowed = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    unknown = sorted(set(spec.params) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Metric {spec.name!r} has no option(s) {', '.join(unknown)}. "
            f"Options: {', '.join(allowed) or 'none'}"
        )
    try:
        return cls(**spec.params)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid options for metric {spec.name!r}: {exc}") from exc
