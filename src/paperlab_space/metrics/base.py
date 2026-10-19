"""The metric interface and registry."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from paperlab_space.config import MetricSpec
from paperlab_space.errors import ConfigError
from paperlab_space.results import MetricResult

if TYPE_CHECKING:
    from paperlab_space.data.mapping import MappedDataset

Predictions = list[str | None]
"""Model outputs; ``None`` marks a failed request."""
References = list[list[str] | None]
"""Valid answers per row; ``None`` marks a row without an answer key entry."""
Indices = npt.NDArray[np.intp]
"""Row positions, possibly repeated (a bootstrap sample)."""
Scorer = Callable[[Indices], "float | None"]
"""Scores the rows at the given positions."""


class Metric(ABC):
    """Base class for metrics. Subclasses declare their options as ``__init__`` arguments."""

    name: ClassVar[str]
    """The name used in run configs."""
    description: ClassVar[str] = ""
    """One sentence, shown by ``paperlab-space metrics``."""
    requires_reference: ClassVar[bool] = True
    """Whether the metric needs an expected-answer column."""
    higher_is_better: ClassVar[bool] = True
    """Which way is better, so the LaTeX table bolds the right score."""
    averages_rows: ClassVar[bool] = False
    """Whether the score is the mean of ``per_row``. Such metrics are resampled from their
    row scores; others through :meth:`scorer`."""

    @abstractmethod
    def compute(self, predictions: Predictions, references: References) -> MetricResult:
        """Score all rows of one model."""

    def check(self, data: MappedDataset) -> None:
        """Raise :class:`ConfigError` if the metric cannot run on ``data``. Called before any
        request is sent. The answer-key check for ``requires_reference`` is done already."""
        return None

    def scorer(self, predictions: Predictions, references: References) -> Scorer:
        """A function that scores any resample of the rows, for confidence intervals and
        significance tests. The default recomputes :meth:`compute` on the resampled rows;
        override it with something faster for corpus-level metrics."""

        def score(indices: Indices) -> float | None:
            return self.compute(
                [predictions[i] for i in indices], [references[i] for i in indices]
            ).value

        return score

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
