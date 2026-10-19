"""The run configuration: one object that fully describes an evaluation.

The same :class:`RunConfig` drives a run and is written out as the reproducibility
config, so a run can be repeated from its exported config alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from paperlab_space.data.mapping import ColumnMapping
from paperlab_space.errors import ConfigError
from paperlab_space.prompt import PromptTemplate


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    """A model to evaluate, addressed by its LiteLLM ID (``provider/model``)."""

    id: str
    """LiteLLM model ID with its provider prefix, e.g. ``gemini/gemini-3.5-flash``."""
    label: str | None = None
    """Display name in tables. Defaults to the model ID without the provider prefix."""
    temperature: float = Field(0.0, ge=0)
    """Sampling temperature. 0 gives the most repeatable outputs."""
    max_tokens: int | None = Field(None, gt=0)
    """Cap on output tokens per response."""
    top_p: float | None = Field(None, gt=0, le=1)
    """Nucleus sampling cutoff."""
    seed: int | None = None
    """Overrides the run-level ``seed`` for this model."""
    requests_per_minute: float | None = Field(None, gt=0)
    """Override the default rate limit for this model."""
    concurrency: int | None = Field(None, gt=0)
    """Override the run-level number of parallel requests for this model."""
    params: dict[str, Any] = {}
    """Extra provider parameters passed through to LiteLLM."""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        value = value.strip()
        if "/" not in value:
            raise ValueError(
                f"Model ID {value!r} needs a provider prefix, e.g. 'openrouter/{value}' or "
                f"'gemini/{value}'."
            )
        return value

    @property
    def provider(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def name(self) -> str:
        """The label used in outputs."""
        return self.label or self.id.split("/", 1)[1]


class MetricSpec(_Strict):
    """A metric name plus its options, e.g. ``{"name": "exact_match", "ignore_case": false}``."""

    name: str
    """A registered metric name; ``paperlab-space metrics`` lists them."""
    label: str | None = None
    """Column name in outputs; needed to use the same metric twice with different options."""
    params: dict[str, Any] = {}
    """Metric options such as ``labels`` or ``ignore_case``. In YAML they can also sit next to
    ``name``."""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        if isinstance(value, dict) and "name" in value and "params" not in value:
            # Allow the flat form ``{name: f1, average: macro}``.
            keys = {"name", "label"}
            return {
                **{k: v for k, v in value.items() if k in keys},
                "params": {k: v for k, v in value.items() if k not in keys},
            }
        return value

    @property
    def key(self) -> str:
        return self.label or self.name


class DatasetSpec(_Strict):
    """Where the data comes from. Optional when the dataset is passed in directly."""

    path: str
    """Data file. A relative path is resolved against the config file's folder."""
    sheet: str | int | None = None
    """Excel sheet name or zero-based index. Defaults to the first sheet."""


class RetrySettings(_Strict):
    """How failed requests are retried, and how long to wait for a response."""

    max_attempts: int = Field(5, ge=1)
    """Tries per row, including the first."""
    base_delay: float = Field(2.0, ge=0)
    """Seconds before the first retry; doubles on each attempt."""
    max_delay: float = Field(60.0, ge=0)
    """Upper bound on the wait between retries, in seconds."""
    timeout: float = Field(120.0, gt=0)
    """Seconds to wait for one model response."""


class StatisticsSettings(_Strict):
    """Confidence intervals for every score, and significance tests between models."""

    confidence: float = Field(0.95, gt=0, lt=1)
    """Confidence level of the intervals. A difference is significant when its corrected
    p-value is below ``1 - confidence``."""
    intervals: bool = True
    """Compute a bootstrap confidence interval for every score."""
    significance: Literal["permutation", "bootstrap", "none"] = "permutation"
    """Paired test between models: ``permutation`` (approximate randomisation),
    ``bootstrap`` (paired bootstrap resampling), or ``none``."""
    baseline: str | None = None
    """Compare every other model with this one (a model label). Default: every pair."""
    correction: Literal["holm", "bonferroni", "none"] = "holm"
    """Multiple-comparison correction, applied per metric across the model pairs."""
    resamples: int = Field(1000, ge=100)
    """Bootstrap samples or permutation rounds."""
    seed: int | None = None
    """Seed for resampling. Defaults to the run ``seed``, or 0."""


class RunConfig(_Strict):
    """Everything needed to run (and re-run) an evaluation."""

    dataset: DatasetSpec | None = None
    """Where to load the data from. Optional when the data is passed to ``run`` directly."""
    mapping: ColumnMapping
    """Which columns are inputs, context, the answer key, and the row ID."""
    prompt: PromptTemplate
    """The prompt template: a string, or an object with ``user`` and ``system``."""
    models: list[ModelSpec]
    """Models to evaluate, as IDs or objects with options."""
    metrics: list[MetricSpec] = []
    """Metrics to compute, as names or objects with options. Empty collects outputs only."""
    seed: int | None = None
    """Default seed for models that do not set their own."""
    concurrency: int = Field(4, gt=0)
    """Parallel requests per model (rate limits still apply)."""
    retry: RetrySettings = RetrySettings()
    """Retry and timeout settings."""
    statistics: StatisticsSettings = StatisticsSettings()
    """Confidence intervals and significance tests."""

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if not self.models:
            raise ValueError("Pick at least one model.")
        names = [m.name for m in self.models]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"Models must have distinct labels; duplicated: {', '.join(dupes)}. "
                "Set `label` to tell them apart (e.g. the same model at two temperatures)."
            )
        baseline = self.statistics.baseline
        if baseline is not None and baseline not in names:
            raise ValueError(
                f"statistics.baseline {baseline!r} is not one of the models: {', '.join(names)}"
            )
        keys = [m.key for m in self.metrics]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(
                f"Metrics must have distinct labels; duplicated: {', '.join(dupes)}. "
                "Set `label` to use the same metric with different options."
            )
        try:
            self.prompt.check(self.mapping.fields)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        return self

    def model_seed(self, model: ModelSpec) -> int | None:
        return model.seed if model.seed is not None else self.seed

    # --- YAML -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Build from a dict; also accepts a reproducibility file (config under ``config``)."""
        if "config" in data and "paperlab_space" in data:
            data = data["config"]
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(_format_errors(exc)) from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a YAML config. A relative dataset path is resolved against the file's folder."""
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Could not read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} does not contain a YAML mapping.")
        config = cls.from_dict(data)
        if config.dataset is not None and not Path(config.dataset.path).is_absolute():
            resolved = (path.parent / config.dataset.path).resolve()
            config = config.model_copy(
                update={"dataset": config.dataset.model_copy(update={"path": str(resolved)})}
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_defaults=False)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _format_errors(exc: ValidationError) -> str:
    lines = ["Invalid run configuration:"]
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "config"
        lines.append(f"  {where}: {error['msg']}")
    return "\n".join(lines)
