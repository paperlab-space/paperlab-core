"""Containers for run outputs, plus JSON save/load for resuming and sharing runs."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from paperlab_space.config import ModelSpec, RunConfig
from paperlab_space.data.load import Dataset
from paperlab_space.data.mapping import MappedDataset
from paperlab_space.errors import DataError

FORMAT_VERSION = 1


@dataclass
class RowOutput:
    record_id: str
    """The ID of the data row."""
    output: str | None
    """The model's answer, or ``None`` if every attempt failed."""
    error: str | None = None
    """Why the request failed, if it did."""
    attempts: int = 0
    """Requests sent for this row, including retries."""
    latency_s: float | None = None
    """Seconds the successful request took."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    """Cost in USD, if the provider's price is known."""
    model_version: str | None = None
    """The exact model name the provider reported."""

    @property
    def ok(self) -> bool:
        return self.output is not None


@dataclass
class ModelRun:
    """One model's outputs for every row."""

    model: ModelSpec
    rows: list[RowOutput]
    """One output per data row, in data order."""

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def n_failed(self) -> int:
        return sum(not row.ok for row in self.rows)

    @property
    def cost_usd(self) -> float | None:
        costs = [row.cost_usd for row in self.rows if row.ok]
        return sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None

    @property
    def input_tokens(self) -> int:
        return sum(row.input_tokens or 0 for row in self.rows)

    @property
    def output_tokens(self) -> int:
        return sum(row.output_tokens or 0 for row in self.rows)

    @property
    def model_versions(self) -> list[str]:
        return sorted({row.model_version for row in self.rows if row.model_version})


@dataclass
class MetricResult:
    """The score of one metric for one model."""

    name: str
    """The metric's key: its label, or its name."""
    value: float | None
    """The headline number (``None`` if no row could be scored)."""
    per_row: list[float | None] | None = None
    """Row-level scores aligned with the dataset records, if the metric has them."""
    extra: dict[str, float] = field(default_factory=dict)
    """Secondary numbers, e.g. precision and recall next to F1."""
    n_scored: int = 0
    """Rows that counted towards the score."""
    higher_is_better: bool = True


@dataclass
class RunResults:
    """Everything a run produced: outputs, scores, and the config and data behind them."""

    config: RunConfig
    data: MappedDataset
    """The data the run used, with the mapping applied."""
    runs: dict[str, ModelRun]
    """Model name -> outputs, in config order."""
    scores: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    """Model name -> metric name -> result."""
    started_at: str | None = None
    """ISO 8601 timestamp (UTC)."""
    finished_at: str | None = None
    """ISO 8601 timestamp (UTC)."""
    environment: dict[str, str] = field(default_factory=lambda: environment_info())
    """Python, paperlab-space, and LiteLLM versions."""

    @property
    def model_names(self) -> list[str]:
        return list(self.runs)

    def output(self, model: str, record_id: str) -> RowOutput:
        for row in self.runs[model].rows:
            if row.record_id == record_id:
                return row
        raise KeyError(record_id)

    # --- JSON -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        dataset = self.data.dataset
        return {
            "format": FORMAT_VERSION,
            "environment": self.environment,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self._shareable_config(),
            "dataset": {
                "source": Path(dataset.source).name if dataset.source else None,
                "sheet": dataset.sheet,
                "sha256": dataset.sha256,
                "columns": dataset.columns,
                "rows": dataset.rows,
            },
            "runs": {name: [asdict(row) for row in run.rows] for name, run in self.runs.items()},
            "scores": {
                model: {name: asdict(result) for name, result in metrics.items()}
                for model, metrics in self.scores.items()
            },
        }

    def _shareable_config(self) -> dict[str, Any]:
        # Keep only the file name of a local dataset path, so sharing results.json does not
        # reveal folder names (the same rule as the reproducibility config).
        config = self.config.to_dict()
        if config.get("dataset"):
            config["dataset"]["path"] = Path(config["dataset"]["path"]).name
        return config

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResults:
        if data.get("format") != FORMAT_VERSION:
            raise DataError(f"Unsupported results format {data.get('format')!r}.")
        config = RunConfig.from_dict(data["config"])
        ds = data["dataset"]
        dataset = Dataset(
            columns=ds["columns"],
            rows=ds["rows"],
            source=ds["source"],
            sheet=ds["sheet"],
            sha256=ds["sha256"],
        )
        models = {m.name: m for m in config.models}
        runs = {
            name: ModelRun(models[name], [RowOutput(**row) for row in rows])
            for name, rows in data["runs"].items()
        }
        scores = {
            model: {name: MetricResult(**result) for name, result in metrics.items()}
            for model, metrics in data.get("scores", {}).items()
        }
        return cls(
            config=config,
            data=dataset.map(config.mapping),
            runs=runs,
            scores=scores,
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            environment=data.get("environment", {}),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> RunResults:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def environment_info() -> dict[str, str]:
    info = {"python": platform.python_version()}
    for package in ("paperlab-space", "litellm"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "unknown"
    return info
