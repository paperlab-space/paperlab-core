"""Reproducibility config: everything needed to repeat a run, in one YAML file.

The file can be loaded back with :meth:`RunConfig.from_yaml`; put the data file next to
it (the dataset path is stored as a bare file name so no local paths leak into papers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from paperlab_core.results import RunResults


def reproducibility(results: RunResults) -> dict[str, Any]:
    config = results.config.to_dict()
    dataset = results.data.dataset
    source_name = Path(dataset.source).name if dataset.source else None
    if config.get("dataset") is not None and source_name:
        config["dataset"]["path"] = source_name
    elif source_name:
        config["dataset"] = {"path": source_name, "sheet": dataset.sheet}
    return {
        "paperlab_core": results.environment.get("paperlab-core", "unknown"),
        "created": results.finished_at,
        "environment": results.environment,
        "data": {
            "file": source_name,
            "sheet": dataset.sheet,
            "sha256": dataset.sha256,
            "rows": len(results.data.records),
        },
        "model_versions": {
            name: run.model_versions for name, run in results.runs.items() if run.model_versions
        },
        "config": config,
    }


def write_repro(results: RunResults, path: str | Path) -> Path:
    path = Path(path)
    text = yaml.safe_dump(reproducibility(results), allow_unicode=True, sort_keys=False)
    path.write_text(
        "# PaperLab reproducibility config. Re-run with: paperlab run " + path.name + "\n" + text,
        encoding="utf-8",
    )
    return path
