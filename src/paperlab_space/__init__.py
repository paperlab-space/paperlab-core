"""paperlab-space: evaluate LLMs on spreadsheet data and produce paper-ready results.

>>> from paperlab_space import RunConfig, load_dataset, run, export
>>> data = load_dataset("data.xlsx")                          # doctest: +SKIP
>>> config = RunConfig.from_yaml("run.yaml")                  # doctest: +SKIP
>>> results = run(config, data)                               # doctest: +SKIP
>>> export.write_all(results, "results/")                     # doctest: +SKIP
"""

__version__ = "0.1.1"

from paperlab_space import export
from paperlab_space.config import DatasetSpec, MetricSpec, ModelSpec, RetrySettings, RunConfig
from paperlab_space.data import ColumnMapping, Dataset, MappedDataset, list_sheets, load_dataset
from paperlab_space.errors import ConfigError, DataError, PaperLabError, TemplateError
from paperlab_space.metrics import available_metrics, register, score
from paperlab_space.prompt import PromptTemplate
from paperlab_space.results import MetricResult, ModelRun, RowOutput, RunResults
from paperlab_space.runner import LiteLLMBackend, Progress, arun, estimate, run

__all__ = [
    "ColumnMapping",
    "ConfigError",
    "DataError",
    "Dataset",
    "DatasetSpec",
    "LiteLLMBackend",
    "MappedDataset",
    "MetricResult",
    "MetricSpec",
    "ModelRun",
    "ModelSpec",
    "PaperLabError",
    "Progress",
    "PromptTemplate",
    "RetrySettings",
    "RowOutput",
    "RunConfig",
    "RunResults",
    "TemplateError",
    "__version__",
    "arun",
    "available_metrics",
    "estimate",
    "export",
    "list_sheets",
    "load_dataset",
    "register",
    "run",
    "score",
]
