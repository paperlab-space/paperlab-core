# Custom metrics

Add a metric of your own and use it by name, like the built-in ones.

## Write the metric

Subclass `Metric`, declare its options as `__init__` arguments, and register the class:

```python
from paperlab_space import register
from paperlab_space.metrics import Metric


@register
class StartsWith(Metric):
    name = "starts_with"
    description = "Share of outputs that start with the reference answer."

    def __init__(self, ignore_case: bool = True) -> None:
        self.ignore_case = ignore_case

    def compute(self, predictions, references):
        scores = []
        for output, refs in zip(predictions, references):
            if not refs:  # no answer key for this row
                scores.append(None)
            elif output is None:  # the request failed
                scores.append(0.0)
            else:
                fold = str.casefold if self.ignore_case else str
                scores.append(float(any(fold(output).startswith(fold(r)) for r in refs)))
        values = [s for s in scores if s is not None]
        mean = sum(values) / len(values) if values else None
        return self.result(mean, per_row=scores, n_scored=len(values))
```

`compute` gets every row of one model at once:

- `predictions`: one output per row, or `None` where the request failed.
- `references`: the valid answers per row, or `None` where the row has no answer key entry.

It returns a `MetricResult`, built with `self.result(...)`:

| Argument | What it is |
| --- | --- |
| `value` | The headline number, or `None` if no row could be scored. |
| `per_row` | Row-level scores aligned with the rows (`None` for rows not scored). These appear in the Outputs sheet. |
| `n_scored` | How many rows counted. |
| `extra` | Secondary numbers by name, such as precision next to F1. They are saved in `results.json`. |

## Class attributes

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | required | The name used in run configs. It must be unique. |
| `description` | `""` | One sentence, shown by `paperlab-space metrics`. |
| `requires_reference` | `True` | Set `False` for a metric that does not need an answer key, such as output length. Without an answer key, `references` is all `None`. |
| `higher_is_better` | `True` | Set `False` where lower is better, so the LaTeX table bolds the right score. |
| `averages_rows` | `False` | Set `True` if `value` is the mean of `per_row`. Confidence intervals and significance tests then resample the row scores, which is fast. |

## Confidence intervals for your metric

[Confidence intervals and significance tests](../guides/statistics.md) score your metric on resamples of the rows. With `averages_rows = True`, they average the resampled row scores. Otherwise they call `compute` once per resample, 1,000 times per model by default. That is fine for a quick metric. For a slow corpus-level metric, override `scorer`. It returns a function that scores the rows at the given positions, which may repeat:

```python
import numpy as np

from paperlab_space.metrics import Metric


class TotalLength(Metric):
    name = "total_length"
    requires_reference = False

    def compute(self, predictions, references):
        return self.result(float(sum(len(p or "") for p in predictions)))

    def scorer(self, predictions, references):
        lengths = np.array([len(p or "") for p in predictions])
        return lambda indices: float(lengths[indices].sum())
```

To check that a metric can run on the data before any request is sent, override `check(self, data)` and raise `ConfigError`.

## Use it

Once the module that defines it has been imported, the metric works by name in a `RunConfig`, including one loaded from YAML:

```python
import my_metrics  # registers starts_with
from paperlab_space import RunConfig, run

config = RunConfig.from_yaml("run.yaml")  # metrics: [{name: starts_with, ignore_case: false}]
results = run(config)
```

Options are checked against the `__init__` arguments, so a misspelt option is an error before any request is sent. Like the built-in metrics, a custom metric can appear twice with different options if each entry has its own `label`.

The `paperlab-space` command does not import your code, so custom metrics are for Python use.

## Reuse the built-in normalisation

To compare answers the way the built-in metrics do, use `Normalizer`, `tokenize`, and `extract_label` from `paperlab_space.metrics`:

```python
from paperlab_space.metrics import Normalizer, tokenize

normalize = Normalizer(ignore_punctuation=True)
tokens = tokenize(normalize("今日は 晴れ!"))  # ['今', '日', 'は', '晴', 'れ']
```

See the [Python API reference](../reference/python.md#metrics) for their options.

## Score existing results

`score` recomputes every configured metric for results you already have, without new requests. Use it to add a metric to an old run:

```python
from paperlab_space import MetricSpec, RunResults, export, score, stats

results = RunResults.load_json("results-run/results.json")
results.config = results.config.model_copy(
    update={"metrics": [*results.config.metrics, MetricSpec(name="starts_with")]}
)
results.scores = score(results)
results.comparisons = stats.compare(results)
export.write_all(results, "results-run/")
```

This does not work for `llm_judge`, which needs new requests. To add a judge to an old run, see [LLM-as-judge](../guides/llm-judge.md#cost-and-time).
