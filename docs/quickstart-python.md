# Quickstart (Python)

Run the same evaluation from a script or a Jupyter notebook.

## A minimal run

```python
from paperlab_space import ColumnMapping, RunConfig, estimate, export, load_dataset, run

data = load_dataset("my_data.xlsx", sheet="test")
config = RunConfig(
    mapping=ColumnMapping(input="question", expected="answer"),
    prompt="Answer briefly: {question}",
    models=["openrouter/nvidia/nemotron-3-super-120b-a12b:free", "gemini/gemini-3.5-flash"],
    metrics=["exact_match", "token_f1"],
)
print(estimate(config, data))
results = run(config, data)
print(results.scores["gemini-3.5-flash"]["exact_match"].value)
export.write_all(results, "results/")
```

`RunConfig` takes the same fields as `run.yaml`, and the short forms work here too: a model can be a plain ID string, a metric a plain name. Scores are keyed by the model's label, which defaults to its ID without the provider prefix.

## Load a config file instead

```python
from paperlab_space import RunConfig, run

config = RunConfig.from_yaml("run.yaml")
results = run(config)  # loads the data from config.dataset
```

A relative `dataset.path` is resolved against the YAML file's folder.

## Check the data first

`run` refuses data with validation errors. To see the problems yourself:

```python
mapped = data.map(config.mapping)
report = mapped.validate()
print(report)  # errors and warnings with row numbers
print(mapped.preview(3))
```

## Jupyter and async code

`run` works in scripts and in Jupyter, where an event loop is already running. Inside your own async code, use `arun`:

```python
from paperlab_space import arun

results = await arun(config, data)
```

## Progress

Pass `on_progress` to see each row as it finishes:

```python
def show(p):
    print(f"{p.model}: {p.done}/{p.total} ({p.failed} failed)")


results = run(config, data, on_progress=show)
```

## Pass API keys directly

By default, keys come from environment variables. To pass them in code, use a `LiteLLMBackend`, keyed by provider prefix:

```python
from paperlab_space import LiteLLMBackend, run

backend = LiteLLMBackend(api_keys={"openrouter": "sk-or-...", "gemini": "..."})
results = run(config, data, backend=backend)
```

Keys are never written to the config or the results.

## Work with the results

```python
for name, model_run in results.runs.items():
    print(name, model_run.n_failed, model_run.cost_usd)

row = results.output("gemini-3.5-flash", "17")  # by record ID
print(row.output, row.error, row.latency_s)
```

`export.write_all` writes all four files. To write one, use `export.write_excel`, `export.write_latex`, or `export.write_repro`. `export.latex_table(results, decimals=2, percent=False)` returns the table as a string. See [Output files](guides/outputs.md).
