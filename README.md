# PaperLab.space core

```bash
pip install paperlab-space
```

Evaluate LLMs on your own spreadsheet data and get paper-ready results.

Give it an Excel or CSV file, say which columns are inputs and answers, pick models and metrics, and get back an Excel workbook of outputs and scores, a LaTeX table for your paper, and a reproducibility config. `paperlab-space` is the open-source engine behind [PaperLab.space](https://paperlab.space); use it from the command line or from Python.

- **Spreadsheets in:** `.xlsx` (any sheet), `.csv`, `.tsv`, with encoding detection and Unicode-safe handling for non-Latin scripts. Validation catches empty cells, duplicate IDs, and garbled text before you spend API calls.
- **Any model:** OpenRouter (including free models), Gemini, OpenAI, Anthropic, Ollama, and everything else [LiteLLM](https://docs.litellm.ai/docs/providers) supports, run side by side. You bring your own API keys.
- **Robust runs:** per-model rate limits tuned for free tiers, retries with backoff, and resume after interruption. A cost and time estimate is shown before anything is sent.
- **Metrics:** exact match, accuracy (with label extraction from chatty answers), precision/recall/F1, token F1. Tokenisation handles Chinese, Japanese, Thai, and Burmese.
- **Paper-ready output:** Excel, a booktabs LaTeX table, and a YAML config recording model versions, prompt, seed, metric settings, and a hash of your data.

[![PyPI](https://img.shields.io/pypi/v/paperlab-space)](https://pypi.org/project/paperlab-space/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23436738.svg)](https://doi.org/10.5281/zenodo.23436738)

> **Status:** early release (0.1). See [PLAN.md](PLAN.md) for the roadmap: BLEU/chrF/ROUGE, LLM-as-judge, and significance tests are next.

## Install

```bash
pip install paperlab-space
```

This installs both the Python library and the `paperlab-space` command.

## Quickstart (command line)

```bash
export OPENROUTER_API_KEY=sk-or-...   # free key from https://openrouter.ai
export GEMINI_API_KEY=...             # free key from https://aistudio.google.com

paperlab-space init my_data.xlsx            # writes run.yaml, guessing the column roles
paperlab-space validate my_data.xlsx -i question -e answer   # check the data, preview JSONL
paperlab-space estimate run.yaml            # tokens, cost, and time; no requests sent
paperlab-space run run.yaml                 # writes results-run/
```

`run.yaml` looks like this (see [examples/sentiment](examples/sentiment)):

```yaml
dataset:
  path: data.csv
mapping:
  input: text            # column(s) available to the prompt
  context: [language]    # extra columns available to the prompt
  expected: label        # the answer key (never shown to the model)
  id: id
prompt: |-
  Classify the sentiment of this {language} text as "positive" or "negative".
  Text: {text}
models:
  - openrouter/nvidia/nemotron-3-super-120b-a12b:free
  - id: gemini/gemini-2.5-flash
    temperature: 0
metrics:
  - name: accuracy
    labels: [positive, negative]
  - f1
seed: 42
```

The output folder contains:

| File | What it is |
| --- | --- |
| `results.xlsx` | Summary sheet (scores, failures, cost), per-row outputs and scores, and the config |
| `table.tex` | A booktabs table to `\input{}` into your paper |
| `run.yaml` | The reproducibility config: re-run with `paperlab-space run run.yaml` |
| `results.json` | Everything, for `paperlab-space run run.yaml --resume results.json` after an interruption |

Free models change often; `paperlab-space run` stops a model early with a clear message if its ID no longer exists. Browse current free models at [openrouter.ai/models](https://openrouter.ai/models?q=free).

## Quickstart (Python)

```python
from paperlab_space import ColumnMapping, RunConfig, estimate, export, load_dataset, run

data = load_dataset("my_data.xlsx", sheet="test")
config = RunConfig(
    mapping=ColumnMapping(input="question", expected="answer"),
    prompt="Answer briefly: {question}",
    models=["openrouter/nvidia/nemotron-3-super-120b-a12b:free", "gemini/gemini-2.5-flash"],
    metrics=["exact_match", "token_f1"],
)
print(estimate(config, data))
results = run(config, data)
print(results.scores["gemini-2.5-flash"]["exact_match"].value)
export.write_all(results, "results/")
```

`run` works in scripts and Jupyter notebooks. Use `await arun(...)` inside your own async code.

## Metrics

Run `paperlab-space metrics` for the list. Every metric takes normalisation options in the config:

```yaml
metrics:
  - exact_match                         # case-insensitive, trims punctuation like "Paris."
  - name: exact_match
    label: exact_match_strict           # same metric again with different options
    ignore_case: false
    trim_punctuation: false
  - name: f1
    average: macro                      # macro | micro | weighted
```

Custom metrics subclass `paperlab_space.metrics.Metric` and use the `@register` decorator.

## Citing

If paperlab-space helps your research, please cite it:

```bibtex
@software{paperlab_space,
  author    = {{Kaung Nay Lin Khant}},
  title     = {paperlab-space: no-code LLM evaluation on spreadsheet data},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23436738},
  url       = {https://doi.org/10.5281/zenodo.23436738}
}
```

This DOI always points to the latest version. To cite the exact version you used (recommended for reproducibility), pick it from the [Zenodo record](https://doi.org/10.5281/zenodo.23436738); the version is also recorded in your `run.yaml`. GitHub's "Cite this repository" button gives the same information from [CITATION.cff](CITATION.cff).

## Development

```bash
uv sync
uv run pytest                   # unit tests, no network
uv run pytest -m live           # real-provider smoke tests (needs API keys)
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
```

## License

[Apache 2.0](LICENSE)
