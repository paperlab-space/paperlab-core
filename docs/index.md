# Overview

Evaluate LLMs on your own spreadsheet data and get paper-ready results, from the command line or from Python.

`paperlab-space` is the open-source engine behind [PaperLab.space](https://paperlab.space). You give it an Excel or CSV file, say which columns are inputs and answers, and pick models and metrics. It sends every row to every model, scores the outputs, and writes:

- `results.xlsx`: scores, per-row outputs, and the config
- `table.tex`: a booktabs table to `\input{}` into your paper
- `run.yaml`: a reproducibility config that re-runs the same evaluation
- `results.json`: everything, for resuming and sharing

## Install

```bash
pip install paperlab-space
```

This installs the Python library and the `paperlab-space` command. Python 3.10 or newer is required.

```bash
paperlab-space --version
```

## What it handles for you

- **Spreadsheets in.** `.xlsx` (any sheet), `.csv`, and `.tsv`, with encoding detection and Unicode-safe handling. Validation catches empty cells, duplicate IDs, and garbled text before you spend API calls. See [Preparing your data](guides/data.md).
- **Any model.** OpenRouter (including free models), Gemini, OpenAI, Anthropic, and everything else [LiteLLM](https://docs.litellm.ai/docs/providers) supports, run side by side with your own API keys. See [Models and API keys](guides/models.md).
- **Robust runs.** Per-model rate limits tuned for free tiers, retries with backoff, and a cost and time estimate before anything is sent. Failed rows can be retried without re-running the rest. See [Resuming a run](guides/resuming.md).
- **Metrics.** Exact match, accuracy with label extraction, precision/recall/F1, token F1, BLEU, chrF, and ROUGE, with tokenisation for Chinese, Japanese, Thai, and Burmese. See [Metrics](guides/metrics.md).
- **LLM-as-judge.** A model grades open-ended answers against the reference or your rubric, and its reasoning is saved for every row. See [LLM-as-judge](guides/llm-judge.md).
- **Statistics.** A bootstrap confidence interval for every score, and paired significance tests between models with multiple-comparison correction. See [Statistics](guides/statistics.md).
- **Reproducibility.** The exported config records model versions, the prompt, the seed, metric settings, and a hash of your data. See [Reproducibility and citing](guides/reproducibility.md).

## Next steps

- [Quickstart (command line)](quickstart-cli.md): your first run in five commands.
- [Quickstart (Python)](quickstart-python.md): the same run from a script or notebook.

## Stability

From 1.0, paperlab-space follows [semantic versioning](https://semver.org). Within 1.x, these keep working the same way:

- the Python API: everything importable from `paperlab_space`, and the documented parts of `paperlab_space.metrics`, `paperlab_space.export`, `paperlab_space.runner`, and `paperlab_space.stats`
- the `run.yaml` format: configs written for 1.0 load in every 1.x release
- the `paperlab-space` commands and their options
- the `results.json` format: results from any 1.x release load in later ones

New features, metrics, and config fields can be added in minor releases. A default that changes scores, such as a metric's normalisation, changes only in a major release. Each run records the exact version in `run.yaml`. See the [changelog](changelog.md).
