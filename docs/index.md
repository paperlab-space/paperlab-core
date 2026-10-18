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
- **Metrics.** Exact match, accuracy with label extraction, precision/recall/F1, and token F1, with tokenisation for Chinese, Japanese, Thai, and Burmese. See [Metrics](guides/metrics.md).
- **Reproducibility.** The exported config records model versions, the prompt, the seed, metric settings, and a hash of your data. See [Reproducibility and citing](guides/reproducibility.md).

## Next steps

- [Quickstart (command line)](quickstart-cli.md): your first run in five commands.
- [Quickstart (Python)](quickstart-python.md): the same run from a script or notebook.

> **Status:** early release (0.1). BLEU, chrF, ROUGE, LLM-as-judge, and significance tests are on the [roadmap](https://github.com/paperlab-space/paperlab-core/blob/main/PLAN.md).
