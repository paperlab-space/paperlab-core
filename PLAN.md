# paperlab-core plan

`paperlab-core` is the open-source evaluation engine behind [PaperLab.space](https://paperlab.space). It takes a spreadsheet, runs one or more LLMs over it, scores the outputs, and writes paper-ready results. It is a plain Python library (plus a small CLI) and is published to PyPI.

## Scope and boundary

Core is pure **input → eval → output**. It never imports web code: no database, auth, HTTP server, payments, or job-queue infrastructure. Anything stateful across sessions belongs to the platform.

| In core | Not in core (platform) |
| --- | --- |
| Excel/CSV loading, column mapping, validation, JSONL conversion | File upload, storage, user accounts |
| Prompt templating | Prompt editor UI |
| Model runner: LiteLLM wrapper, per-provider rate limiting, retries, checkpoint/resume to a local file | Distributed job queue, workers, progress over websockets |
| Metrics | Credits, billing, cost markup |
| Excel / LaTeX / reproducibility-config export | Run history, sharing |
| Cost and time estimates (from token counts and public prices) | Managed API keys, encrypted key storage |

The platform calls core through its public API only. If the platform needs a hook (e.g. a progress callback), core exposes a generic one.

## Tech choices

- Python ≥ 3.10, `src/` layout, built with hatchling, managed with `uv`
- `pandas` + `openpyxl` for Excel/CSV
- `pydantic` v2 for all config and data models (serialisable, validated)
- `litellm` for one interface to every provider
- `asyncio` runner with a token-bucket limiter keyed by `(provider, model)`
- `jinja2`-free templating: Python `str.format`-style `{column}` placeholders with clear errors for missing columns
- `ruff` for lint/format, `pytest` for tests, `mypy` for types
- `typer` for the CLI (optional extra: `pip install paperlab-core[cli]`)

## Package layout

```
src/paperlab_core/
  data/        load.py, mapping.py, validate.py, jsonl.py
  prompt.py    PromptTemplate
  runner/      runner.py, ratelimit.py, retry.py, checkpoint.py, estimate.py
  metrics/     base.py, exact_match.py, classification.py (accuracy, P/R/F1)
  export/      excel.py, latex.py, repro.py
  config.py    RunConfig (the single object describing a run)
  cli.py       `paperlab run config.yaml`
```

Central object: **`RunConfig`**, which holds data source, column mapping, prompt template, models (+ temperature, seed, max tokens), and metrics. The same object drives a run and is written out as the reproducibility config, so a run is reproducible from its exported config alone.

## Public API sketch

```python
from paperlab_core import load_dataset, ColumnMapping, PromptTemplate, RunConfig, run, export

ds = load_dataset("data.xlsx", sheet="test")
ds = ds.map(ColumnMapping(input="sentence", expected="gold", id="row_id"))
report = ds.validate()  # empty cells, duplicate IDs, encoding issues

cfg = RunConfig(
    prompt=PromptTemplate("Translate to English: {sentence}"),
    models=["openrouter/meta-llama/llama-3.3-70b-instruct:free", "gemini/gemini-2.5-flash"],
    metrics=["exact_match", "accuracy", "f1"],
)
print(cfg.estimate(ds))  # tokens, cost, time per model
results = run(ds, cfg, on_progress=print)
export.excel(results, "results.xlsx")
export.latex(results, "table.tex")
export.repro(cfg, "run.yaml")
```

## Milestones

Each milestone ends with a tagged release once tests pass.

### M0: Skeleton (this commit)
- Package, CI (ruff + pytest on 3.10–3.13), license, README, plan

### M1: Data in (`0.1.0`)
- Load `.xlsx` (multiple sheets) and `.csv`, with encoding detection and Unicode-safe handling for non-Latin scripts (test fixtures in Burmese, Thai, Chinese, Japanese, Korean)
- `ColumnMapping`: input column(s), extra context columns, expected-output column, optional row ID
- Multiple valid answers per row via delimiter
- Validation report: empty cells, duplicate IDs, mixed types, encoding problems
- JSONL conversion + preview (`ds.preview(n=5)`)

### M2: Prompt + runner (`0.2.0`)
- `PromptTemplate` with variable checking against mapped columns
- Async runner over LiteLLM; providers for MVP: **OpenRouter** and **Gemini**
- Per-`(provider, model)` rate limiter with sensible defaults (OpenRouter `:free` ≈ 20 req/min, Gemini free ≈ 5–10 req/min), overridable
- Retries with exponential backoff + jitter on 429/5xx/timeouts
- Pre-run estimate: tokens, cost, wall time
- Progress callback hook
- Tests use a fake LiteLLM backend; no network in CI

### M3: Metrics (`0.3.0`)
- Metric interface: `score_row(output, expected) -> float` + `aggregate(scores) -> dict`
- Exact match (with normalisation options: case, whitespace, punctuation), accuracy, precision/recall/F1 (macro and micro)
- Multi-answer support (best match across valid answers)

### M4: Export (`0.4.0`)
- Excel: per-row inputs, each model's output, per-row scores; summary sheet
- LaTeX: summary comparison table (booktabs, best score bolded), ready to `\input{}`
- Reproducibility config (YAML): core version, model IDs, prompt, temperature, seed, metric settings, dataset hash

### M5: CLI + first public release (`1.0.0`)
- `paperlab run run.yaml`, `paperlab validate data.xlsx`, `paperlab estimate run.yaml`
- Docs site (mkdocs-material) with a 5-minute quickstart
- PyPI publishing via GitHub Actions trusted publishing
- `CITATION.cff` so researchers can cite the tool

### After 1.0 (matches the product roadmap)
- v1.1: separate answer-key file joined by row ID; BLEU, chrF, ROUGE (`sacrebleu`, `rouge-score`); more providers incl. Ollama; checkpoint/resume after failure
- v1.2: LLM-as-judge (with reference, and rubric-only with user rubric); significance tests and bootstrap confidence intervals
- v2: inter-annotator agreement metrics (Cohen's/Fleiss' kappa, Krippendorff's alpha)

## Quality bar

- Every public function typed and documented
- No network access in unit tests; an opt-in `-m live` marker for real-provider smoke tests
- Deterministic outputs given the same config + seed where providers allow it
- Semantic versioning; the reproducibility config records the exact core version
