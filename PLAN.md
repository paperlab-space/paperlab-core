# PaperLab.space core plan

This repo (`paperlab-core`) holds the open-source evaluation engine behind [PaperLab.space](https://paperlab.space). It takes a spreadsheet, runs one or more LLMs over it, scores the outputs, and writes paper-ready results. It is a plain Python library plus a small CLI, published to PyPI as `paperlab-space` (`import paperlab_space`, command `paperlab-space`). The bare name `paperlab` belongs to an unrelated PyPI project, so the package name carries the `.space` brand, as the PRD intends.

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
- `openpyxl` + the standard `csv` module for input (no pandas: lighter installs and full control over cell-to-text conversion), `charset-normalizer` for legacy CSV encodings
- `pydantic` v2 for config models (serialisable, validated, clear error messages)
- `litellm` for one interface to every provider, imported lazily so `import paperlab_space` stays fast
- `asyncio` runner with an even-spacing rate limiter per model
- Own `{column}` placeholder syntax (column names may contain spaces; `{{`/`}}` for literal braces)
- `ruff` for lint/format, `pytest` for tests, `mypy --strict` for types
- `typer` for the CLI, included in the base install (an optional extra saved only ~8 MB next to LiteLLM, and tripped up non-programmers)

## Package layout

```
src/paperlab_space/
  data/        load.py (Excel/CSV -> Dataset), mapping.py (ColumnMapping, JSONL), validate.py
  prompt.py    PromptTemplate
  config.py    RunConfig, ModelSpec, MetricSpec (the single object describing a run)
  runner/      backend.py (LiteLLM + Backend protocol), ratelimit.py, runner.py, estimate.py
  metrics/     base.py (interface + registry), builtin.py, normalize.py
  results.py   RunResults, JSON save/load
  export/      excel.py, latex.py, repro.py
  _cli.py      the `paperlab` command
```

Central object: **`RunConfig`**, which holds data source, column mapping, prompt template, models (+ temperature, seed, max tokens, rate limit), and metrics. The same object drives a run and is written out as the reproducibility config, so a run is reproducible from its exported config alone.

## Public API

```python
from paperlab_space import ColumnMapping, RunConfig, estimate, export, load_dataset, run

data = load_dataset("data.xlsx", sheet="test")
mapping = ColumnMapping(input="sentence", expected="gold", id="row_id")
print(data.map(mapping).validate())  # empty cells, duplicate IDs, encoding issues

config = RunConfig(
    mapping=mapping,
    prompt="Translate to English: {sentence}",
    models=["openrouter/nvidia/nemotron-3-super-120b-a12b:free", "gemini/gemini-3.5-flash"],
    metrics=["exact_match", "token_f1"],
)
print(estimate(config, data))  # tokens, cost, time per model
results = run(config, data, on_progress=print)
export.write_all(results, "results/")  # results.xlsx, table.tex, run.yaml, results.json
```

The platform plugs in through: `LiteLLMBackend(api_keys=...)` (keys never enter configs or results), the `Backend` protocol, `on_progress`, `arun` for its own event loop, and `previous=` / `RunResults.load_json` to resume.

## Milestones

### M0: Skeleton ✅
- Package, CI, license, README, plan

### M1: Data in ✅
- Load `.xlsx` (multiple sheets), `.csv`, `.tsv`; UTF-8/BOM first, then encoding detection (tested with CP874 Thai, Shift-JIS, GBK)
- Unicode NFC normalisation; fixtures in Burmese, Thai, Chinese, Japanese, Korean
- Excel cell types converted to text predictably (`2.0` -> `2`, dates -> ISO); blank/duplicate headers fixed
- `ColumnMapping`: input column(s), context columns, expected column, row ID, multiple answers via delimiter; the expected column can never leak into the prompt
- Validation report: empty cells, empty/duplicate IDs, mojibake, invisible characters
- JSONL conversion + preview

### M2: Prompt + runner ✅
- `PromptTemplate` (user + optional system prompt) checked against mapped columns
- Async runner over LiteLLM; tested live against **OpenRouter** and **Gemini**
- Rate limits per model: OpenRouter `:free` 20/min, Gemini Flash 10/min, Gemini Pro 5/min; overridable
- Retries with exponential backoff + jitter, honouring `Retry-After`; a rejected API key or unknown model stops that model immediately instead of failing every row
- Resume: rows that succeeded in an earlier run are reused (pulled forward from v1.1)
- Pre-run estimate: tokens (script-aware heuristic), cost (LiteLLM price table), wall time, warnings (OpenRouter daily caps, Gemini free-tier data use)
- Works in scripts and Jupyter

### M3: Metrics ✅
- Metric interface (`compute(predictions, references) -> MetricResult`) with a registry so users can add their own
- `exact_match`, `accuracy` (optional label extraction from chatty answers), `f1` (macro/micro/weighted, per-class), `token_f1` (character tokens for Chinese, Japanese, Thai, Burmese)
- Configurable normalisation (NFKC, case, punctuation), recorded in the reproducibility config
- Multi-answer support; failed requests count as wrong unless `count_failed: false`
- Metrics are checked before any request is sent

### M4: Export ✅
- Excel: Summary (scores, failures, cost, tokens), Outputs (per-row outputs and scores), Config sheets; formula injection blocked
- LaTeX: booktabs table, best score bolded, escaped names, percent or raw
- Reproducibility YAML: core/LiteLLM/Python versions, provider-reported model versions, dataset SHA-256, full config (dataset path reduced to file name); loads back with `RunConfig.from_yaml`
- Full results JSON for resuming and sharing

### M5: CLI + first release (in progress)
- ✅ `paperlab-space init | validate | estimate | run [--resume] | metrics`
- ✅ `CITATION.cff`, release workflow (PyPI trusted publishing)
- ⬜ Configure the PyPI trusted publisher, then publish `0.1.0`
- ✅ Docs in `docs/` (Markdown + `nav.json`), published at paperlab.space/docs by the platform frontend
- ✅ Generated reference (`scripts/build_docs.py`): config file, metrics, and Python API from the source; checked in CI

### After 0.1 (matches the product roadmap)
- v1.1: separate answer-key file joined by row ID; BLEU, chrF, ROUGE (`sacrebleu`, `rouge-score`); Ollama and more providers documented and tested; Zawgyi detection for Burmese input
- v1.2: LLM-as-judge (with reference, and rubric-only with user rubric); significance tests and bootstrap confidence intervals
- v2: inter-annotator agreement metrics (Cohen's/Fleiss' kappa, Krippendorff's alpha)

## Quality bar

- Every public function typed (`mypy --strict`) and documented
- No network access in unit tests; `uv run pytest -m live` for real-provider smoke tests
- Deterministic outputs given the same config + seed where providers allow it
- Semantic versioning; the reproducibility config records the exact core version
