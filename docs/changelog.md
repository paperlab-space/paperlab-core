# Changelog

What changed in each release of paperlab-space.

Versions follow [semantic versioning](https://semver.org). Each release is on [PyPI](https://pypi.org/project/paperlab-space/#history) and has its own DOI on [Zenodo](https://doi.org/10.5281/zenodo.23436738). The version you used is recorded in every `run.yaml`.

## 1.0.0 (2026-10-19)

The first stable release. From here on, the Python API, the `run.yaml` and `results.json` formats, and the `paperlab-space` command follow semantic versioning: configs and results from 1.0 keep working in every 1.x release. See [Stability](index.md#stability).

- Text metrics: `bleu` and `chrf` (chrF and chrF++) computed with sacreBLEU, with the sacreBLEU signature saved for your paper, and `rouge1`, `rouge2`, and `rougeL`. BLEU and ROUGE split Chinese, Japanese, Thai, Burmese, and other scripts without spaces into characters. See [Metrics](guides/metrics.md#bleu-chrf-and-rouge).
- `llm_judge`: a model grades each output against the reference answer, or against a rubric you write. Judge requests get the same rate limits, retries, resume, and estimates as model requests, and the judge's reasoning is saved for every row. See [LLM-as-judge](guides/llm-judge.md).
- Every score has a 95% bootstrap confidence interval, in the Summary sheet, the CLI, and optionally `table.tex` (`ci=True`). See [Statistics](guides/statistics.md).
- Paired significance tests between models (approximate randomisation by default, or paired bootstrap), with Holm correction. Results are in a new Significance sheet and in `results.comparisons`. With `statistics.baseline`, `table.tex` marks significant differences with a dagger.
- A new `statistics` section in the run config. Existing configs work unchanged and get intervals and tests by default; set `intervals: false` and `significance: none` to turn them off.
- `MetricResult` has `ci_low`, `ci_high`, and `info`. `RunResults` has `comparisons` and `judgments`. `results.json` files from 0.1 still load.
- `Progress.metric` and the `on_judge_progress` callback report judge progress. `Estimate.judges` lists judge requests, and the estimate's cost and time include them.
- Custom metrics can set `averages_rows`, and override `scorer` for fast resampling and `check` to validate the data before a run. See [Custom metrics](extend/custom-metrics.md).
- New dependencies: `sacrebleu` and `numpy`.

## 0.1.6 (2026-10-18)

- Runs save finished rows to `results.json` as they go, and on Ctrl+C, so a stopped run can continue with `--resume`. In Python, pass `checkpoint` to `run` or `arun`.
- Documentation at [paperlab.space/docs](https://paperlab.space/docs), with a reference generated from the source.
- Docstrings for every public field, shown in `help()` and in editors.

## 0.1.5 (2026-10-18)

- A key with no quota at all for a model, such as a Pro model on Gemini's free tier, now stops that model straight away instead of retrying every row.
- Progress updates carry the row that just finished (`Progress.row`), so callers can save partial results.

## 0.1.4 (2026-10-17)

- Errors show the provider's own message instead of LiteLLM's wrapped JSON.
- `results.json` records the dataset by file name only, so sharing it does not reveal local folder names.

## 0.1.3 (2026-10-17)

- `list_sheets` raises `DataError` for a corrupt Excel file, like `load_dataset`.
- Zenodo DOI in the citation file and README.

## 0.1.2 (2026-10-17)

- Author and ORCID in the citation metadata.

## 0.1.1 (2026-10-17)

- The `paperlab-space` command is part of the base install. `pip install 'paperlab-space[cli]'` still works.

## 0.1.0 (2026-10-17)

First release.

- Loading `.xlsx`, `.csv`, and `.tsv` files, column mapping, validation, and JSONL preview.
- Prompt templates and the YAML run config.
- An async runner with per-model rate limits, retries, resume, and cost and time estimates, for every provider LiteLLM supports.
- Metrics: exact match, accuracy with label extraction, classification F1, and token F1, with tokenisation for scripts written without spaces.
- Exports: Excel, a booktabs LaTeX table, the reproducibility config, and `results.json`.
- The `paperlab-space` command: `init`, `validate`, `estimate`, `run`, and `metrics`.
