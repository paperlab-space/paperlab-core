# Changelog

What changed in each release of paperlab-space.

Versions follow [semantic versioning](https://semver.org). Each release is on [PyPI](https://pypi.org/project/paperlab-space/#history) and has its own DOI on [Zenodo](https://doi.org/10.5281/zenodo.23436738). The version you used is recorded in every `run.yaml`.

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
