# Reproducibility and citing

What the exported config records, how to re-run it, and how to cite the package.

## The reproducibility config

Every run writes `run.yaml` to its results folder:

```yaml
# PaperLab reproducibility config. Re-run with: paperlab-space run run.yaml
paperlab_space: 0.1.5
created: '2026-10-18T09:12:44+00:00'
environment:
  python: 3.12.4
  paperlab-space: 0.1.5
  litellm: 1.79.0
data:
  file: data.csv
  sheet: null
  sha256: 3f5a...
  rows: 40
model_versions:
  Gemini 3.5 Flash:
  - gemini-3.5-flash-001
config:
  dataset:
    path: data.csv
    sheet: null
  mapping: ...
  prompt: ...
  models: ...
  metrics: ...
  seed: 42
```

| Field | What it records |
| --- | --- |
| `paperlab_space`, `environment` | The package, LiteLLM, and Python versions. |
| `data` | The data file name, sheet, row count, and the SHA-256 hash of the file. |
| `model_versions` | The exact model names the providers reported, which can be more specific than the ID you asked for. |
| `config` | The full run config, with every default written out. |

The dataset path is stored as a bare file name, so no local folder names end up in a paper's supplementary material. Runs on [PaperLab.space](https://paperlab.space) use the same engine, and the `run.yaml` they export works the same way.

## Re-running

Put the data file next to `run.yaml` and run it:

```bash
paperlab-space run run.yaml
```

The file loads like any run config (`RunConfig.from_yaml` in Python). To check that you have the same data, compare its hash with `data.sha256`:

```bash
shasum -a 256 data.csv
```

The config fixes everything on the evaluation side. Model outputs can still differ between runs, because providers update models behind the same ID and few guarantee deterministic outputs, even at temperature 0 with a seed. The recorded `model_versions` show whether the model changed.

## Reporting in a paper

Good practice for the evaluation section:

- Name each model by its full ID and, if reported, its version from `model_versions`.
- Give the prompt template, the temperature, and the run date (`created`).
- State how failed requests were scored (`count_failed`), and how many rows failed (in the Summary sheet).
- Share `run.yaml` with your supplementary material, so others can re-run it.

## Citing paperlab-space

If paperlab-space helps your research, please cite it:

```bibtex
@software{paperlab_space,
  author    = {{Kaung Nay Lin Khant}},
  title     = {paperlab-space: LLM evaluation on spreadsheet data},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23436738},
  url       = {https://doi.org/10.5281/zenodo.23436738}
}
```

This DOI always points to the latest version. To cite the exact version you used, pick it from the [Zenodo record](https://doi.org/10.5281/zenodo.23436738); the version is also recorded in your `run.yaml`. GitHub's "Cite this repository" button gives the same information from [CITATION.cff](../../CITATION.cff).
