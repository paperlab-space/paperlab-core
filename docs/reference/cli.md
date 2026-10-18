# CLI reference

Every `paperlab-space` command and option.

```bash
paperlab-space --version
paperlab-space COMMAND --help
```

Every command exits with status 1 and an `Error:` message when something is wrong.

## init

Write a starter run config for a data file, guessing the column roles.

```bash
paperlab-space init DATA [--output run.yaml] [--sheet NAME] [--force]
```

| Option | Default | Description |
| --- | --- | --- |
| `DATA` | | Excel or CSV file to evaluate on. |
| `--output`, `-o` | `run.yaml` | Config file to write. |
| `--sheet` | first sheet | Sheet name for Excel files. |
| `--force` | off | Overwrite an existing config. |

The guesses are based on column names:

- **ID:** a column named `id`, `row_id`, `index`, `no`, `no.`, or `#` (any case).
- **Expected:** the first column whose name contains `expected`, `gold`, `answer`, `label`, `reference`, `target`, `output`, or `ref`.
- **Input:** the first remaining column.

The config starts with the prompt `{input column}`, two free models, `exact_match` if an expected column was found, and `seed: 42`. A comment at the top lists the columns and metrics. The dataset path is written relative to the config.

## validate

Check a data file and preview the records as JSONL.

```bash
paperlab-space validate DATA [--sheet NAME] -i COLUMN [-i COLUMN ...] [-e COLUMN] [--id COLUMN]
```

| Option | Default | Description |
| --- | --- | --- |
| `DATA` | | Excel or CSV file. |
| `--sheet` | first sheet | Sheet name for Excel files. |
| `--input`, `-i` | | Input column. Repeat for several. |
| `--expected`, `-e` | | Expected-answer column. |
| `--id` | | Row ID column. |
| `--delimiter` | | Separator for several valid answers in one cell. |
| `--preview` | `3` | Number of records to print as JSONL. `0` turns it off. |

Without `--input`, `validate` only prints the row count, the sheets, and the columns. With a mapping, it runs the [validation checks](../guides/data.md#validation) and exits with status 1 if there are errors.

## estimate

Estimate tokens, cost, and time for a run without calling any model.

```bash
paperlab-space estimate CONFIG
```

The data is loaded and validated first. See [Cost estimates](../guides/models.md#cost-estimates) for how the numbers are worked out.

## run

Run an evaluation and write the results.

```bash
paperlab-space run CONFIG [--out FOLDER] [--yes] [--resume RESULTS_JSON]
```

| Option | Default | Description |
| --- | --- | --- |
| `CONFIG` | | Run config (YAML). |
| `--out`, `-o` | `results-<config name>/` next to the config | Output folder. |
| `--yes`, `-y` | off | Skip the confirmation after the estimate. |
| `--resume` | | `results.json` of an earlier or stopped run; only rows without an output are run. See [Resuming a run](../guides/resuming.md). |

The run prints the estimate, asks to start, and shows a progress bar per model. At the end, it prints a score table and writes `results.xlsx`, `table.tex`, `run.yaml`, and `results.json` to the output folder. See [Output files](../guides/outputs.md).

While the run goes, finished rows are saved to `results.json` in the output folder every few seconds. If you stop it with Ctrl+C, it saves the rows finished so far, prints the `--resume` command to continue, and exits with status 130.

## metrics

List the available metrics, whether each needs an answer key, and what it measures.

```bash
paperlab-space metrics
```

See [Metrics](../guides/metrics.md) for their options.
