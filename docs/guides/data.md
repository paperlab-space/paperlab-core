# Preparing your data

How spreadsheets are read, how columns map to roles, and what validation checks.

## File formats

| Format | Notes |
| --- | --- |
| `.xlsx`, `.xlsm` | Any sheet, by name or zero-based index. The first sheet by default. |
| `.csv`, `.txt` | The separator is detected (comma, semicolon, tab, or `|`). |
| `.tsv` | Always tab-separated. |

Legacy `.xls` files are not supported; re-save them as `.xlsx`.

In `run.yaml`:

```yaml
dataset:
  path: data.xlsx
  sheet: test        # or 0 for the first sheet
```

## How cells are read

- The first non-empty row is the header. Fully empty rows are dropped, and so are trailing columns with neither a header nor data.
- Blank header cells become `column_1`, `column_2`, ... and duplicate names get a suffix (`answer`, `answer_2`).
- Every cell becomes text. Whole numbers lose their `.0` (`3`, not `3.0`), dates become ISO dates (`2026-01-31`), and booleans become `TRUE` or `FALSE`.
- Text is trimmed and Unicode-normalised (NFC). An empty cell stays empty rather than becoming `""`.
- For formulas, the value Excel last calculated is used. A file written by a script and never opened in Excel may have no calculated values, so those cells read as empty.
- CSV text is read as UTF-8, or as UTF-16 when the file starts with a byte-order mark. Otherwise the encoding is detected.

## Mapping columns to roles

```yaml
mapping:
  input: question               # one column or a list
  context: [subject, language]  # optional
  expected: answer              # optional; needed for scoring
  id: item_id                   # optional
  answer_delimiter: "|"         # optional
```

| Role | What it does |
| --- | --- |
| `input` | Columns available to the prompt template. At least one is required. |
| `context` | More columns available to the prompt, such as metadata. Only `input` columns are checked for empty cells. |
| `expected` | The answer key that outputs are scored against. It is never available to the prompt, so the answer cannot leak in, and it cannot also be an input or context column. |
| `id` | A stable row ID used in outputs and when resuming. Without one, rows are numbered from 1. |
| `answer_delimiter` | Splits the expected cell into several valid answers, e.g. `Paris|Paris, France`. An output matching any of them counts as correct. |

Use an `id` column if you might edit the spreadsheet between runs. Row numbers shift when rows are added or removed, and [resuming](resuming.md) matches rows by ID.

## Validation

`paperlab-space validate` and `paperlab-space run` check the mapped data first. A run refuses to start while there are errors. Warnings are shown but do not block it.

| Check | Level |
| --- | --- |
| No data rows | error |
| An input column is empty in every row | error |
| An input column is empty in some rows | warning |
| The expected column is empty in every row | error |
| The expected column is empty in some rows (they are not scored) | warning |
| An ID cell is empty, or two rows share an ID | error |
| Text that looks wrongly decoded, such as `Ã©` or `�` | warning |
| Invisible characters (zero-width space, BOM, non-breaking space) | warning |

Invisible characters are worth fixing: a zero-width space in the answer key makes an exact match fail even when the output looks identical.

```bash
paperlab-space validate data.xlsx --sheet test -i question -e answer --id item_id --preview 5
```

The preview shows the records as JSONL, the way they are passed on: the ID, the input and context fields, and the expected answers.

## From Python

```python
from paperlab_space import ColumnMapping, list_sheets, load_dataset

print(list_sheets("data.xlsx"))
data = load_dataset("data.xlsx", sheet="test")
print(data.columns, len(data))

mapped = data.map(ColumnMapping(input="question", expected="answer", id="item_id"))
report = mapped.validate()
print(report.ok, report.errors, report.warnings)
mapped.to_jsonl("data.jsonl")
```

To build a dataset without a file, use `Dataset.from_records([{"question": ..., "answer": ...}, ...])`.
