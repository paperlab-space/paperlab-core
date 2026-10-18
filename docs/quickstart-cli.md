# Quickstart (command line)

Run your first evaluation from a spreadsheet in five commands.

## 1. Get API keys

You bring your own keys. Both of these have free tiers:

```bash
export OPENROUTER_API_KEY=sk-or-...   # https://openrouter.ai
export GEMINI_API_KEY=...             # https://aistudio.google.com
```

Other providers read their usual variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). See [Models and API keys](guides/models.md).

## 2. Write a starter config

```bash
paperlab-space init my_data.xlsx
```

This writes `run.yaml`, guessing which columns hold the input, the expected answer, and the row ID from their names. Pass `--sheet` to pick an Excel sheet.

## 3. Edit the config

Open `run.yaml` and check the mapping, then write your prompt. A finished config looks like this (from [examples/sentiment](../examples/sentiment)):

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
  - id: gemini/gemini-3.5-flash
    temperature: 0
metrics:
  - name: accuracy
    labels: [positive, negative]
  - f1
seed: 42
```

`{text}` and `{language}` are filled from each row. See [Prompt templates](guides/prompts.md).

## 4. Check the data and the cost

```bash
paperlab-space validate my_data.xlsx -i text -e label --id id
paperlab-space estimate run.yaml
```

`validate` reports empty cells, duplicate IDs, and encoding problems, and previews the first rows as JSONL. `estimate` prints the tokens, cost, and time for each model without sending any requests.

## 5. Run it

```bash
paperlab-space run run.yaml
```

The run shows the estimate again and asks for confirmation (`--yes` skips it). When it finishes, it prints a score table and writes the results to `results-run/`:

| File | What it is |
| --- | --- |
| `results.xlsx` | Summary, per-row outputs and scores, and the config |
| `table.tex` | A booktabs table for your paper |
| `run.yaml` | The reproducibility config |
| `results.json` | Everything, for `--resume` and sharing |

See [Output files](guides/outputs.md) for what each contains.

If some requests failed (a free model's daily limit, for example), the command prints a `--resume` line that retries only those rows. See [Resuming a run](guides/resuming.md).

Free models change often. If a model ID no longer exists, the run stops that model early with a clear message. Browse current free models at [openrouter.ai/models](https://openrouter.ai/models?q=free).
