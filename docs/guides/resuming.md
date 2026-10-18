# Resuming a run

Continue a stopped run, or retry the rows that failed, without paying for the rest again.

## When to resume

- **Some rows failed.** A free model hit its daily limit, a provider had an outage, or retries ran out. The run still finishes and writes its results, with those rows marked as failed, then prints how to retry them:

  ```text
  12 request(s) failed. Re-run with --resume results-run/results.json to retry only those rows.
  ```

- **The run was stopped.** While a run goes, finished rows are saved to `results.json` in the results folder every few seconds. If you press Ctrl+C, the rows finished so far are saved, and the command prints how to continue:

  ```text
  Stopped. Finished rows are saved in results-run/results.json. Continue with:
    paperlab-space run run.yaml --resume results-run/results.json
  ```

  If the process is killed outright or the machine loses power, you lose at most the last few seconds of rows.

## How to resume

```bash
paperlab-space run run.yaml --resume results-run/results.json
```

Rows that already have an output are reused. Only rows that failed or never ran are sent, and the estimate shown before the run still counts every row. The results folder is then rewritten with the combined results and fresh scores.

- **Same model settings.** A model's options (temperature, `max_tokens`, and so on) must not change between the runs, or its outputs would be mixed. Resuming stops with an error if they did.
- **New models.** A model added to the config since the last run gets all of its rows run.
- **Matched by ID.** Rows are matched by their ID. Use an [`id` column](data.md#mapping-columns-to-roles) if you might edit the data in between, because row numbers shift when rows are added or removed.

## From Python

Pass `checkpoint` to save rows as the run goes, and `previous` to continue from a saved file:

```python
from paperlab_space import RunResults, run

results = run(config, data, checkpoint="results/results.json")

# After an interruption:
previous = RunResults.load_json("results/results.json")
results = run(config, data, previous=previous, checkpoint="results/results.json")
```

When the run finishes, the full results replace the checkpoint. A checkpoint from a stopped run holds only the finished rows: use it to resume, not with `export`.
