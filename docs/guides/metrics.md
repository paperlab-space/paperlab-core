# Metrics

The built-in metrics, how answers are normalised before comparison, and how to add your own.

## Built-in metrics

| Metric | What it measures |
| --- | --- |
| `exact_match` | Share of rows where the output equals a reference answer after normalisation. |
| `accuracy` | Classification accuracy. With `labels`, a free-form answer that mentions exactly one label counts as that label. |
| `f1` | Classification precision, recall, and F1 over the gold labels, macro-averaged by default. |
| `token_f1` | SQuAD-style token-overlap F1 between output and reference, averaged over rows. |

All of them need an [`expected` column](data.md#mapping-columns-to-roles). This is checked before any request is sent. `paperlab-space metrics` lists the metrics your installed version has.

## Configuring metrics

A metric is a name, or a name plus options:

```yaml
metrics:
  - exact_match                     # the defaults
  - name: exact_match
    label: exact_match_strict       # the same metric again with other options
    ignore_case: false
    trim_punctuation: false
  - name: accuracy
    labels: [positive, negative, neutral]
  - name: f1
    labels: [positive, negative, neutral]
    average: weighted               # macro | micro | weighted
  - name: token_f1
    remove_articles: true
```

`label` names the metric's column in the outputs. Labels must be distinct, so set one to use a metric twice. An unknown option is an error that lists the valid ones.

## Normalisation

Before comparing, outputs and references are normalised the same way. The defaults suit short answers and labels:

| Option | Default | Effect |
| --- | --- | --- |
| `nfkc` | `true` | Unicode NFKC, so full-width `ＡＢＣ` equals `ABC`. |
| `ignore_case` | `true` | Case-insensitive comparison. |
| `trim_punctuation` | `true` | Strip punctuation and quotes from the start and end only, so `"Positive."` matches `positive`. |
| `ignore_punctuation` | `false` | Remove all punctuation, including inside the text. |
| `count_failed` | `true` | Score failed requests as wrong. Set `false` to leave them out. |

Whitespace is always collapsed. The characters `+ - % $ # @ &` are never treated as punctuation, so `-5` and `5` stay different.

`token_f1` always trims punctuation, removes all punctuation by default (`ignore_punctuation: true`), and adds `remove_articles` (default `false`), which drops English *a*, *an*, and *the* as in SQuAD evaluation.

## Label extraction

Models often answer "The sentiment is positive." instead of `positive`. With `labels`, `accuracy` and `f1` map such answers to a label:

1. If the normalised answer equals a label, that label is used.
2. Otherwise, if exactly one label appears in the answer, that label is used. The longest match wins, so "very positive" beats "positive" when both are labels.
3. Otherwise the answer is left as it is, and it matches no label.

English labels must appear as whole words. Labels in scripts without spaces, such as Thai or Japanese, match anywhere in the answer.

## Rows that are not scored

- Rows with an empty `expected` cell are skipped by every metric.
- Failed requests count as wrong unless `count_failed: false`.
- With [`answer_delimiter`](data.md#mapping-columns-to-roles), a row has several valid answers. An output matching any of them is correct, and `token_f1` takes the best-matching one.

## Where scores appear

The headline number for each model and metric is in the Summary sheet of `results.xlsx`, in `table.tex`, and in the table the CLI prints. Row-level scores are in the Outputs sheet. `f1` also records precision, recall, the macro, micro, and weighted F1, and the F1 of each class. These are in `results.json` and in Python:

```python
result = results.scores["gemini-3.5-flash"]["f1"]
print(result.value, result.n_scored, result.extra["precision"], result.extra["f1[positive]"])
```

## Custom metrics

To add a metric of your own, subclass `Metric` and register it. See [Custom metrics](../extend/custom-metrics.md). For every built-in metric's options, see the [metrics reference](../reference/metrics.md).
