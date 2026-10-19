# Metrics

The built-in metrics, how answers are normalised before comparison, and how to add your own.

## Built-in metrics

For short answers and labels:

| Metric | What it measures |
| --- | --- |
| `exact_match` | Share of rows where the output equals a reference answer after normalisation. |
| `accuracy` | Classification accuracy. With `labels`, a free-form answer that mentions exactly one label counts as that label. |
| `f1` | Classification precision, recall, and F1 over the gold labels, macro-averaged by default. |
| `token_f1` | SQuAD-style token-overlap F1 between output and reference, averaged over rows. |

For longer text, such as translations and summaries:

| Metric | What it measures |
| --- | --- |
| `bleu` | Corpus BLEU, computed with [sacreBLEU](https://github.com/mjpost/sacrebleu). |
| `chrf` | Character n-gram F-score (chrF) from sacreBLEU. `word_order: 2` gives chrF++. |
| `rouge1`, `rouge2`, `rougeL` | ROUGE-1, ROUGE-2, and ROUGE-L F-measure, averaged over rows. |
| `llm_judge` | A model grades each output against the reference, or against a rubric you write. See [LLM-as-judge](llm-judge.md). |

Every metric except `llm_judge` needs an [`expected` column](data.md#mapping-columns-to-roles). This is checked before any request is sent. `paperlab-space metrics` lists the metrics your installed version has.

Every score comes with a 95% bootstrap confidence interval, and models are compared with paired significance tests. See [Statistics](statistics.md).

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

## BLEU, chrF, and ROUGE

Scores are on a 0–1 scale like the other metrics, so `table.tex` shows BLEU 0.253 as 25.3.

`bleu` and `chrf` are corpus-level scores computed by sacreBLEU with its default settings, so they match numbers computed with the `sacrebleu` command. The sacreBLEU signature is saved with each score. Report it in your paper so others can compare:

```python
results.scores["gemini-3.5-flash"]["bleu"].info["signature"]
# 'nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0'
```

BLEU needs words. By default (`tokenize: auto`) it uses sacreBLEU's `13a` tokenizer. If the references contain a script written without spaces, such as Chinese, Japanese, Thai, or Burmese, it splits that text into characters first, as `token_f1` does. The choice depends on the references only, so every model is scored the same way. Set `tokenize` to use a specific sacreBLEU tokenizer instead, such as `zh`, `char`, or `flores200`.

chrF compares characters, so it works on every script without a tokenizer. For low-resource and non-Latin languages it is often the better choice.

The `rouge` metrics are case-insensitive, ignore punctuation, and use the same script-aware tokenizer as `token_f1`. On plain English text they give the same scores as Google's `rouge-score` package without stemming. Unlike `rouge-score`, they also work on non-Latin text. With several references, the best-matching one counts.

The Outputs sheet shows a row-level score for each of these metrics. For BLEU it is sentence BLEU, which is noisy for short sentences. Report the corpus score.

```yaml
metrics:
  - bleu
  - name: chrf
    label: chrf++
    word_order: 2
  - rougeL
```

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

The headline number for each model and metric is in the Summary sheet of `results.xlsx`, in `table.tex`, and in the table the CLI prints, with its confidence interval. Row-level scores are in the Outputs sheet. Some metrics record more numbers in `extra`:

| Metric | `extra` |
| --- | --- |
| `f1` | `precision`, `recall`, `f1_macro`, `f1_micro`, `f1_weighted`, and `f1[<class>]` for each class |
| `bleu` | `precision_1` to `precision_4`, `brevity_penalty` |
| `rouge1`, `rouge2`, `rougeL` | `precision`, `recall` |
| `llm_judge` | `mean_rating` on the judge's scale, `judge_failed`, `unparsed` |

Every metric also records `failed`, the number of failed requests. These are in `results.json` and in Python:

```python
result = results.scores["gemini-3.5-flash"]["f1"]
print(result.value, result.ci_low, result.ci_high, result.extra["f1[positive]"])
```

## Custom metrics

To add a metric of your own, subclass `Metric` and register it. See [Custom metrics](../extend/custom-metrics.md). For every built-in metric's options, see the [metrics reference](../reference/metrics.md).
