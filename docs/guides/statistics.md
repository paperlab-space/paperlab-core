# Statistics

Confidence intervals for every score, and significance tests for the differences between models.

A score on 100 examples is an estimate: on another 100 it would be somewhat different. Every run therefore reports how certain each score is, and whether the gaps between models are larger than chance would explain. Both are on by default.

## Confidence intervals

Each score gets a 95% percentile bootstrap interval. The rows are resampled with replacement 1,000 times, the metric is computed on each resample, and the middle 95% of those scores is the interval. This works for every metric, including corpus-level ones such as BLEU and F1, which are not an average of row scores.

The interval is in the Summary sheet (`<metric> 95% CI low` and `high`), in the table the CLI prints, and in Python:

```python
result = results.scores["gemini-3.5-flash"]["accuracy"]
print(f"{result.value:.3f} [{result.ci_low:.3f}, {result.ci_high:.3f}]")
```

To show it in the LaTeX table, pass `ci=True` to `latex_table` (see [Output files](outputs.md#tabletex)):

```latex
Gemini 3.5 Flash & \textbf{92.5} {\scriptsize[87.5, 97.5]} \\
```

## Significance tests

With two or more models, each pair is compared on each metric with a paired test on the same rows:

- `permutation` (default): approximate randomisation. Each row's two outputs are swapped between the models at random, 1,000 times. The p-value is the share of shuffles in which the difference is at least as large as the observed one. It makes no assumptions about how scores are distributed.
- `bootstrap`: the paired bootstrap. Both models are scored on the same 1,000 resamples of rows, which also gives a confidence interval for the difference.

Comparing many pairs makes some small p-values likely by chance, so p-values are corrected with Holm's method by default, separately for each metric. A difference is significant when its corrected p-value is below 0.05 (`1 - confidence`).

Results are in the Significance sheet of `results.xlsx`, after the score table the CLI prints, and in Python:

```python
for c in results.comparisons:
    print(c.metric, c.model_a, "vs", c.model_b, f"{c.difference:+.3f}", c.p_adjusted, c.significant)
```

`difference` is `model_a` minus `model_b`.

## Comparing against a baseline

To compare every model with one model, such as your baseline or the model you propose, set `baseline` to its label:

```yaml
statistics:
  baseline: gpt-5.4-mini
```

There are fewer comparisons than with every pair, so the correction is milder. In `table.tex`, scores that differ significantly from the baseline's get a dagger, and the caption says what it means:

```latex
Gemini 3.5 Flash & \textbf{92.5}$^\dagger$ \\
...
\caption{Results on 200 examples (\%). $^\dagger$: significantly different from gpt-5.4-mini
(paired permutation test, Holm-corrected, $p < 0.05$).}
```

Pass `mark_significant=False` to `latex_table` to leave the daggers out.

## Settings

```yaml
statistics:
  confidence: 0.95          # interval level; significance threshold is 1 - confidence
  intervals: true           # bootstrap intervals for every score
  significance: permutation # permutation | bootstrap | none
  baseline: null            # a model label, or null for every pair
  correction: holm          # holm | bonferroni | none
  resamples: 1000           # resamples or shuffles; more is slower but more precise
  seed: null                # defaults to the run seed
```

The resampling is seeded, so the same results give the same intervals and p-values. Every setting is recorded in `run.yaml`.

## In Python

`score(results)` adds intervals to the scores, and `stats.compare(results)` runs the tests. Both follow `results.config.statistics`:

```python
from paperlab_space import score, stats

results.scores = score(results)
results.comparisons = stats.compare(results)
```

## Reading the results

- An interval that is wide compared with the gaps between models means you need more rows to tell the models apart.
- "Not significant" does not mean the models are equally good. It means this data cannot show a difference.
- Rows a metric does not score, such as rows without an answer, are left out of the resampling too.
- For a paper, report the test, the correction, the number of resamples, and the number of rows. The caption of `table.tex` names the test and correction.

## References

- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
- Koehn, P. (2004). Statistical significance tests for machine translation evaluation. *EMNLP.*
- Riezler, S. and Maxwell, J. T. (2005). On some pitfalls in automatic evaluation and significance testing for MT. *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures.*
- Berg-Kirkpatrick, T., Burkett, D., and Klein, D. (2012). An empirical investigation of statistical significance in NLP. *EMNLP.*
- Dror, R., Baumer, G., Shlomov, S., and Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. *ACL.*
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics.*
