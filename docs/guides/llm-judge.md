# LLM-as-judge

Have a model grade each output, against the reference answer or against a rubric you write.

Use a judge when the right answer can be worded many ways, as in open-ended questions, explanations, and summaries, where exact match is too strict and BLEU or ROUGE miss the meaning. Judges have known biases, so treat their scores as an estimate. Check a sample by hand before you rely on them (see [Good practice](#good-practice)).

## With a reference answer

If an `expected` column is mapped, the judge sees the task, the reference answer, and the model's response, and rates how well the response agrees with the reference:

```yaml
metrics:
  - exact_match
  - name: llm_judge
    model: openai/gpt-5.4-mini      # the judge: any model ID, as in `models`
```

The default rubric asks for a score from 1 to 5, where 5 means fully correct and complete and 1 means wrong or unrelated. Differences in wording, length, and formatting are ignored.

## With a rubric only

Without an answer key, or with `reference: false`, the judge grades by your rubric alone:

```yaml
metrics:
  - name: llm_judge
    label: helpfulness
    model: gemini/gemini-3.5-pro
    reference: false
    rubric: >-
      Rate how helpful the answer is to a first-year student. {max_score}: clear, correct,
      and complete. {min_score}: wrong, confusing, or off-topic.
    min_score: 1
    max_score: 10
```

`{min_score}` and `{max_score}` in the rubric are replaced with the scale. A rubric is required when the judge sees no reference.

To judge more than one thing, add the metric again with its own `label`, such as one entry for correctness and one for fluency.

## What the judge sees

For each row, the judge gets a system prompt asking for impartial grading and a message like this:

```text
Grade the response to the task below.

[Task]
<the prompt the model received>

[Reference answer]
<the expected answer>

[Response]
<the model's output>

[Rubric]
<the rubric>

Explain your judgement in a few sentences, then end with a line of the form "Score: N",
where N is a whole number from 1 to 5.
```

The last `Score: N` line in the response is the rating. To write your own prompt, set `prompt`, using the placeholders `{input}`, `{output}`, `{reference}`, `{rubric}`, `{min_score}`, and `{max_score}`. Ask for a `Score: N` line at the end. `system` replaces the system prompt; `null` sends none.

The judge model takes the same options as an entry in `models`, so you can set its temperature, `max_tokens`, rate limit, or provider parameters:

```yaml
  - name: llm_judge
    model:
      id: openrouter/qwen/qwen3.5-72b-instruct
      temperature: 0
      requests_per_minute: 30
```

It uses the run's `seed` unless it sets its own.

## Scores

Ratings are mapped onto a 0–1 scale, with the lowest rating 0 and the highest 1, so the judge fits in the same table as the other metrics. The mean rating on the judge's own scale is in `extra["mean_rating"]`.

| Row | Score |
| --- | --- |
| The model's request failed | The lowest rating (0), or left out with `count_failed: false` |
| The judge's request failed | Left out, counted in `judge_failed` |
| The judge's response has no score in range | Left out, counted in `unparsed` |
| No reference answer, when the judge uses references | Left out |

The judge's full response for each row is in the Outputs sheet (`<model> | <metric> judge`) and in `results.json` under `judgments`, so you can check its reasoning.

## Cost and time

Judging starts after every model has finished. It sends one request per row per model, and the estimate shown before a run includes them. The CLI shows a progress bar for each judge.

Judge responses are reused when you resume a run, as long as the judged output and the judge's settings are unchanged. To judge an earlier run without sending its model requests again, add the `llm_judge` metric to its config and resume:

```bash
paperlab-space run run.yaml --resume results-run/results.json
```

## Good practice

- **Use a judge from another model family.** Judges tend to prefer their own answers. The estimate warns when the judge is also one of the models being evaluated.
- **Check a sample by hand.** Read the judge's reasoning for 20 to 50 rows in the Outputs sheet, and report how often you agree with it.
- **Keep temperature at 0** for more repeatable grades, and report the judge model and its version. Both are recorded in `run.yaml`.
- **Report the rubric.** It is part of the config, so it is in `run.yaml` too.
