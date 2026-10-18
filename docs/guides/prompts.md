# Prompt templates

Write one prompt and fill it from each row with `{column}` placeholders.

## Placeholders

```yaml
prompt: "Translate to English: {sentence}"
```

Each `{name}` is replaced with that row's value from an `input` or `context` column. Column names can contain spaces: `{Source sentence}` works. An empty cell becomes an empty string.

The template must use at least one placeholder, or every row would get the same prompt. A placeholder that is not an input or context column is an error, reported before any request is sent. The `expected` column is never available, so the answer cannot leak into the prompt.

## Literal braces

Write `{{` and `}}` for literal braces, for example when asking for JSON:

```yaml
prompt: |-
  Extract the entities from the text as JSON, like {{"person": [...], "place": [...]}}.
  Text: {text}
```

## System prompts

Give the prompt as an object to add a system message. Placeholders work in both parts.

```yaml
prompt:
  system: You are a careful annotator for sentiment analysis research.
  user: |-
    Classify the sentiment of this {language} text as "positive" or "negative".
    Answer with one word.

    Text: {text}
```

## Multi-line prompts in YAML

Use `|-` for a block that keeps line breaks and drops the final newline. Indent every line of the block by the same amount.

## Tips for scoreable answers

- Ask for the answer format you will score, such as "Answer with one word" for classification or "Answer with the number only" for arithmetic.
- For classification, give the [`accuracy` or `f1` metric](metrics.md) a `labels` list, so a chatty answer like "The sentiment is positive." still counts as `positive`.
- Check a few outputs in `results.xlsx` before trusting a score. Formatting problems show up in the Outputs sheet straight away.

## From Python

```python
from paperlab_space import PromptTemplate

template = PromptTemplate("Translate to English: {sentence}", system="Be literal.")
print(template.variables)  # ['sentence']
print(template.render({"sentence": "สวัสดี"}))  # chat messages
```

A plain string works wherever a template is expected, such as `RunConfig(prompt="...")`.
