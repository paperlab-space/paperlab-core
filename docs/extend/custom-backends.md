# Custom backends

Send requests somewhere other than LiteLLM, or nowhere at all for testing.

## Do you need one?

Probably not for a hosted API. LiteLLM supports [most providers](https://docs.litellm.ai/docs/providers), and an OpenAI-compatible server (vLLM, LM Studio, llama.cpp) works by pointing `LiteLLMBackend` at it:

```python
from paperlab_space import LiteLLMBackend, run

backend = LiteLLMBackend(
    api_bases={"openai": "http://localhost:8000/v1"}, api_keys={"openai": "none"}
)
results = run(config, data, backend=backend)  # models: [openai/my-local-model]
```

Write a backend when the model is not behind an API, such as a Hugging Face model loaded in the same process, or to test a pipeline without sending requests.

## The interface

A backend is any object with these two methods:

```python
class MyBackend:
    async def complete(self, model, messages, *, seed, timeout):
        """Return one Completion."""

    def price(self, model):
        """USD per input token and per output token, or None if unknown."""
```

- `model` is the `ModelSpec` from the config, with `id`, `temperature`, `max_tokens`, `top_p`, and `params`.
- `messages` is a list of chat messages: `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`. The system message is there only if the prompt has one.
- `seed` is the model's seed, or the run's, or `None`. `timeout` is in seconds.
- `complete` returns a `Completion` with the answer `text` and, if known, `input_tokens`, `output_tokens`, `cost_usd`, and `model_version`.

The runner handles rate limits, concurrency, retries, and progress. Your backend only answers one request at a time.

## Errors

Raise these from `complete` so the runner can react:

| Exception | When | What the runner does |
| --- | --- | --- |
| `RetryableError(message, retry_after=None)` | A transient problem: rate limit, timeout, server error. | Retries with backoff, honouring `retry_after` (seconds). |
| `ProviderError(message)` | This request cannot succeed. | Marks the row as failed. |
| `ProviderError(message, fatal=True)` | No request to this model can succeed, such as a bad key. | Stops the model and marks its remaining rows as failed. |

Any other exception marks the row as failed with the exception's message, so a bug in a backend does not stop the whole run.

## Example: a test backend

A backend that answers instantly is useful to check data, prompts, metrics, and exports before you spend anything:

```python
from paperlab_space import RunConfig, export, load_dataset, run
from paperlab_space.runner import Completion


class EchoBackend:
    """Answers with the last line of the prompt."""

    async def complete(self, model, messages, *, seed, timeout):
        return Completion(text=messages[-1]["content"].splitlines()[-1])

    def price(self, model):
        return (0.0, 0.0)


config = RunConfig.from_yaml("run.yaml")
results = run(config, backend=EchoBackend())
export.write_all(results, "dry-run/")
```

## Example: a local Hugging Face model

```python
import asyncio

from transformers import pipeline

from paperlab_space.runner import Completion


class TransformersBackend:
    """Runs models in this process. Use IDs like hf/Qwen/Qwen2.5-0.5B-Instruct."""

    def __init__(self):
        self._pipes = {}

    async def complete(self, model, messages, *, seed, timeout):
        name = model.id.split("/", 1)[1]
        if name not in self._pipes:
            self._pipes[name] = pipeline("text-generation", model=name)
        options = {"max_new_tokens": model.max_tokens or 256}
        if model.temperature > 0:
            options.update(do_sample=True, temperature=model.temperature)
        output = await asyncio.to_thread(self._pipes[name], messages, **options)
        return Completion(text=output[0]["generated_text"][-1]["content"])

    def price(self, model):
        return (0.0, 0.0)
```

Set `concurrency: 1` for the model, since one process runs one generation at a time. The model loads on its first request.

## Estimates with a custom backend

Pass the backend to `estimate` so it uses your prices:

```python
from paperlab_space import estimate

print(estimate(config, data, backend=EchoBackend()))
```
