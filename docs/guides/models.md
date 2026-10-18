# Models and API keys

Pick models by their LiteLLM ID, bring your own keys, and tune rate limits and retries.

## Model IDs

Models are addressed as `provider/model`, the same IDs [LiteLLM](https://docs.litellm.ai/docs/providers) uses:

```yaml
models:
  - openrouter/nvidia/nemotron-3-super-120b-a12b:free
  - gemini/gemini-3.5-flash
  - openai/gpt-4o-mini
  - anthropic/claude-haiku-4-5
```

Every model runs on every row, and models run in parallel. An ID without a provider prefix is rejected.

## API keys

Keys are read from each provider's usual environment variable:

| Provider | Variable | Free tier |
| --- | --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` | Yes, for models ending in `:free` |
| Gemini (Google AI Studio) | `GEMINI_API_KEY` | Yes, rate-limited |
| OpenAI | `OPENAI_API_KEY` | No |
| Anthropic | `ANTHROPIC_API_KEY` | No |

Keys are never written to `run.yaml`, `results.json`, or any other output. To pass keys in Python, see [Pass API keys directly](../quickstart-python.md#pass-api-keys-directly).

A rejected key, an unknown model ID, or a key with no quota for a model stops that model straight away, with the provider's own message. Its remaining rows are marked as failed, and the other models carry on.

## Model options

A model can be a plain ID, or an object with options:

```yaml
models:
  - id: gemini/gemini-3.5-flash
    label: Gemini Flash (t=0.7)   # name in tables; defaults to the ID without the provider
    temperature: 0.7              # default 0
    max_tokens: 256
    top_p: 0.9
    seed: 7                       # overrides the run-level seed
    requests_per_minute: 60       # overrides the default rate limit
    concurrency: 8                # overrides the run-level concurrency
    params:                       # passed through to LiteLLM as-is
      reasoning_effort: low
```

To compare the same model at two settings, give each entry its own `label`; labels must be distinct.

The run-level `seed` is used for every model that does not set its own. Providers that do not support seeds ignore it, and a seed does not guarantee identical outputs even where it is supported.

## Rate limits

Requests to each model are spaced evenly to stay under a per-minute limit. The defaults are conservative, for free tiers:

| Model | Default limit |
| --- | --- |
| OpenRouter models ending in `:free` | 20 per minute |
| Gemini models with `pro` in the ID | 5 per minute |
| Other Gemini models | 10 per minute |
| Everything else | No limit |

With a paid key, raise the limit with `requests_per_minute`. `concurrency` (default 4, set for the whole run or per model) caps how many requests to one model are in flight at once.

OpenRouter also limits free models per day: as few as 50 requests a day on accounts that have never bought credits. `estimate` warns when a run is larger than that. Rows over the limit fail, and you can [resume](resuming.md) the next day.

## Retries and timeouts

Rate-limit errors, timeouts, and server errors are retried with exponential backoff and jitter. When a provider sends a `Retry-After` header, the wait follows it and every request to that model pauses.

```yaml
retry:
  max_attempts: 5   # including the first try
  base_delay: 2     # seconds before the first retry; doubles each time
  max_delay: 60
  timeout: 120      # seconds to wait for one response
concurrency: 4
```

Other errors, such as a request the provider rejects, fail that row without a retry. Failed rows are recorded with their error message in the outputs.

## Cost estimates

`paperlab-space estimate run.yaml` (or `estimate(config, data)` in Python) prints, per model, the request count, the input and output tokens, the cost, and the time, without sending anything.

- Prices come from LiteLLM's price table. OpenRouter `:free` models and Ollama count as free. A model with no known price shows "unknown cost".
- Token counts are a script-aware approximation, not the model's own tokenizer. Treat them as ballpark figures.
- Output length is guessed from the expected answers, or from `max_tokens`.
- Reasoning models get a warning: hidden thinking tokens are billed as output and can make the real cost several times higher. Cap them with `max_tokens` or a provider option such as `reasoning_effort` under `params`.
- Gemini models get a warning that on the free tier, Google may use submitted data to improve its products. Check this against your ethics approval.

## Local models with Ollama

Ollama models go through LiteLLM like any other provider, for example `ollama/llama3.2`, and are counted as free. Nothing leaves your machine. In Python, point at a non-default server with `LiteLLMBackend(api_bases={"ollama": "http://localhost:11434"})`. Ollama is not yet covered by the test suite.
