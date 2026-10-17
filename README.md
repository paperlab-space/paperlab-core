# paperlab-core

Open-source engine for evaluating LLMs on your own spreadsheet data and getting paper-ready results.

Upload an Excel or CSV file, map columns, pick models and metrics, run, and export results as Excel, a LaTeX table, and a reproducibility config. `paperlab-core` is the library behind [PaperLab.space](https://paperlab.space), and you can use it directly from Python or the command line.

> **Status:** pre-alpha. See [PLAN.md](PLAN.md) for the roadmap.

## Planned features

- Excel (.xlsx, multi-sheet) and CSV input, with column mapping, validation, and JSONL conversion
- Unicode-safe handling for non-Latin scripts
- One interface to many providers via [LiteLLM](https://github.com/BerriAI/litellm) (OpenRouter, Gemini, OpenAI, Anthropic, Ollama, ...)
- Rate-limit-aware runner with retries and cost/time estimates
- Metrics: exact match, accuracy, precision/recall/F1, later BLEU, chrF, ROUGE, and LLM-as-judge
- Exports: per-row Excel, LaTeX comparison table, reproducibility config

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## License

[Apache 2.0](LICENSE)
